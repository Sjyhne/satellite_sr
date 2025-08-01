import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from pathlib import Path
import random
import argparse
import cv2
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchmetrics.functional import peak_signal_noise_ratio
import matplotlib.pyplot as plt
import glob
import json
from datetime import datetime

from data import get_dataset
from utils import bilinear_resize_torch, align_output_to_target, get_valid_mask
from losses import BasicLosses
from models.utils import get_decoder
from input_projections.utils import get_input_projection
from models.inr import INR
from models.nir import NIR, nir_loss

import os
import lpips
from torchmetrics.functional import structural_similarity_index_measure as ssim

def train_one_iteration(model, optimizer, train_sample, device, downsample_factor):
    model.train()
    
    # Initialize loss functions
    recon_criterion = BasicLosses.mse_loss
    trans_criterion = BasicLosses.mae_loss
    
    if model.use_gnll:
        recon_criterion = nn.GaussianNLLLoss()

    input = train_sample['input'].to(device)
    lr_target = train_sample['lr_target'].to(device)
    sample_id = train_sample['sample_id'].to(device)
    scale_factor = train_sample['scale_factor'].to(device)
    # Get ground truth shifts
    if 'shifts' in train_sample and 'dx_percent' in train_sample['shifts']:
        gt_dx = train_sample['shifts']['dx_percent'].to(device)
        gt_dy = train_sample['shifts']['dy_percent'].to(device)
    else:
        gt_dx = torch.zeros(lr_target.shape[0], device=device)
        gt_dy = torch.zeros(lr_target.shape[0], device=device)

    optimizer.zero_grad()

    if model.use_gnll:
        output, pred_shifts, pred_variance = model(input, sample_id, scale_factor=1/scale_factor, lr_frames=lr_target)
        recon_loss = recon_criterion(output, lr_target, pred_variance)
    else:
        if isinstance(model, INR):
            output, pred_shifts = model(input, sample_id, scale_factor=1/scale_factor, lr_frames=lr_target)
            recon_loss = recon_criterion(output, lr_target)
        elif isinstance(model, NIR):
            output, pred_shifts = model(input, sample_id, scale_factor=1/scale_factor, lr_frames=lr_target)
            recon_loss = nir_loss(output, lr_target)

    if isinstance(model, (INR, NIR)):
        pred_dx, pred_dy = pred_shifts
        trans_loss = trans_criterion(pred_dx, gt_dx) + trans_criterion(pred_dy, gt_dy)
    else:
        trans_loss = torch.zeros(1, device=device)

    # Only backpropagate the reconstruction loss
    recon_loss.backward()
    optimizer.step()
    
    return {
        'recon_loss': recon_loss.item(),
        'trans_loss': trans_loss.item(),
        'total_loss': recon_loss.item() + trans_loss.item()
    }


def test_one_epoch(model, test_loader, device):
    model.eval()
    
    with torch.no_grad():
        hr_coords = test_loader.get_hr_coordinates().unsqueeze(0).to(device)
        hr_image = test_loader.get_original_hr().unsqueeze(0).to(device)
        sample_id = torch.tensor([0]).to(device)
        
        if model.use_gnll:
            output, _, _ = model(hr_coords, sample_id, scale_factor=1, training=False)
        else:
            if isinstance(model, INR):
                output, _ = model(hr_coords, sample_id, scale_factor=1, training=False)
            elif isinstance(model, NIR):
                output, _ = model(hr_coords, sample_id, scale_factor=1, training=False, lr_frames=hr_image)
                output = output.reshape(hr_image.shape[1], hr_image.shape[2], 3).unsqueeze(0)

        # Unstandardize the output
        output = output * test_loader.get_lr_std(0).to(device) + test_loader.get_lr_mean(0).to(device)
        
        loss = F.mse_loss(output, hr_image)
        
        # Calculate PSNR
        psnr = -10 * torch.log10(loss)
        
    return loss.item(), psnr.item()


def main():
    parser = argparse.ArgumentParser(description="Minimal Satellite Super-Resolution Training")
    
    # Essential parameters only
    parser.add_argument("--dataset", type=str, default="satburst_synth", 
                       choices=["satburst_synth", "worldstrat", "burst_synth"])
    parser.add_argument("--sample_id", default="Landcover-743192_rgb")
    parser.add_argument("--df", type=int, default=4, help="Downsampling factor, or upsampling factor for the data")
    parser.add_argument("--scale_factor", type=float, default=4, help="scale factor for the input training grid")

    parser.add_argument("--lr_shift", type=float, default=1.0)
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--aug", type=str, default="none", choices=['none', 'light', 'medium', 'heavy'])
    
    # Model parameters
    parser.add_argument("--model", type=str, default="mlp", 
                       choices=["mlp", "siren", "wire", "linear", "conv", "thera", "nir"])
    parser.add_argument("--network_depth", type=int, default=4)
    parser.add_argument("--network_hidden_dim", type=int, default=256)
    parser.add_argument("--projection_dim", type=int, default=256)
    parser.add_argument("--input_projection", type=str, default="fourier_10", 
                       choices=["linear", "fourier_10", "fourier_5", "fourier_20", "fourier_40", "fourier", "legendre", "none"])
    parser.add_argument("--fourier_scale", type=float, default=10.0)
    parser.add_argument("--use_gnll", action="store_true")
    
    # Training parameters
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--iters", type=int, default=2000)
    parser.add_argument("--learning_rate", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--device", type=str, default="7", help="CUDA device number")
    
    args = parser.parse_args()

    # Setup device
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    
    # Set seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Parse input projection
    if args.input_projection.startswith("fourier_"):
        args.fourier_scale = float(args.input_projection.split("_")[1])
        args.input_projection = "fourier"

    # Setup dataset
    if args.dataset == "satburst_synth":
        # Check if we have an absolute path from the batch script
        if 'DATA_DIR_ABSOLUTE' in os.environ:
            args.root_satburst_synth = os.environ['DATA_DIR_ABSOLUTE']
        else:
            args.root_satburst_synth = f"data/{args.sample_id}/scale_{args.df}_shift_{args.lr_shift:.1f}px_aug_{args.aug}"

    train_data = get_dataset(args=args, name=args.dataset)
    train_dataloader = DataLoader(train_data, batch_size=2, shuffle=False)

    # Setup model
    input_projection = get_input_projection(args.input_projection, 2, args.projection_dim, device, args.fourier_scale)
    decoder = get_decoder(args.model, args.network_depth, args.projection_dim, args.network_hidden_dim)
    model = INR(input_projection, decoder, args.num_samples, use_gnll=args.use_gnll).to(device)
    # model = NIR(input_projection, decoder, args.num_samples, use_gnll=args.use_gnll).to(device)
    # Setup optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.iters, eta_min=1e-5)

    print(f"Starting training for {args.iters} iterations...")
    
    # Training loop
    iteration = 0
    progress_bar = tqdm(total=args.iters, desc="Training")
    
    # Lists to store PSNR and losses for plotting
    psnr_list = []
    recon_loss_list = []
    trans_loss_list = []
    total_loss_list = []
    iteration_list = []
    
    while iteration < args.iters:
        for train_sample in train_dataloader:
            if iteration >= args.iters:
                break
                
            # Train one iteration
            train_losses = train_one_iteration(model, optimizer, train_sample, device, args.df)
            scheduler.step()
            iteration += 1

            # Update progress bar
            progress_bar.update(1)
            progress_bar.set_postfix({
                'recon': f"{train_losses['recon_loss']:.4f}",
                'trans': f"{train_losses['trans_loss']:.4f}"
            })
            
            # Periodic evaluation
            if iteration % 100 == 0:
                test_loss, test_psnr = test_one_epoch(model, train_data, device)
                print(f"\nIter {iteration}: Train Loss: {train_losses['total_loss']:.6f}, "
                      f"Test Loss: {test_loss:.6f}, Test PSNR: {test_psnr:.2f} dB")

                # Append to lists for plotting
                iteration_list.append(iteration)
                psnr_list.append(test_psnr)
                recon_loss_list.append(train_losses['recon_loss'])
                trans_loss_list.append(train_losses['trans_loss'])
                total_loss_list.append(train_losses['total_loss'])

    progress_bar.close()
    
    # Final evaluation and save output
    model.eval()
    with torch.no_grad():
        hr_coords = train_data.get_hr_coordinates().unsqueeze(0).to(device)
        hr_image = train_data.get_original_hr().unsqueeze(0).to(device)
        sample_id = torch.tensor([0]).to(device)
        
        if model.use_gnll:
            output, _, _ = model(hr_coords, sample_id, scale_factor=1, training=False)
        else:
            if isinstance(model, INR):
                output, _ = model(hr_coords, sample_id, scale_factor=1, training=False)
            elif isinstance(model, NIR):
                output, _ = model(hr_coords, sample_id, scale_factor=1, training=False, lr_frames=hr_image)
                output = output.reshape(hr_image.shape[1], hr_image.shape[2], 3).unsqueeze(0)

        # Unstandardize the output
        output = output * train_data.get_lr_std(0).to(device) + train_data.get_lr_mean(0).to(device)
        
        final_test_loss = F.mse_loss(output, hr_image).item()   
        final_psnr = -10 * torch.log10(torch.tensor(final_test_loss)).item()
        
        # Convert tensors to numpy for saving as images
        pred_np = output.squeeze().cpu().numpy()
        gt_np = hr_image.squeeze().cpu().numpy()
        lr_original = train_data.get_lr_sample(0).cpu().numpy()
        lr_h, lr_w = lr_original.shape[:2]
        hr_h, hr_w = gt_np.shape[:2]
        lr_bilinear = cv2.resize(lr_original, (hr_w, hr_h), interpolation=cv2.INTER_LINEAR)
        pred_np = np.clip(pred_np, 0, 1)
        gt_np = np.clip(gt_np, 0, 1)
        lr_original = np.clip(lr_original, 0, 1)
        lr_bilinear = np.clip(lr_bilinear, 0, 1)

        # Convert numpy arrays to torch tensors for metrics
        pred_tensor = torch.from_numpy(pred_np).unsqueeze(0).permute(0, 3, 1, 2)  # [1, C, H, W]
        gt_tensor = torch.from_numpy(gt_np).unsqueeze(0).permute(0, 3, 1, 2)  # [1, C, H, W]
        bilinear_tensor = torch.from_numpy(lr_bilinear).unsqueeze(0).permute(0, 3, 1, 2)  # [1, C, H, W]

        # PSNR
        model_psnr = peak_signal_noise_ratio(pred_tensor, gt_tensor, data_range=1.0).item()
        bilinear_psnr = peak_signal_noise_ratio(bilinear_tensor, gt_tensor, data_range=1.0).item()

        # SSIM
        model_ssim = ssim(pred_tensor, gt_tensor, data_range=1.0).item()
        bilinear_ssim = ssim(bilinear_tensor, gt_tensor, data_range=1.0).item()

        # LPIPS (expects [-1,1] range)
        lpips_fn = lpips.LPIPS(net='vgg').to(device)
        pred_lpips = lpips_fn((pred_tensor*2-1).to(device), (gt_tensor*2-1).to(device)).item()
        bilinear_lpips = lpips_fn((bilinear_tensor*2-1).to(device), (gt_tensor*2-1).to(device)).item()

        # Save comparison figure with LR, bilinear upsampling, model output, and ground truth
        
        # Create comparison figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        
        # Original LR image
        axes[0, 0].imshow(lr_original)
        axes[0, 0].set_title('Original LR Image', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Bilinear upsampling
        axes[0, 1].imshow(lr_bilinear)
        axes[0, 1].set_title(f'Bilinear Upsampling\nPSNR: {bilinear_psnr:.2f} dB', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')
        
        # Model output
        axes[1, 0].imshow(pred_np)
        axes[1, 0].set_title(f'Model Output (Super-Resolution)\nPSNR: {model_psnr:.2f} dB', fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')
        
        # Ground truth
        axes[1, 1].imshow(gt_np)
        axes[1, 1].set_title('Ground Truth HR', fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')
        
        plt.tight_layout(pad=2.0)
        comparison_path = Path("super_resolution_comparison.png")
        plt.savefig(comparison_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close()
        
        # Also save individual images for reference
        plt.figure(figsize=(8, 8))
        plt.imshow(pred_np)
        plt.axis('off')
        plt.tight_layout(pad=0)
        pred_path = Path("super_resolution_output.png")
        plt.savefig(pred_path, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()
        
        plt.figure(figsize=(8, 8))
        plt.imshow(gt_np)
        plt.axis('off')
        plt.tight_layout(pad=0)
        gt_path = Path("ground_truth.png")
        plt.savefig(gt_path, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()
        
        output_path = comparison_path
        
    print(f"\nFinal Results:")
    print(f"Test Loss: {final_test_loss:.6f}")
    print(f"Test PSNR: {final_psnr:.2f} dB")
    print(f"Model PSNR: {model_psnr:.2f} dB")
    print(f"Bilinear PSNR: {bilinear_psnr:.2f} dB")
    print(f"PSNR Improvement: {model_psnr - bilinear_psnr:.2f} dB")
    print(f"Model output saved to {output_path}")
    
    # Save PSNR results to a text file
    results_text = f"""Super-Resolution Results
=======================

Dataset: {args.dataset}
Sample ID: {args.sample_id}
Downsampling Factor: {args.df}
Model: {args.model}
Iterations: {args.iters}

PSNR Results:
- Model Output: {model_psnr:.2f} dB
- Bilinear Interpolation: {bilinear_psnr:.2f} dB
- PSNR Improvement: {model_psnr - bilinear_psnr:.2f} dB

SSIM Results:
- Model Output: {model_ssim:.4f}
- Bilinear Interpolation: {bilinear_ssim:.4f}
- SSIM Improvement: {model_ssim - bilinear_ssim:.4f}

LPIPS Results:
- Model Output: {pred_lpips:.4f}
- Bilinear Interpolation: {bilinear_lpips:.4f}
- LPIPS Improvement: {bilinear_lpips - pred_lpips:.4f}

Training Results:
- Final Test Loss: {final_test_loss:.6f}
- Final Test PSNR: {final_psnr:.2f} dB
- Final Reconstruction Loss: {recon_loss_list[-1] if recon_loss_list else 0:.6f}
- Final Transformation Loss: {trans_loss_list[-1] if trans_loss_list else 0:.6f}
- Final Total Loss: {total_loss_list[-1] if total_loss_list else 0:.6f}

Training Metrics History:
"""
    
    if len(psnr_list) > 0:
        results_text += f"- Number of evaluation points: {len(psnr_list)}\n"
        results_text += f"- PSNR range: {min(psnr_list):.2f} - {max(psnr_list):.2f} dB\n"
        results_text += f"- Reconstruction loss range: {min(recon_loss_list):.6f} - {max(recon_loss_list):.6f}\n"
        results_text += f"- Transformation loss range: {min(trans_loss_list):.6f} - {max(trans_loss_list):.6f}\n"
        results_text += f"- Total loss range: {min(total_loss_list):.6f} - {max(total_loss_list):.6f}\n"
        results_text += f"- Final PSNR: {psnr_list[-1]:.2f} dB\n"
        results_text += f"- Final reconstruction loss: {recon_loss_list[-1]:.6f}\n"
        results_text += f"- Final transformation loss: {trans_loss_list[-1]:.6f}\n"
        results_text += f"- Final total loss: {total_loss_list[-1]:.6f}\n"
    else:
        results_text += "- No training metrics recorded (training may have been too short)\n"
    
    with open("psnr_results.txt", "w") as f:
        f.write(results_text)
    
    print(f"PSNR results saved to psnr_results.txt")

    # Plot PSNR and all losses
    if len(psnr_list) > 0:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot PSNR on top subplot
        ax1.plot(iteration_list, psnr_list, color='blue', linewidth=2, label='PSNR (Test)')
        ax1.set_xlabel('Iteration', fontsize=12)
        ax1.set_ylabel('PSNR (dB)', fontsize=12)
        ax1.set_title('Training PSNR Evolution', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot all losses on bottom subplot
        ax2.plot(iteration_list, recon_loss_list, color='red', linewidth=2, label='Reconstruction Loss')
        ax2.plot(iteration_list, trans_loss_list, color='green', linewidth=2, label='Transformation Loss')
        ax2.plot(iteration_list, total_loss_list, color='purple', linewidth=2, label='Total Loss')
        ax2.set_xlabel('Iteration', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.set_title('Training Loss Evolution', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig("training_metrics.png", bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close()
        
        print(f"Training metrics plot saved to training_metrics.png")
    else:
        print("No metrics data available for plotting (training may have been too short)")


if __name__ == "__main__":
    main() 