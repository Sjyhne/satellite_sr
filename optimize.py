import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from pathlib import Path
import random
import argparse
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from data import get_dataset
from utils import bilinear_resize_torch, align_output_to_target, get_valid_mask
from losses import BasicLosses
from models.utils import get_decoder
from input_projections.utils import get_input_projection
from models.inr import INR


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

    # Get ground truth shifts
    if 'shifts' in train_sample and 'dx_percent' in train_sample['shifts']:
        gt_dx = train_sample['shifts']['dx_percent'].to(device)
        gt_dy = train_sample['shifts']['dy_percent'].to(device)
    else:
        gt_dx = torch.zeros(lr_target.shape[0], device=device)
        gt_dy = torch.zeros(lr_target.shape[0], device=device)

    optimizer.zero_grad()

    if model.use_gnll:
        output, pred_shifts, pred_variance = model(input, sample_id, scale_factor=1/downsample_factor, lr_frames=lr_target)
        recon_loss = recon_criterion(output, lr_target, pred_variance)
    else:
        output, pred_shifts = model(input, sample_id, scale_factor=1/downsample_factor, lr_frames=lr_target)
        recon_loss = recon_criterion(output, lr_target)

    pred_dx, pred_dy = pred_shifts
    trans_loss = trans_criterion(pred_dx, gt_dx) + trans_criterion(pred_dy, gt_dy)

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
            output, _ = model(hr_coords, sample_id, scale_factor=1, training=False)

        # Unstandardize the output
        output = output * test_loader.get_lr_std(0).cuda() + test_loader.get_lr_mean(0).cuda()
        
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
    parser.add_argument("--df", type=int, default=4, help="Downsampling factor")
    parser.add_argument("--lr_shift", type=float, default=1.0)
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--aug", type=str, default="none", choices=['none', 'light', 'medium', 'heavy'])
    
    # Model parameters
    parser.add_argument("--model", type=str, default="mlp", 
                       choices=["mlp", "siren", "wire", "linear", "conv", "thera"])
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
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--device", type=str, default="0", help="CUDA device number")
    
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
        args.root_satburst_synth = f"data/{args.sample_id}/scale_{args.df}_shift_{args.lr_shift:.1f}px_aug_{args.aug}"

    train_data = get_dataset(args=args, name=args.dataset)
    train_dataloader = DataLoader(train_data, batch_size=1, shuffle=False)

    # Setup model
    input_projection = get_input_projection(args.input_projection, 2, args.projection_dim, device, args.fourier_scale)
    decoder = get_decoder(args.model, args.network_depth, args.projection_dim, args.network_hidden_dim)
    model = INR(input_projection, decoder, args.num_samples, use_gnll=args.use_gnll).to(device)

    # Setup optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.iters, eta_min=1e-6)

    print(f"Starting training for {args.iters} iterations...")
    
    # Training loop
    iteration = 0
    progress_bar = tqdm(total=args.iters, desc="Training")
    
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
            output, _ = model(hr_coords, sample_id, scale_factor=1, training=False)

        # Unstandardize the output
        output = output * train_data.get_lr_std(0).cuda() + train_data.get_lr_mean(0).cuda()
        
        final_test_loss = F.mse_loss(output, hr_image).item()
        final_psnr = -10 * torch.log10(torch.tensor(final_test_loss)).item()
        
        # Save the output as PNG
        import matplotlib.pyplot as plt
        
        # Convert tensors to numpy for saving as images
        pred_np = output.squeeze().cpu().numpy()
        gt_np = hr_image.squeeze().cpu().numpy()
        
        # Clamp values to [0, 1] range
        pred_np = np.clip(pred_np, 0, 1)
        gt_np = np.clip(gt_np, 0, 1)
        
        # Save prediction
        plt.figure(figsize=(8, 8))
        plt.imshow(pred_np)
        plt.axis('off')
        plt.tight_layout(pad=0)
        pred_path = Path("super_resolution_output.png")
        plt.savefig(pred_path, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()
        
        # Save ground truth for comparison
        plt.figure(figsize=(8, 8))
        plt.imshow(gt_np)
        plt.axis('off')
        plt.tight_layout(pad=0)
        gt_path = Path("ground_truth.png")
        plt.savefig(gt_path, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()
        
        output_path = pred_path
        
    print(f"\nFinal Results:")
    print(f"Test Loss: {final_test_loss:.6f}")
    print(f"Test PSNR: {final_psnr:.2f} dB")
    print(f"Model output saved to {output_path}")


if __name__ == "__main__":
    main() 