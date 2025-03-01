import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import random
import wandb  # Add at the top with other imports
import pandas as pd  # Add to imports
from torch.optim.lr_scheduler import CosineAnnealingLR  # Change import
import lpips  # Add at the top
from torchmetrics.functional import structural_similarity_index_measure as ssim
from torch.utils.data import DataLoader
from data import get_dataset
import cv2
from utils import apply_shift_torch, bilinear_resize_torch, align_output_to_target, get_valid_mask
from coordinate_based_mlp import FourierNetwork
from losses import BasicLosses, AdvancedLosses, CharbonnierLoss, RelativeLosses

import einops
import argparse

import torch.fft


def visualize_masked_images(output, target, mask, iteration, save_dir='./mask_vis'):
    """Visualize the masked output and target images."""
    # Create absolute path
    save_dir = Path(save_dir).absolute()
    save_dir.mkdir(exist_ok=True, parents=True)  # Add parents=True to create all necessary directories
    
    print(f"Saving masked images to: {save_dir}")  # Debug print
    
    # Convert tensors to numpy arrays
    output = output.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    mask = mask.detach().cpu().numpy()
    
    # Create figure with subplots for each sample in batch
    B = output.shape[0]
    fig, axes = plt.subplots(B, 4, figsize=(16, 4*B))
    
    if B == 1:  # Handle single sample case
        axes = axes.reshape(1, -1)
    
    for i in range(B):
        # Original output
        axes[i, 0].imshow(output[i])
        axes[i, 0].set_title(f'Output {i}')
        axes[i, 0].axis('off')
        
        # Masked output
        axes[i, 1].imshow(output[i] * mask[i])
        axes[i, 1].set_title(f'Masked Output {i}')
        axes[i, 1].axis('off')
        
        # Original target
        axes[i, 2].imshow(target[i])
        axes[i, 2].set_title(f'Target {i}')
        axes[i, 2].axis('off')
        
        # Masked target
        axes[i, 3].imshow(target[i] * mask[i])
        axes[i, 3].set_title(f'Masked Target {i}')
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    save_path = save_dir / f'masked_images_iter_{iteration}.png'
    plt.savefig(save_path)
    plt.close()

def train_one_iteration(model, optimizer, train_sample, device, iteration=0, use_gt=False):
    model.train()

    # Initialize loss functions
    recon_criterion = BasicLosses.mae_loss  # Use MSE loss instead of GaussianNLLLoss
    trans_criterion = BasicLosses.mae_loss

    epoch_recon_loss = 0.0
    epoch_trans_loss = 0.0

    pred_dxs = []
    pred_dys = []
    gt_dxs = []
    gt_dys = []

    input = train_sample['input'].to(device)
    lr_target = train_sample['lr_target'].to(device)
    sample_id = train_sample['sample_id'].to(device)

    if 'shifts' in train_sample and 'dx_percent' in train_sample['shifts']:
        gt_dx = train_sample['shifts']['dx_percent'].to(device)
        gt_dy = train_sample['shifts']['dy_percent'].to(device)
    else:
        gt_dx = torch.zeros(lr_target.shape[0], device=device)
        gt_dy = torch.zeros(lr_target.shape[0], device=device)

    optimizer.zero_grad()

    output, pred_shifts = model(input, sample_id)
    
    # Downsample to match target resolution
    output = bilinear_resize_torch(output.permute(0, 3, 1, 2), (lr_target.shape[1], lr_target.shape[2])).permute(0, 2, 3, 1)

    # Calculate reconstruction loss - ensure it's a scalar
    recon_loss = recon_criterion(output, lr_target)
    
    pred_dx, pred_dy = pred_shifts
    trans_loss = trans_criterion(pred_dx, gt_dx) + trans_criterion(pred_dy, gt_dy)
    
    # Only backpropagate the reconstruction loss
    recon_loss.backward()
    optimizer.step()
    
    return {
        'recon_loss': recon_loss.item(),
        'trans_loss': trans_loss.item(),
        'total_loss': recon_loss.item() + trans_loss.item(),
        'pred_dx': pred_dx.detach().cpu().numpy(),
        'pred_dy': pred_dy.detach().cpu().numpy(),
        'gt_dx': gt_dx.detach().cpu().numpy(),
        'gt_dy': gt_dy.detach().cpu().numpy()
    }

def test_one_epoch(model, test_loader, device):
    model.eval()

    # Get HR features from the test loader
    hr_coords = test_loader.get_hr_coordinates().unsqueeze(0).to(device)
    hr_image = test_loader.get_original_hr().unsqueeze(0).to(device)
    sample_id = torch.tensor([0]).to(device)
    
    output, _ = model(hr_coords, sample_id)

    loss = F.mse_loss(output, hr_image)
    
    return loss.item(), output.detach(), hr_image.detach()



def visualize_translations(pred_dx, pred_dy, target_dx, target_dy, save_path='translation_vis.png'):
    """Create a simple visualization comparing predicted and target translations."""
    plt.figure(figsize=(10, 8))
    
    # Set up the plot with a light gray background grid
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    
    # Plot target points (blue circles)
    plt.scatter(target_dx, target_dy, c='blue', s=90, label='Target', alpha=0.7)
    
    # Plot predicted points (red x's)
    plt.scatter(pred_dx, pred_dy, c='red', s=90, label='Predicted', marker='x', alpha=0.7)
    
    # Connect corresponding points with gray lines
    for i in range(len(target_dx)):
        plt.plot([target_dx[i], pred_dx[i]], [target_dy[i], pred_dy[i]], 'gray', alpha=0.5)
        
        # Add sample number next to the target point with larger font and background
        plt.text(target_dx[i], target_dy[i], f' {i}', 
                fontsize=10,  # Increased font size
                fontweight='normal'  # Make the text bold
                )
    
    # Add labels and title
    plt.xlabel('X Translation')
    plt.ylabel('Y Translation')
    plt.title('Target vs. Predicted Translations')
    plt.legend(loc='upper right')
    
    # Make sure the aspect ratio is equal
    plt.axis('equal')
    
    # Add a bit of padding around the points
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(save_path, dpi=150)
    plt.close()

def calculate_metrics(pred, target):
    """Calculate multiple image quality metrics.
    
    Args:
        pred: Predicted image tensor [B, C, H, W]
        target: Target image tensor [B, C, H, W]
        
    Returns:
        Dictionary of metrics
    """
    # Ensure inputs are in correct format
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
    if target.dim() == 3:
        target = target.unsqueeze(0)
    
    # Move to same device as inputs
    loss_fn_alex = lpips.LPIPS(net='alex').to(pred.device)
    
    # Calculate metrics
    mse = F.mse_loss(pred, target)
    psnr = -10 * torch.log10(mse)
    lpips_value = loss_fn_alex(pred, target).mean()
    ssim_value = ssim(pred, target)
    
    return {
        'mse': mse.item(),
        'psnr': psnr.item(),
        'lpips': lpips_value.item(),
        'ssim': ssim_value.item()
    }

def plot_training_curves(history, save_path='training_curves.png'):
    """Plot training curves including losses and all metrics."""
    plt.figure(figsize=(15, 20))
    
    # Plot losses
    plt.subplot(5, 1, 1)
    plt.plot(history['iterations'], history['recon_loss'], label='Reconstruction Loss')
    plt.plot(history['iterations'], history['trans_loss'], label='Transform Loss')
    plt.plot(history['iterations'], history['test_loss'], label='Test Loss')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title('Training Losses')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')

    # Plot PSNR
    plt.subplot(5, 1, 2)
    plt.plot(history['iterations'], history['psnr'], label='Model PSNR')
    plt.axhline(y=history['baseline_psnr'][-1], color='r', linestyle='--', 
                label=f'Baseline PSNR: {history["baseline_psnr"][-1]:.2f}dB')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title('PSNR')
    plt.xlabel('Iteration')
    plt.ylabel('PSNR (dB)')

    # Plot LPIPS
    plt.subplot(5, 1, 3)
    plt.plot(history['iterations'], history['lpips'], label='Model LPIPS')
    plt.axhline(y=history['baseline_lpips'][-1], color='r', linestyle='--', 
                label=f'Baseline LPIPS: {history["baseline_lpips"][-1]:.4f}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title('LPIPS (lower is better)')
    plt.xlabel('Iteration')
    plt.ylabel('LPIPS')

    # Plot SSIM
    plt.subplot(5, 1, 4)
    plt.plot(history['iterations'], history['ssim'], label='Model SSIM')
    plt.axhline(y=history['baseline_ssim'][-1], color='r', linestyle='--', 
                label=f'Baseline SSIM: {history["baseline_ssim"][-1]:.4f}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title('SSIM (higher is better)')
    plt.xlabel('Iteration')
    plt.ylabel('SSIM')

    # Plot learning rates
    plt.subplot(5, 1, 5)
    plt.plot(history['iterations'], history['learning_rate'], label='Learning Rate')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title('Learning Rate')
    plt.xlabel('Iteration')
    plt.ylabel('Learning Rate')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=str, default="2")
    parser.add_argument("--model", type=str, default="FourierNetwork")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--df", type=int, default=4)
    parser.add_argument("--lr_shift", type=float, default=0.5)
    parser.add_argument("--bs", type=int, default=2)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--use_gt", type=bool, default=False)
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--aug", type=str, default="none", 
                       choices=['none', 'light', 'medium', 'heavy'],
                       help="Augmentation level to use")
    # Add dataset argument
    parser.add_argument("--dataset", type=str, default="satburst_synth",
                        choices=["satburst_synth", "worldstrat", "burst_synth"],
                        help="Dataset implemented in data.get_dataset()")
    parser.add_argument("--root_burst_synth", default="~/data/burst_synth", help="Set root of burst_synth")
    parser.add_argument("--root_satburst_synth", default="~/data/satburst_synth", help="Set root of worldstrat dataset")
    parser.add_argument("--root_worldstrat", default="~/data/worldstrat_kaggle", help="Set root of worldstrat dataset")
    parser.add_argument("--area_name", type=str, default="UNHCR-LBNs006446", help="str: a sample name of worldstrat dataset")
    parser.add_argument("--worldstrat_hr_size", type=int, default=None, help="int: Default size is 1054")
    parser.add_argument("--sample_id", type=int, default="1", help="int: a sample index of burst_synth")


    args = parser.parse_args()

    # set torch cuda device. note: do not set default_device to cuda:0
    if torch.cuda.is_available():
        torch.cuda.set_device(f"cuda:{args.d}")  
    device = torch.device(f"cuda:{args.d}" if torch.cuda.is_available() else "cpu")
    
    # Create base results directory
    Path('results').mkdir(exist_ok=True)

    # Set all seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # For completely reproducible results
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    network_size = (4, 256)
    learning_rate = 5e-3
    iters = args.iters
    mapping_size = 128
    downsample_factor = args.df
    lr_shift = args.lr_shift
    num_samples = args.samples

    # Create results directory based on dataset parameters
    results_dir = Path(f"results/lr_factor_{downsample_factor}x_shift_{lr_shift:.1f}px_samples_{num_samples}_aug_{args.aug}")
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving results to: {results_dir}")

    # Load the dataset
    if args.dataset == "satburst_synth":
        args.root_satburst_synth = f"data/lr_factor_{downsample_factor}x_shift_{lr_shift:.1f}px_samples_{num_samples}_aug_{args.aug}"

    train_data = get_dataset(args=args, name=args.dataset)
    
    batch_size = args.bs

    # initialize the dataloader here
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    model = FourierNetwork(mapping_size, *network_size, len(train_data), rggb=False).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=iters, eta_min=1e-6)

    # Initialize history dictionary with new metrics
    history = {
        'iterations': [],
        'recon_loss': [],
        'trans_loss': [],
        'test_loss': [],
        'psnr': [],
        'lpips': [],
        'ssim': [],
        'baseline_psnr': [],
        'baseline_lpips': [],
        'baseline_ssim': [],
        'translation_data': [],
        'learning_rate': []  # Changed from recon_lr and trans_lr to single learning_rate
    }

    # Initialize wandb if enabled
    if args.wandb:
        dataset_name = f"lr_factor_{downsample_factor}x_shift_{lr_shift:.1f}px_samples_{num_samples}_aug_{args.aug}"
        wandb.init(
            project="satellite-super-res",
            name=dataset_name,
            group=args.model,  # Group runs by model type
            config={
                "model": args.model,
                "downsample_factor": downsample_factor,
                "lr_shift": lr_shift,
                "num_samples": num_samples,
                "iters": iters,
                "network_size": network_size,
                "learning_rate": learning_rate,
                "mapping_size": mapping_size,
                "use_gt": args.use_gt,
                "augmentation": args.aug
            }
        )

    # Initialize LPIPS model once
    loss_fn_alex = lpips.LPIPS(net='alex').to(device)

    def calculate_metrics(pred, target):
        """Calculate multiple image quality metrics."""
        # Ensure inputs are in correct format
        if pred.dim() == 3:
            pred = pred.unsqueeze(0)
        if target.dim() == 3:
            target = target.unsqueeze(0)
        
        # Calculate metrics using the shared LPIPS model
        mse = F.mse_loss(pred, target)
        psnr = -10 * torch.log10(mse)
        lpips_value = loss_fn_alex(pred, target).mean()
        ssim_value = ssim(pred, target)
        
        return {
            'mse': mse.item(),
            'psnr': psnr.item(),
            'lpips': lpips_value.item(),
            'ssim': ssim_value.item()
        }

    i = 0
    aggregate_train_losses = {}
    
    # Create a tqdm progress bar for the entire training
    progress_bar = tqdm(total=iters, desc="Training")
    
    while i < iters:
        for train_sample in train_dataloader:
            train_losses = train_one_iteration(model, optimizer, train_sample, device, iteration=i+1, use_gt=args.use_gt)
            scheduler.step()
            i += 1

            for key, value in train_losses.items():
                if key not in aggregate_train_losses:
                    aggregate_train_losses[key] = []
                aggregate_train_losses[key].append(value)
            
            # Update the progress bar
            progress_bar.update(1)
            progress_bar.set_postfix({
                'recon_loss': f"{train_losses['recon_loss']:.4f}",
                'trans_loss': f"{train_losses['trans_loss']:.4f}"
            })
            
            
            
            if (i + 1) % 40 == 0:  # More frequent logging
                with torch.no_grad():
                    test_loss, test_output, hr_image = test_one_epoch(model, train_data, device)

                    print("ff_scale:", model.ff_scale.item())

                    test_output = einops.rearrange(test_output, 'b h w c -> b c h w')
                    hr_image = einops.rearrange(hr_image, 'b h w c -> b c h w')
                    lr_sample = train_data.get_lr_sample(0).unsqueeze(0).to(device)
                    baseline_pred = bilinear_resize_torch(lr_sample, (hr_image.shape[2], hr_image.shape[3]))

                    # align predictions and targets spectrally (no spatial alignment while optimizing)
                    test_output = align_output_to_target(input=test_output, reference=hr_image, spatial=False)
                    baseline_pred = align_output_to_target(input=baseline_pred, reference=hr_image, spatial=False)

                    # Calculate metrics for model output
                    baseline_metrics = calculate_metrics(baseline_pred, hr_image)
                    model_metrics = calculate_metrics(test_output, hr_image)
                    


                
                # Store values in history
                history['iterations'].append(i + 1)
                history['recon_loss'].append(aggregate_train_losses['recon_loss'][-1])
                history['trans_loss'].append(aggregate_train_losses['trans_loss'][-1])
                history['test_loss'].append(test_loss)
                history['psnr'].append(model_metrics['psnr'])
                history['lpips'].append(model_metrics['lpips'])
                history['ssim'].append(model_metrics['ssim'])
                history['baseline_psnr'].append(baseline_metrics['psnr'])
                history['baseline_lpips'].append(baseline_metrics['lpips'])
                history['baseline_ssim'].append(baseline_metrics['ssim'])

                translation_data = []
                for idx in range(len(train_data)):
                    gt_shift = train_data[idx]['shifts']
                    pred_shift = model.transform_vectors[idx]

                    gt_dx = gt_shift["dx_percent"]
                    gt_dy = gt_shift["dy_percent"]

                    pred_dy = pred_shift[1]
                    pred_dx = pred_shift[0]

                    translation_data.append({
                        'target_dx': gt_dx,
                        'target_dy': gt_dy,
                        'pred_dx': pred_dx.detach().cpu().numpy(),
                        'pred_dy': pred_dy.detach().cpu().numpy(),
                    })

                # Store all translations for this iteration
                history['translation_data'].append(translation_data)

                # Store learning rates in history
                history['learning_rate'].append(scheduler.get_last_lr()[0])

                # Print all metrics
                print(f"Iter {i+1}: "
                    f"Train recon: {aggregate_train_losses['recon_loss'][-1]:.6f}, "
                    f"trans: {aggregate_train_losses['trans_loss'][-1]:.6f}, "
                    f"total: {aggregate_train_losses['total_loss'][-1]:.6f}, "
                    f"Test: {test_loss:.6f}\n"
                    f"Metrics vs Baseline:\n"
                    f"PSNR: {model_metrics['psnr']:.2f}dB vs {baseline_metrics['psnr']:.2f}dB\n"
                    f"LPIPS: {model_metrics['lpips']:.4f} vs {baseline_metrics['lpips']:.4f} (lower is better)\n"
                    f"SSIM: {model_metrics['ssim']:.4f} vs {baseline_metrics['ssim']:.4f} (higher is better)")

                if args.wandb:
                    wandb.log({
                        "iteration": i + 1,
                        "train/recon_loss": aggregate_train_losses['recon_loss'][-1],
                        "train/trans_loss": aggregate_train_losses['trans_loss'][-1],
                        "train/total_loss": aggregate_train_losses['total_loss'][-1],
                        "test/loss": test_loss,
                        "test/psnr": model_metrics['psnr'],
                        "test/lpips": model_metrics['lpips'],
                        "test/ssim": model_metrics['ssim'],
                        "test/baseline_psnr": baseline_metrics['psnr'],
                        "test/baseline_lpips": baseline_metrics['lpips'],
                        "test/baseline_ssim": baseline_metrics['ssim'],
                        "learning_rate": scheduler.get_last_lr()[0],
                        "metrics/psnr_improvement": model_metrics['psnr'] - baseline_metrics['psnr'],
                        "metrics/lpips_improvement": baseline_metrics['lpips'] - model_metrics['lpips'],
                        "metrics/ssim_improvement": model_metrics['ssim'] - baseline_metrics['ssim'],
                    })

                # Update progress bar description with current metrics
                progress_bar.set_description(
                    f"Train: {aggregate_train_losses['recon_loss'][-1]:.4f} | "
                    f"PSNR: {model_metrics['psnr']:.2f}dB"
                )
                
                aggregate_train_losses = {}

            # Break if we've reached the desired number of iterations
            if i >= iters:
                break
    
    # Close the progress bar
    progress_bar.close()

    # Create all visualizations at the end
    # Training curves
    plot_training_curves(history, save_path=results_dir / 'final_training_curves.png')
    
    # Translation visualization (using last iteration's data)
    last_trans_data = history['translation_data'][-1]  # Get the last iteration's data

    # Extract all translation data
    pred_dx_list = []
    pred_dy_list = []
    target_dx_list = []
    target_dy_list = []

    for sample_data in last_trans_data:
        # Extract values and convert to float
        pred_dx_list.append(float(sample_data['pred_dx']))
        pred_dy_list.append(float(sample_data['pred_dy']))
        target_dx_list.append(float(sample_data['target_dx']))
        target_dy_list.append(float(sample_data['target_dy']))

    # Convert to numpy arrays
    pred_dx_array = np.array(pred_dx_list)
    pred_dy_array = np.array(pred_dy_list)
    target_dx_array = np.array(target_dx_list)
    target_dy_array = np.array(target_dy_list)

    # Visualize translations
    visualize_translations(
        pred_dx_array,
        pred_dy_array,
        target_dx_array,
        target_dy_array,
        save_path=results_dir / 'final_translation_vis.png'
    )


    # Create downsampled then upsampled version for comparison
    with torch.no_grad():

        # Final test and visualization
        test_loss, test_output, hr_image = test_one_epoch(model, train_data, device)
        lr_sample = train_data.get_lr_sample(0).unsqueeze(0).to(device)

        test_output = einops.rearrange(test_output, 'b h w c -> b c h w')
        hr_image = einops.rearrange(hr_image, 'b h w c -> b c h w')

        baseline_pred = bilinear_resize_torch(lr_sample, (hr_image.shape[2], hr_image.shape[3]))

        # align predictions and targets
        test_output = align_output_to_target(input=test_output, reference=hr_image)
        baseline_pred = align_output_to_target(input=baseline_pred, reference=hr_image)        

        # Calculate PSNR for model output
        valid_mask = get_valid_mask(test_output, hr_image)
        print("valid_mask:", valid_mask.shape)
        mse_model = BasicLosses.mse_loss(test_output, hr_image, mask=valid_mask)
        psnr_model = -10 * torch.log10(mse_model)
        
        # Calculate PSNR for baseline
        valid_mask = get_valid_mask(baseline_pred, hr_image)
        print("valid_mask:", valid_mask.shape)
        mse_baseline = BasicLosses.mse_loss(baseline_pred, hr_image, mask=valid_mask)
        psnr_baseline = -10 * torch.log10(mse_baseline)
    
    # Get LR target image (use sample_00 as it matches the HR ground truth)
    lr_target_img = train_data.get_lr_sample(0)  # Get sample_00
    if torch.is_tensor(lr_target_img):
        # First, ensure the image is in the right format
        # If it's [C, H, W], convert to [H, W, C]
        if lr_target_img.shape[0] == 4 or lr_target_img.shape[0] == 3:
            # It's in [C, H, W] format, convert to [H, W, C]
            lr_target_img = lr_target_img.permute(1, 2, 0)
        
        # Now check if it's a RAW Bayer image with 4 channels
        if lr_target_img.shape[2] == 4:
            # Convert RGGB to RGB for visualization
            R = lr_target_img[..., 0]
            G = (lr_target_img[..., 1] + lr_target_img[..., 2]) / 2
            B = lr_target_img[..., 3]
            # Stack to create RGB image
            lr_target_img = torch.stack([R, G, B], dim=-1)
        
        # Convert to numpy for plotting
        lr_target_img = lr_target_img.numpy()
    
    # Convert tensors to numpy arrays for visualization
    hr_image = hr_image.cpu().permute(0, 2, 3, 1).numpy()
    baseline_pred = baseline_pred.cpu().permute(0, 2, 3, 1).numpy()
    test_output = test_output.cpu().permute(0, 2, 3, 1).numpy()

    # Create side by side visualization
    plt.figure(figsize=(24, 6))
    
    plt.subplot(1, 4, 1)
    plt.imshow(hr_image[0])
    plt.title('HR GT')
    plt.axis('off')
    
    plt.subplot(1, 4, 2)
    plt.imshow(lr_target_img)
    plt.title('LR Reference (Sample 00)')
    plt.axis('off')
    
    plt.subplot(1, 4, 3)
    plt.imshow(baseline_pred[0])
    plt.title(f'Bilinear\nPSNR: {psnr_baseline:.2f} dB')
    plt.axis('off')
    
    plt.subplot(1, 4, 4)
    plt.imshow(test_output[0])
    plt.title(f'Pred\nPSNR: {psnr_model:.2f} dB')
    plt.axis('off')
    
    plt.suptitle(f'Test loss: {test_loss:.6f} | Model PSNR: {psnr_model:.2f} dB\nBaseline PSNR: {psnr_baseline:.2f} dB | Dataset scale: {downsample_factor}x\nIterations: {iters} | learning_rate: {scheduler.get_last_lr()[0]}\nModel size: {network_size} | mapping_size: {mapping_size}')
    plt.tight_layout()
    plt.savefig(results_dir / 'comparison.png')
    plt.close()

    # At the end, log final images
    if args.wandb:
        # Log the final comparison image
        wandb.log({
            "final_comparison": wandb.Image(str(results_dir / 'comparison.png')),
            "final_translation_vis": wandb.Image(str(results_dir / 'final_translation_vis.png')),
            "final_training_curves": wandb.Image(str(results_dir / 'final_training_curves.png')),
        })
        
        # If we have mask visualization
        mask_vis_path = results_dir / 'final_mask_vis' / 'masked_images_iter_final.png'
        if mask_vis_path.exists():
            wandb.log({"final_mask_vis": wandb.Image(str(mask_vis_path))})
        
        wandb.finish()

    # Store final metrics in a dictionary
    final_metrics = {
        'downsampling_factor': downsample_factor,
        'lr_shift': lr_shift,
        'num_samples': num_samples,
        'model': args.model,
        'iterations': iters,
        'final_recon_loss': train_losses['recon_loss'],
        'final_trans_loss': train_losses['trans_loss'],
        'final_total_loss': train_losses['total_loss'],
        'final_test_loss': test_loss,
        'final_psnr': model_metrics['psnr'],
        'final_lpips': model_metrics['lpips'],
        'final_ssim': model_metrics['ssim'],
        'final_baseline_psnr': baseline_metrics['psnr'],
        'final_baseline_lpips': baseline_metrics['lpips'],
        'final_baseline_ssim': baseline_metrics['ssim'],
        'psnr_improvement': model_metrics['psnr'] - baseline_metrics['psnr'],
        'lpips_improvement': baseline_metrics['lpips'] - model_metrics['lpips'],
        'ssim_improvement': model_metrics['ssim'] - baseline_metrics['ssim'],
    }

    # Save metrics to CSV for this experiment
    metrics_df = pd.DataFrame([final_metrics])
    metrics_df.to_csv(results_dir / 'metrics.csv', index=False)

    # Try to update the master results file
    master_results_path = Path('results/all_experiments.xlsx')
    try:
        if master_results_path.exists():
            master_df = pd.read_excel(master_results_path)
            master_df = pd.concat([master_df, metrics_df], ignore_index=True)
        else:
            master_df = metrics_df
        
        # Sort by factor, shift, and samples for easier reading
        master_df = master_df.sort_values(['downsampling_factor', 'lr_shift', 'num_samples'])
        
        # Save with some nice formatting
        with pd.ExcelWriter(master_results_path, engine='openpyxl') as writer:
            master_df.to_excel(writer, index=False, sheet_name='Results')
            
            # Auto-adjust column widths
            worksheet = writer.sheets['Results']
            for idx, col in enumerate(master_df.columns):
                max_length = max(
                    master_df[col].astype(str).apply(len).max(),
                    len(str(col))
                )
                worksheet.column_dimensions[chr(65 + idx)].width = max_length + 2

    except Exception as e:
        print(f"Warning: Could not update master results file: {e}")
        print("Individual results still saved to CSV in experiment directory")

if __name__ == "__main__":
    main()