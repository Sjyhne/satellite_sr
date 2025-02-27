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
from data import SRData, SyntheticBurstVal
import cv2
from utils import apply_shift_torch, downsample_torch
from coordinate_based_mlp import FourierNetwork
from losses import BasicLosses

import argparse


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

def train_one_epoch(model, optimizer, train_loader, device, iteration=0, use_gt=False):
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

    for sample in train_loader:
        input = sample['input'].to(device)
        lr_target = sample['lr_target'].to(device)
        sample_id = sample['sample_id'].to(device)
        
        # Handle the case where shifts might not be present in the dataset
        if 'shifts' in sample and 'dx_percent' in sample['shifts']:
            gt_dx = sample['shifts']['dx_percent'].to(device)
            gt_dy = sample['shifts']['dy_percent'].to(device)
        else:
            # Default to zeros if shifts aren't available
            gt_dx = torch.zeros(lr_target.shape[0], device=device)
            gt_dy = torch.zeros(lr_target.shape[0], device=device)

        optimizer.zero_grad()

        output, pred_shifts = model(input, sample_id)
        
        # Extract variance and output
        # variance = torch.exp(output[..., -1:])
        # output = output[..., :-1]
        
        # Downsample to match target resolution
        output = downsample_torch(output.permute(0, 3, 1, 2), (lr_target.shape[1], lr_target.shape[2])).permute(0, 2, 3, 1)
        # variance = downsample_torch(variance.permute(0, 3, 1, 2), (lr_target.shape[1], lr_target.shape[2])).permute(0, 2, 3, 1)

        # Calculate reconstruction loss - ensure it's a scalar
        recon_loss = recon_criterion(output, lr_target)
        
        # Calculate translation loss
        pred_dx, pred_dy = pred_shifts
        trans_loss = trans_criterion(pred_dx, gt_dx) + trans_criterion(pred_dy, gt_dy)
        
        # Combine losses and backpropagate
        recon_loss.backward()
        optimizer.step()
        
        # Track metrics
        epoch_recon_loss += recon_loss.item()
        epoch_trans_loss += trans_loss.item()
        
        # Store predictions for visualization
        pred_dxs.extend(pred_dx.detach().cpu().numpy())
        pred_dys.extend(pred_dy.detach().cpu().numpy())
        gt_dxs.extend(gt_dx.detach().cpu().numpy())
        gt_dys.extend(gt_dy.detach().cpu().numpy())
    
    # Calculate average losses
    epoch_recon_loss /= len(train_loader)
    epoch_trans_loss /= len(train_loader)
    epoch_total_loss = epoch_recon_loss + epoch_trans_loss

    return {
        'recon_loss': epoch_recon_loss,
        'trans_loss': epoch_trans_loss,
        'total_loss': epoch_total_loss,
        'pred_dx': pred_dxs,
        'pred_dy': pred_dys,
        'gt_dx': gt_dxs,
        'gt_dy': gt_dys
    }

def test_one_epoch(model, test_loader, device):
    model.eval()

    # Get HR features from the test loader
    hr_coords = test_loader.get_hr_coordinates().unsqueeze(0).to(device)
    img = test_loader.get_original_hr().unsqueeze(0).to(device)
    
    # Add sample_idx=0 for test/inference
    sample_id = torch.tensor([0]).to(device)
    
    try:
        output, _ = model(hr_coords, sample_id)
        
        # Calculate loss
        loss = F.mse_loss(output, img)
        
        return loss.item(), output.detach().cpu().numpy(), img.detach().cpu().numpy()
    except RuntimeError as e:
        print(f"Error during testing: {e}")
        # Return placeholder values
        return 0.0, img.detach().cpu().numpy(), img.detach().cpu().numpy()


def visualize_translations(pred_dx, pred_dy, target_dx, target_dy, save_path='translation_vis.png'):
    """Create a visualization comparing predicted and target translations."""
    plt.figure(figsize=(10, 10))
    
    # Ensure we're working with flattened tensors
    pred_dx = pred_dx.squeeze()  # Remove any extra dimensions
    pred_dy = pred_dy.squeeze()
    target_dx = target_dx.squeeze()
    target_dy = target_dy.squeeze()
    
    # Plot target translations
    plt.scatter(target_dx, target_dy, c='blue', label='Target', alpha=0.6)
    
    # Plot predicted translations
    plt.scatter(pred_dx, pred_dy, c='red', label='Predicted', alpha=0.6)
    
    # Draw lines connecting corresponding points and add annotations
    # check if pred_dx has a length
    if pred_dx.dim() > 0:
        for i in range(len(pred_dx)):
            # Draw connection line
            plt.plot([target_dx[i], pred_dx[i]], 
                    [target_dy[i], pred_dy[i]], 
                    'gray', alpha=0.3)
            
            # Add sample index annotations
            # For target point
            plt.annotate(f'{i:02d}', 
                        (target_dx[i], target_dy[i]),
                        xytext=(5, 5), textcoords='offset points',
                        color='blue', fontsize=8)
            
            # For predicted point
            plt.annotate(f'{i:02d}', 
                        (pred_dx[i], pred_dy[i]),
                        xytext=(5, 5), textcoords='offset points',
                        color='red', fontsize=8)
    
    # Add labels and title
    plt.xlabel('Translation X')
    plt.ylabel('Translation Y')
    plt.title(f'Predicted vs Target Translations\n(Numbers indicate sample indices)')
    plt.legend()
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Make axes equal to preserve translation proportions
    plt.axis('equal')
    
    # Save the plot
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
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
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--use_gt", type=bool, default=False)
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--aug", type=str, default="none", 
                       choices=['none', 'light', 'medium', 'heavy'],
                       help="Augmentation level to use")

    args = parser.parse_args()

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

    device = torch.device(f"cuda:{args.d}" if torch.cuda.is_available() else "cpu")
    network_size = (4, 128)
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

    train_data = SRData(
        f"data/lr_factor_{downsample_factor}x_shift_{lr_shift:.1f}px_samples_{num_samples}_aug_{args.aug}",
    )
    
    # train_data = SyntheticBurstVal("SyntheticBurstVal", 6)

    batch_size = 4

    # initialize the dataloader here
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    model = FourierNetwork(mapping_size * 2, *network_size, len(train_data), rggb=False).to(device)

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

    for i, _ in tqdm(enumerate(range(iters)), total=iters):
        train_losses = train_one_epoch(model, optimizer, train_dataloader, device, iteration=i+1, use_gt=args.use_gt)
        
        # Regular step (no need to pass iteration number)
        scheduler.step()
        
        if (i + 1) % 40 == 0:  # More frequent logging
            test_loss, test_output, test_img = test_one_epoch(model, train_data, device)
            
            # Convert test outputs to correct format
            test_output_tensor = torch.from_numpy(test_output[0]).permute(2, 0, 1).unsqueeze(0).to(device)
            test_img_tensor = torch.from_numpy(test_img[0]).permute(2, 0, 1).unsqueeze(0).to(device)
            
            # Calculate metrics for model output
            model_metrics = calculate_metrics(test_output_tensor, test_img_tensor)
            
            # Calculate metrics for baseline
            hr_img = test_img_tensor


            baseline_pred = F.interpolate(train_data.get_lr_sample(0).unsqueeze(0).permute(0, 3, 1, 2).to("cuda:2"), 
                                    size=(hr_img.shape[2], hr_img.shape[3]), 
                                    mode='bilinear', 
                                    align_corners=False)

            baseline_metrics = calculate_metrics(baseline_pred, hr_img)
            
            # Store values in history
            history['iterations'].append(i + 1)
            history['recon_loss'].append(train_losses['recon_loss'])
            history['trans_loss'].append(train_losses['trans_loss'])
            history['test_loss'].append(test_loss)
            history['psnr'].append(model_metrics['psnr'])
            history['lpips'].append(model_metrics['lpips'])
            history['ssim'].append(model_metrics['ssim'])
            history['baseline_psnr'].append(baseline_metrics['psnr'])
            history['baseline_lpips'].append(baseline_metrics['lpips'])
            history['baseline_ssim'].append(baseline_metrics['ssim'])
            # Store translation data
            history['translation_data'].append({
                'pred_dx': train_losses['pred_dx'],
                'pred_dy': train_losses['pred_dy'],
                'target_dx': train_losses['gt_dx'],
                'target_dy': train_losses['gt_dy'],
            })

            # Store learning rates in history
            history['learning_rate'].append(scheduler.get_last_lr()[0])

            # Print all metrics
            print(f"Iter {i+1}: "
                  f"Train recon: {train_losses['recon_loss']:.6f}, "
                  f"trans: {train_losses['trans_loss']:.6f}, "
                  f"total: {train_losses['total_loss']:.6f}, "
                  f"Test: {test_loss:.6f}\n"
                  f"Metrics vs Baseline:\n"
                  f"PSNR: {model_metrics['psnr']:.2f}dB vs {baseline_metrics['psnr']:.2f}dB\n"
                  f"LPIPS: {model_metrics['lpips']:.4f} vs {baseline_metrics['lpips']:.4f} (lower is better)\n"
                  f"SSIM: {model_metrics['ssim']:.4f} vs {baseline_metrics['ssim']:.4f} (higher is better)")

            if args.wandb:
                wandb.log({
                    "iteration": i + 1,
                    "train/recon_loss": train_losses['recon_loss'],
                    "train/trans_loss": train_losses['trans_loss'],
                    "train/total_loss": train_losses['total_loss'],
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

    # Create all visualizations at the end
    # Training curves
    plot_training_curves(history, save_path=results_dir / 'final_training_curves.png')
    
    # Translation visualization (using last iteration's data)
    last_trans_data = history['translation_data'][-1]
    visualize_translations(
        torch.tensor(last_trans_data['pred_dx']),
        torch.tensor(last_trans_data['pred_dy']),
        torch.tensor(last_trans_data['target_dx']),
        torch.tensor(last_trans_data['target_dy']),
        save_path=results_dir / 'final_translation_vis.png'
    )

    # Final test and visualization
    test_loss, test_output, test_img = test_one_epoch(model, train_data, device)
    
    # Create downsampled then upsampled version for comparison
    with torch.no_grad():
        hr_img = torch.from_numpy(test_img[0]).to(device)
        upsampled = downsample_torch(train_data.get_lr_sample(0).unsqueeze(0).permute(0, 3, 1, 2).to("cuda:2"), (hr_img.shape[0], hr_img.shape[1]))
        upsampled = upsampled[0].permute(1, 2, 0).cpu().numpy()
        
        # Calculate PSNR for model output
        mse_model = F.mse_loss(torch.tensor(test_output[0]), torch.tensor(test_img[0]))
        psnr_model = -10 * torch.log10(mse_model)
        
        # Calculate PSNR for baseline
        mse_baseline = F.mse_loss(torch.tensor(upsampled), torch.tensor(test_img[0]))
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

    # Create side by side visualization
    plt.figure(figsize=(24, 6))
    
    plt.subplot(1, 4, 1)
    plt.imshow(test_img[0])
    plt.title('HR GT')
    plt.axis('off')
    
    plt.subplot(1, 4, 2)
    plt.imshow(lr_target_img)
    plt.title('LR Reference (Sample 00)')
    plt.axis('off')
    
    plt.subplot(1, 4, 3)
    plt.imshow(upsampled)
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