import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import random

from data import SRData
import cv2
from utils import apply_shift_torch, downsample_torch
from coordinate_based_mlp import TransformFourierNetwork, FourierNetwork
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

def train_one_epoch(model, recon_optimizer, trans_optimizer, train_loader, hr_coords, device, iteration=0, use_gt=False):
    model.train()

    downsample = True

    # if iteration > 20000:
    #     # Freeze the recon layers in the model
    #     for layer in model.layers:
    #         layer.requires_grad_(False)

    # Initialize loss functions
    recon_criterion = BasicLosses.mse_loss
    trans_criterion = BasicLosses.mse_loss
    
    # Move Fourier features to device
    features = hr_coords.to(device)
    if len(features.shape) == 3:  # If not batched yet
        features = features.unsqueeze(0)

    # hr_coords 1, 256, 256, 2
    
    # Collect all samples into batches
    all_imgs = []
    all_dx_percent = []  # For shifting in HR space
    all_dy_percent = []
    all_dx_pixels_hr = []  # For masking in LR space
    all_dy_pixels_hr = []
    all_sample_ids = []

    for sample in train_loader:
        all_imgs.append(sample['image'])
        all_sample_ids.append(sample['sample_id'])
        # HR translations for shifting
        all_dx_percent.append(sample['transform']['dx_percent'])
        all_dy_percent.append(sample['transform']['dy_percent'])
        # LR translations for masking
        all_dx_pixels_hr.append(sample['transform']['dx_hr'])
        all_dy_pixels_hr.append(sample['transform']['dy_hr'])

    # Stack into batches and move to device
    img_batch = torch.stack(all_imgs).to(device)
    dx_batch_percent = torch.tensor(all_dx_percent).to(device)  # For shifting
    dy_batch_percent = torch.tensor(all_dy_percent).to(device)
    dx_batch_pixels_hr = torch.tensor(all_dx_pixels_hr).to(device)  # For masking
    dy_batch_pixels_hr = torch.tensor(all_dy_pixels_hr).to(device)
    sample_id_batch = torch.tensor(all_sample_ids).to(device)

    # Process entire batch at once
    features = features.repeat(len(img_batch), 1, 1, 1)  # [B, H, W, C]
    # 16, 256, 256, 2

    dx_batch_hr = dx_batch_percent.unsqueeze(1)
    dy_batch_hr = dy_batch_percent.unsqueeze(1)

    # Forward pass
    recon_optimizer.zero_grad()
    trans_optimizer.zero_grad()

    if use_gt:
        output, transforms = model(
            features, 
            sample_id_batch,
            dx_percent=dx_batch_percent,
            dy_percent=dy_batch_percent,
            dx_pixels_hr=dx_batch_pixels_hr,
            dy_pixels_hr=dy_batch_pixels_hr,
            lr_shape=(img_batch.shape[1], img_batch.shape[2])
        )
    else:
        output, transforms = model(
            features, 
            sample_id_batch,
            lr_shape=(img_batch.shape[1], img_batch.shape[2])
        )

    mask = None
    
    if isinstance(output, tuple):
        output, mask = output  # Unpack output and mask
        # Downsample mask to LR space
        mask_lr = mask.permute(0, 3, 1, 2)  # [B, 1, H, W]
        mask_lr = downsample_torch(mask_lr, (img_batch.shape[1], img_batch.shape[2]))
        mask_lr = mask_lr.permute(0, 2, 3, 1)  # [B, H, W, 1]


    output_permuted = output.permute(0, 3, 1, 2)  # [B, 3, H, W]
    output_downsampled = downsample_torch(output_permuted, (img_batch.shape[1], img_batch.shape[2]))
    output_final = output_downsampled.permute(0, 2, 3, 1)  # [B, H, W, 3]
    
    # Only compute loss on valid pixels
    #if mask is not None:
    #    recon_loss = recon_criterion(output_final * mask_lr, img_batch * mask_lr).mean()
    #else:
    #    recon_loss = recon_criterion(output_final, img_batch).mean()

    recon_loss = recon_criterion(output_final * mask_lr, img_batch * mask_lr).mean()
    # recon_loss = recon_criterion(output_final, img_batch).mean()

    all_pred_dx = transforms[0].unsqueeze(1)
    all_pred_dy = transforms[1].unsqueeze(1)
    
    translation_loss = trans_criterion(all_pred_dx, dx_batch_hr).mean() + trans_criterion(all_pred_dy, dy_batch_hr).mean()
    translation_loss = translation_loss / 2

    # Combined loss with weighting
    # Give more weight to reconstruction loss initially, then gradually increase translation weight
    total_loss = recon_loss
    
    total_loss.backward()
    recon_optimizer.step()
    trans_optimizer.step()
    
    if (iteration + 1) % 100 == 0 and mask is not None:
        pass
    
    return {
        'total_loss': total_loss.item(),
        'recon_loss': recon_loss.item(),
        'trans_loss': translation_loss.item(),
        'all_pred_dx': all_pred_dx.detach().cpu().numpy(),
        'all_pred_dy': all_pred_dy.detach().cpu().numpy(),
        'dx_batch_hr': dx_batch_hr.detach().cpu().numpy(),
        'dy_batch_hr': dy_batch_hr.detach().cpu().numpy(),
    }

def test_one_epoch(model, test_loader, hr_fourier_features, device):
    model.eval()

    hr_fourier_features = hr_fourier_features.unsqueeze(0).to(device)
    img = test_loader.get_original_hr()
    img = img.unsqueeze(0).to(device)
    
    # Add sample_idx=0 for test/inference
    sample_id = torch.tensor([0]).to(device)
    output, _ = model(hr_fourier_features)
    
    # Handle case where output is (output, mask) tuple
    if isinstance(output, tuple):
        output, _ = output

    loss = F.mse_loss(output, img)

    return loss.item(), output.detach().cpu().numpy(), img.detach().cpu().numpy()


def visualize_translations(pred_dx, pred_dy, target_dx, target_dy, transform_scale=None, save_path='translation_vis.png'):
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
    scale_text = f'\nLearned Scale: {transform_scale:.3f}' if transform_scale is not None else ''
    plt.title(f'Predicted vs Target Translations\n(Numbers indicate sample indices){scale_text}')
    plt.legend()
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Make axes equal to preserve translation proportions
    plt.axis('equal')
    
    # Save the plot
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_training_curves(history, save_path='training_curves.png'):
    """Plot training curves including losses and PSNR."""
    plt.figure(figsize=(15, 10))
    
    # Plot losses
    plt.subplot(2, 1, 1)
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
    plt.subplot(2, 1, 2)
    plt.plot(history['iterations'], history['psnr'], label='Model PSNR')
    plt.axhline(y=history['baseline_psnr'][-1], color='r', linestyle='--', 
                label=f'Baseline PSNR: {history["baseline_psnr"][-1]:.2f}dB')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title('PSNR')
    plt.xlabel('Iteration')
    plt.ylabel('PSNR (dB)')

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
    parser.add_argument("--use_gt", type=bool, default=False)

    args = parser.parse_args()

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
    recon_lr = 5e-4
    trans_lr = 1e-3
    iters = args.iters
    mapping_size = 128
    downsample_factor = args.df
    lr_shift = args.lr_shift
    num_samples = 16

    # Create results directory based on dataset parameters
    results_dir = Path(f"results/lr_factor_{downsample_factor}x_shift_{lr_shift:.1f}px")
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving results to: {results_dir}")

    train_data = SRData(
        f"data/lr_factor_{downsample_factor}x_shift_{lr_shift:.1f}px",
    )

    original_hr = train_data.get_original_hr()
    lr_sample = train_data.get_random_lr_sample()

    # Create input pixel coordinates in the unit square
    hr_coords = np.linspace(0, 1, original_hr.shape[0], endpoint=False)
    hr_coords = np.stack(np.meshgrid(hr_coords, hr_coords), -1)
    hr_coords = torch.FloatTensor(hr_coords).to(device)


    # 16, 128, 128, 512

    if args.model == "TransformFourierNetwork":
        model = TransformFourierNetwork(mapping_size * 2, *network_size, num_samples).to(device)
        recon_optimizer = optim.AdamW(model.layers.parameters(), lr=recon_lr)
        trans_optimizer = optim.AdamW(
            list(model.transform_vectors.parameters()) + [model.learnable_transform_scale], 
            lr=trans_lr
        )
    else:
        model = FourierNetwork(mapping_size * 2, *network_size, num_samples).to(device)
        recon_optimizer = optim.AdamW(model.layers.parameters(), lr=recon_lr)
        trans_optimizer = optim.AdamW(list(model.transform_vectors.parameters()) + [model.learnable_transform_scale], lr=trans_lr)


    # Initialize history dictionary
    history = {
        'iterations': [],
        'recon_loss': [],
        'trans_loss': [],
        'test_loss': [],
        'psnr': [],
        'baseline_psnr': [],
        'translation_data': []  # Add this to store translation data
    }

    for i, _ in tqdm(enumerate(range(iters)), total=iters):
        train_losses = train_one_epoch(model, recon_optimizer, trans_optimizer, train_data, hr_coords, device, iteration=i+1, use_gt=args.use_gt)
        if (i + 1) % 100 == 0:  # More frequent logging
            test_loss, test_output, test_img = test_one_epoch(model, train_data, hr_coords, device)
            
            # Calculate PSNR
            mse_model = F.mse_loss(torch.tensor(test_output[0]), torch.tensor(test_img[0]))
            psnr_model = -10 * torch.log10(mse_model)
            
            # Calculate baseline PSNR
            hr_img = torch.from_numpy(test_img[0]).to(device)
            downsampled = downsample_torch(hr_img.permute(2, 0, 1).unsqueeze(0), 
                                         (hr_img.shape[0]//downsample_factor, hr_img.shape[1]//downsample_factor))
            upsampled = F.interpolate(downsampled, size=(hr_img.shape[0], hr_img.shape[1]), 
                                    mode='bilinear', align_corners=False)
            upsampled = upsampled[0].permute(1, 2, 0)
            mse_baseline = F.mse_loss(upsampled, hr_img)
            psnr_baseline = -10 * torch.log10(mse_baseline)

            # Store values in history
            history['iterations'].append(i + 1)
            history['recon_loss'].append(train_losses['recon_loss'])
            history['trans_loss'].append(train_losses['trans_loss'])
            history['test_loss'].append(test_loss)
            history['psnr'].append(psnr_model.item())
            history['baseline_psnr'].append(psnr_baseline.item())
            # Store translation data
            history['translation_data'].append({
                'pred_dx': train_losses['all_pred_dx'],
                'pred_dy': train_losses['all_pred_dy'],
                'target_dx': train_losses['dx_batch_hr'],
                'target_dy': train_losses['dy_batch_hr'],
                'transform_scale': model.learnable_transform_scale.item()
            })

            print(f"Iter {i+1}: Train recon: {train_losses['recon_loss']:.6f}, "
                  f"trans: {train_losses['trans_loss']:.6f}, "
                  f"total: {train_losses['total_loss']:.6f}, "
                  f"Test: {test_loss:.6f}, "
                  f"PSNR: {psnr_model:.2f}dB")
            
            # Remove visualization calls from here

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
        transform_scale=last_trans_data['transform_scale'],
        save_path=results_dir / 'final_translation_vis.png'
    )

    # Final test and visualization
    test_loss, test_output, test_img = test_one_epoch(model, train_data, hr_coords, device)
    
    # Create downsampled then upsampled version for comparison
    with torch.no_grad():
        hr_img = torch.from_numpy(test_img[0]).to(device)
        downsampled = downsample_torch(hr_img.permute(2, 0, 1).unsqueeze(0), 
                                     (hr_img.shape[0]//downsample_factor, hr_img.shape[1]//downsample_factor))
        upsampled = downsample_torch(downsampled, (hr_img.shape[0], hr_img.shape[1]))
        upsampled = upsampled[0].permute(1, 2, 0).cpu().numpy()
        
        # Calculate PSNR for model output
        mse_model = F.mse_loss(torch.tensor(test_output[0]), torch.tensor(test_img[0]))
        psnr_model = -10 * torch.log10(mse_model)
        
        # Calculate PSNR for baseline
        mse_baseline = F.mse_loss(torch.tensor(upsampled), torch.tensor(test_img[0]))
        psnr_baseline = -10 * torch.log10(mse_baseline)
    
    # Get LR target image (assuming it's the first sample in the dataset)
    lr_target_img = train_data.get_random_lr_sample()  # This is already a tensor
    if torch.is_tensor(lr_target_img):
        lr_target_img = lr_target_img.numpy()  # Convert to numpy if it's a tensor
    
    # Create side by side visualization
    plt.figure(figsize=(24, 6))
    
    plt.subplot(1, 4, 1)
    plt.imshow(test_img[0])
    plt.title('HR GT')
    plt.axis('off')
    
    plt.subplot(1, 4, 2)
    plt.imshow(lr_target_img)  # Should now be a numpy array
    plt.title('LR Target')
    plt.axis('off')
    
    plt.subplot(1, 4, 3)
    plt.imshow(upsampled)
    plt.title(f'Bilinear\nPSNR: {psnr_baseline:.2f} dB')
    plt.axis('off')
    
    plt.subplot(1, 4, 4)
    plt.imshow(test_output[0])
    plt.title(f'Pred\nPSNR: {psnr_model:.2f} dB')
    plt.axis('off')
    
    plt.suptitle(f'Test loss: {test_loss:.6f} | Model PSNR: {psnr_model:.2f} dB\nBaseline PSNR: {psnr_baseline:.2f} dB | Dataset scale: {downsample_factor}x\nIterations: {iters} | recon_lr: {recon_lr} | trans_lr: {trans_lr}\nModel size: {network_size} | mapping_size: {mapping_size}')
    plt.tight_layout()
    plt.savefig(results_dir / 'comparison.png')
    plt.close()

    # Final masking visualization
    output, transforms = model(
        hr_coords.unsqueeze(0), 
        torch.tensor([1]).to(device),
        lr_shape=(lr_target_img.shape[0], lr_target_img.shape[1])
    )
    
    if isinstance(output, tuple):
        output, mask = output
        # Downsample mask to LR space
        mask_lr = mask.permute(0, 3, 1, 2)  # [B, 1, H, W]
        mask_lr = downsample_torch(mask_lr, (lr_target_img.shape[0], lr_target_img.shape[1]))
        
        # Upsample mask back to HR for visualization
        mask_hr = F.interpolate(mask_lr, size=(output.shape[1], output.shape[2]), 
                              mode='nearest')  # Use nearest to keep binary mask
        mask_hr = mask_hr.permute(0, 2, 3, 1)  # [B, H, W, 1]
        
        # Visualize masked images
        visualize_masked_images(
            output.detach(), 
            torch.from_numpy(test_img).to(device),
            mask_hr,
            'final',
            save_dir=results_dir / 'final_mask_vis'
        )

if __name__ == "__main__":
    main()