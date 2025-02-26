import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import random
import wandb
import pandas as pd
from torch.optim.lr_scheduler import CosineAnnealingLR
import lpips
from torchmetrics.functional import structural_similarity_index_measure as ssim

from data import BurstData
from coordinate_based_mlp import FourierNetwork
import argparse

def visualize_translations(pred_dx, pred_dy, save_path='translation_vis.png'):
    """Create a visualization of predicted translations."""
    plt.figure(figsize=(10, 10))
    
    # Ensure we're working with flattened tensors
    pred_dx = pred_dx.squeeze()  # Remove any extra dimensions
    pred_dy = pred_dy.squeeze()
    
    # Plot predicted translations
    plt.scatter(pred_dx, pred_dy, c='red', label='Predicted', alpha=0.6)
    
    # Add sample index annotations
    if pred_dx.dim() > 0:
        for i in range(len(pred_dx)):
            # Add sample index annotations
            plt.annotate(f'{i:02d}', 
                        (pred_dx[i], pred_dy[i]),
                        xytext=(5, 5), textcoords='offset points',
                        color='red', fontsize=8)
    
    # Add labels and title
    plt.xlabel('Translation X')
    plt.ylabel('Translation Y')
    plt.title('Predicted Translations\n(Numbers indicate sample indices)')
    plt.legend()
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Make axes equal to preserve translation proportions
    plt.axis('equal')
    
    # Save the plot
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def train_one_epoch(model, optimizer, train_loader, coords, device, iteration=0, accumulate_grad_steps=1):
    model.train()
    
    # Initialize loss function
    recon_criterion = nn.GaussianNLLLoss()
    
    # Move Fourier features to device
    features = coords.to(device)
    if len(features.shape) == 3:  # If not batched yet
        features = features.unsqueeze(0)
    
    # Initialize tracking variables
    total_recon_loss = 0.0
    all_pred_dx = []
    all_pred_dy = []
    
    # Zero gradients at the beginning
    optimizer.zero_grad()
    
    # Process each sample individually
    for sample_idx, sample in enumerate(train_loader):
        # Get current sample
        img = sample['image'].unsqueeze(0).to(device)  # Add batch dimension
        sample_id = torch.tensor([sample['sample_id']]).to(device)
        
        # Forward pass with current sample
        output, transforms = model(features.clone(), sample_id)
        
        # Split output into prediction and variance
        log_variance = output[..., 1:]
        output = output[..., :1]  # Single channel for raw data
        variance = torch.exp(log_variance)
        
        # Calculate reconstruction loss for this sample
        recon_loss = recon_criterion(output, img, variance)
        
        # Scale the loss by the accumulation factor
        scaled_loss = recon_loss / accumulate_grad_steps
        
        # Backward pass
        scaled_loss.backward()
        
        # Track metrics
        total_recon_loss += recon_loss.item()
        
        # Extract and store predicted translations if available
        if transforms is not None:
            pred_dx, pred_dy = transforms
            all_pred_dx.append(pred_dx.detach().cpu())
            all_pred_dy.append(pred_dy.detach().cpu())
        
        # Update weights if we've accumulated enough gradients or this is the last sample
        if (sample_idx + 1) % accumulate_grad_steps == 0 or sample_idx == len(train_loader) - 1:
            optimizer.step()
            optimizer.zero_grad()
    
    # Calculate average loss
    avg_recon_loss = total_recon_loss / len(train_loader)
    
    # Combine all predictions if available
    pred_dx_combined = torch.cat(all_pred_dx) if all_pred_dx else None
    pred_dy_combined = torch.cat(all_pred_dy) if all_pred_dy else None
    
    return {
        'recon_loss': avg_recon_loss,
        'total_loss': avg_recon_loss,
        'pred_dx': pred_dx_combined,
        'pred_dy': pred_dy_combined,
    }

def test_one_epoch(model, test_loader, coords, device):
    model.eval()
    coords = coords.unsqueeze(0).to(device)
    
    # Use first image as reference
    reference_img = test_loader[0]['image'].unsqueeze(0).to(device)
    sample_id = torch.tensor([0]).to(device)
    
    with torch.no_grad():
        output, transforms = model(coords, sample_id)
        output = output[..., :1]  # Take only the prediction, not variance
        loss = F.mse_loss(output, reference_img)

    # Extract predicted translations if available
    pred_dx = None
    pred_dy = None
    if transforms is not None:
        pred_dx, pred_dy = transforms
        pred_dx = pred_dx.cpu()
        pred_dy = pred_dy.cpu()

    return loss.item(), output.detach().cpu().numpy(), reference_img.detach().cpu().numpy(), pred_dx, pred_dy

def plot_training_curves(history, save_path='training_curves.png'):
    plt.figure(figsize=(15, 10))
    
    # Plot losses
    plt.subplot(2, 1, 1)
    plt.plot(history['iterations'], history['recon_loss'], label='Reconstruction Loss')
    plt.plot(history['iterations'], history['test_loss'], label='Test Loss')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title('Training Losses')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')

    # Plot learning rates
    plt.subplot(2, 1, 2)
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
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing burst DNG files")
    parser.add_argument("--d", type=str, default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--accumulate_grad", type=int, default=1, 
                        help="Number of steps to accumulate gradients before updating weights")
    args = parser.parse_args()

    # Create base results directory
    Path('results').mkdir(exist_ok=True)

    # Set all seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # For reproducible results
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device(f"cuda:{args.d}" if torch.cuda.is_available() else "cpu")
    network_size = (4, 128)  # Can be adjusted
    learning_rate = 5e-3
    iters = args.iters
    mapping_size = 128
    num_samples = 0  # Will be determined from dataset

    # Create results directory
    results_dir = Path(f"results/burst_{Path(args.data_dir).name}")
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving results to: {results_dir}")

    # Load dataset
    train_data = BurstData(args.data_dir)
    num_samples = len(train_data)
    print(f"Found {num_samples} samples in dataset")
    
    # Get image dimensions from first sample
    first_sample = train_data[0]['image']
    H, W = first_sample.shape[1:]  # [C, H, W]

    # Create input pixel coordinates in the unit square
    coords_x = np.linspace(0, 1, W, endpoint=False)
    coords_y = np.linspace(0, 1, H, endpoint=False)
    coords = np.stack(np.meshgrid(coords_x, coords_y), -1)
    coords = torch.FloatTensor(coords).to(device)

    # Initialize model - single channel output for raw data plus variance
    model = FourierNetwork(mapping_size * 2, *network_size, num_samples=num_samples).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=iters, eta_min=1e-6)

    # Initialize history dictionary
    history = {
        'iterations': [],
        'recon_loss': [],
        'test_loss': [],
        'learning_rate': [],
        'translation_data': []
    }

    # Initialize wandb if enabled
    if args.wandb:
        wandb.init(
            project="burst-super-res",
            name=f"burst_{Path(args.data_dir).name}",
            config={
                "network_size": network_size,
                "learning_rate": learning_rate,
                "iters": iters,
                "mapping_size": mapping_size,
                "num_samples": num_samples,
                "accumulate_grad_steps": args.accumulate_grad
            }
        )

    for i in tqdm(range(iters)):
        train_losses = train_one_epoch(
            model, optimizer, train_data, coords, device, 
            iteration=i+1, accumulate_grad_steps=args.accumulate_grad
        )
        scheduler.step()
        
        if (i + 1) % 100 == 0:  # Log every 100 iterations
            test_loss, test_output, test_img, pred_dx, pred_dy = test_one_epoch(model, train_data, coords, device)
            
            # Store values in history
            history['iterations'].append(i + 1)
            history['recon_loss'].append(train_losses['recon_loss'])
            history['test_loss'].append(test_loss)
            history['learning_rate'].append(scheduler.get_last_lr()[0])
            
            # Store translation data if available
            if train_losses['pred_dx'] is not None and train_losses['pred_dy'] is not None:
                history['translation_data'].append({
                    'pred_dx': train_losses['pred_dx'].numpy(),
                    'pred_dy': train_losses['pred_dy'].numpy(),
                })
                
                # Visualize translations
                visualize_translations(
                    train_losses['pred_dx'],
                    train_losses['pred_dy'],
                    save_path=results_dir / f'translation_vis_{i+1:04d}.png'
                )

            # Print metrics
            print(f"Iter {i+1}: "
                  f"Train recon: {train_losses['recon_loss']:.6f}, "
                  f"Test: {test_loss:.6f}")

            if args.wandb:
                log_dict = {
                    "iteration": i + 1,
                    "train/recon_loss": train_losses['recon_loss'],
                    "test/loss": test_loss,
                    "learning_rate": scheduler.get_last_lr()[0],
                }
                
                # Add translation visualization if available
                if train_losses['pred_dx'] is not None:
                    trans_vis_path = results_dir / f'translation_vis_{i+1:04d}.png'
                    if trans_vis_path.exists():
                        log_dict["translation_vis"] = wandb.Image(str(trans_vis_path))
                
                wandb.log(log_dict)

            # Visualize current results
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.imshow(test_img[0, 0], cmap='gray')
            plt.title('Reference Raw')
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(test_output[0, 0], cmap='gray')
            plt.title(f'Predicted Raw\nLoss: {test_loss:.6f}')
            plt.axis('off')
            
            plt.suptitle(f'Iteration {i+1}')
            plt.tight_layout()
            plt.savefig(results_dir / f'comparison_{i+1:04d}.png')
            plt.close()

    # Create final visualizations
    plot_training_curves(history, save_path=results_dir / 'final_training_curves.png')
    
    # Final translation visualization if available
    if history['translation_data']:
        last_trans_data = history['translation_data'][-1]
        visualize_translations(
            torch.tensor(last_trans_data['pred_dx']),
            torch.tensor(last_trans_data['pred_dy']),
            save_path=results_dir / 'final_translation_vis.png'
        )

    if args.wandb:
        log_dict = {
            "final_training_curves": wandb.Image(str(results_dir / 'final_training_curves.png')),
            "final_comparison": wandb.Image(str(results_dir / f'comparison_{iters:04d}.png')),
        }
        
        # Add final translation visualization if available
        final_trans_path = results_dir / 'final_translation_vis.png'
        if final_trans_path.exists():
            log_dict["final_translation_vis"] = wandb.Image(str(final_trans_path))
            
        wandb.log(log_dict)
        wandb.finish()

    # Store final metrics
    final_metrics = {
        'final_recon_loss': train_losses['recon_loss'],
        'final_test_loss': test_loss,
    }

    # Save metrics to CSV
    metrics_df = pd.DataFrame([final_metrics])
    metrics_df.to_csv(results_dir / 'metrics.csv', index=False)

if __name__ == "__main__":
    main()
