import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from pathlib import Path
import random
import argparse
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import glob
from PIL import Image

from data import get_dataset
from losses import BasicLosses
from models.utils import get_decoder
from input_projections.utils import get_input_projection
from models.inr import INR


def train_one_iteration(model, optimizer, train_sample, device):
    model.train()
    
    # Initialize loss functions
    recon_criterion = BasicLosses.mse_loss
    trans_criterion = BasicLosses.mae_loss
    
    if model.use_gnll:
        recon_criterion = nn.GaussianNLLLoss()

    input = train_sample['input'].to(device)
    lr_target = train_sample['lr_target'].to(device)
    sample_id = train_sample['sample_id'].to(device)

    # Get ground truth shifts (if available, otherwise zero)
    if 'shifts' in train_sample and 'dx_percent' in train_sample['shifts']:
        gt_dx = train_sample['shifts']['dx_percent'].to(device)
        gt_dy = train_sample['shifts']['dy_percent'].to(device)
    else:
        gt_dx = torch.zeros(lr_target.shape[0], device=device)
        gt_dy = torch.zeros(lr_target.shape[0], device=device)

    optimizer.zero_grad()

    if model.use_gnll:
        output, pred_shifts, pred_variance = model(input, sample_id, scale_factor=0.25, lr_frames=lr_target)
        recon_loss = recon_criterion(output, lr_target, pred_variance)
    else:
        output, pred_shifts = model(input, sample_id, scale_factor=0.25, lr_frames=lr_target)
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


def super_resolve_image(model, image_path, device, scale_factor=4):
    """Super-resolve a single image using the trained model."""
    model.eval()
    
    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img) / 255.0  # Normalize to [0, 1]
    
    # Convert to tensor and add batch dimension
    img_tensor = torch.from_numpy(img_array).float().permute(2, 0, 1).unsqueeze(0)
    
    # Create coordinate grid for the target resolution
    h, w = img_array.shape[:2]
    target_h, target_w = h * scale_factor, w * scale_factor
    
    # Create coordinate grid
    y_coords = torch.linspace(-1, 1, target_h)
    x_coords = torch.linspace(-1, 1, target_w)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
    coords = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2).unsqueeze(0).to(device)
    
    with torch.no_grad():
        sample_id = torch.tensor([0]).to(device)
        
        if model.use_gnll:
            output, _, _ = model(coords, sample_id, scale_factor=scale_factor, training=False)
        else:
            output, _ = model(coords, sample_id, scale_factor=scale_factor, training=False)
        
        # Reshape output to image dimensions
        output = output.reshape(1, target_h, target_w, -1)
        
    return output.squeeze().cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="Super-Resolution for Folder of Images")
    
    # Essential parameters
    parser.add_argument("--input_folder", type=str, required=True, help="Path to folder containing LR images")
    parser.add_argument("--output_folder", type=str, default="sr_outputs", help="Output folder for super-resolved images")
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
    parser.add_argument("--network_depth", type=int, default=10)
    parser.add_argument("--network_hidden_dim", type=int, default=256)
    parser.add_argument("--projection_dim", type=int, default=256)
    parser.add_argument("--input_projection", type=str, default="fourier_10", 
                       choices=["linear", "fourier_10", "fourier_5", "fourier_20", "fourier_40", "fourier", "legendre", "none"])
    parser.add_argument("--fourier_scale", type=float, default=10.0)
    parser.add_argument("--use_gnll", action="store_true")
    
    # Training parameters
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
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

    # Setup dataset for training (we still need this for model initialization)
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

    print(f"Starting training for {args.iters} iterations...")
    
    # Training loop (using the training dataset)
    iteration = 0
    progress_bar = tqdm(total=args.iters, desc="Training")
    
    while iteration < args.iters:
        for train_sample in train_dataloader:
            if iteration >= args.iters:
                break
                
            # Train one iteration
            train_losses = train_one_iteration(model, optimizer, train_sample, device)
            iteration += 1

            # Update progress bar
            progress_bar.update(1)
            progress_bar.set_postfix({
                'recon': f"{train_losses['recon_loss']:.4f}",
                'trans': f"{train_losses['trans_loss']:.4f}"
            })
            
            # Print progress every 100 iterations (no test evaluation available)
            if iteration % 100 == 0:
                print(f"\nIter {iteration}: Train Loss: {train_losses['total_loss']:.6f}")

    progress_bar.close()
    
    # Now super-resolve all images in the input folder
    print(f"\nTraining complete! Super-resolving images from {args.input_folder}...")
    
    # Create output directory
    output_dir = Path(args.output_folder)
    output_dir.mkdir(exist_ok=True)
    
    # Find all image files in input folder
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(str(Path(args.input_folder) / ext)))
    
    if not image_paths:
        print(f"No images found in {args.input_folder}")
        return
    
    print(f"Found {len(image_paths)} images to process...")
    
    # Process each image
    for img_path in tqdm(image_paths, desc="Super-resolving"):
        img_name = Path(img_path).stem
        
        try:
            # Super-resolve the image
            sr_output = super_resolve_image(model, img_path, device, scale_factor=args.df)
            
            # Clamp values to [0, 1] range
            sr_output = np.clip(sr_output, 0, 1)
            
            # Save super-resolved image
            plt.figure(figsize=(8, 8))
            plt.imshow(sr_output)
            plt.axis('off')
            plt.tight_layout(pad=0)
            
            output_path = output_dir / f"{img_name}_sr.png"
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)
            plt.close()
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    print(f"\nSuper-resolution complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main() 