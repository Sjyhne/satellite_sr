import cv2
import numpy as np
import pathlib
import random
import json
import torch
import torch.nn.functional as F
from utils import apply_shift_torch, downsample_torch

def generate_random_translations(num_samples, max_pixels_x, max_pixels_y=None):
    """
    Generate random translations with independent x and y limits.
    
    Args:
        num_samples: Number of translations to generate
        max_pixels_x: Maximum translation in x direction (in LR space)
        max_pixels_y: Maximum translation in y direction (in LR space)
    """
    if max_pixels_y is None:
        max_pixels_y = max_pixels_x
        
    translations = []
    for _ in range(num_samples):
        dx = random.uniform(-max_pixels_x, max_pixels_x)
        dy = random.uniform(-max_pixels_y, max_pixels_y)
        translations.append((dx, dy))
    return translations

if __name__ == "__main__":
    # Set seeds
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    device = "cuda:0"
    
    # Image settings
    target_hr_size = 740  # Size to resize the full HR image to (after border crop)
    hr_patch_size = 256   # Size of HR patches to extract
    
    # Multiple downsampling factors to generate
    downsampling_factors = [1, 2, 4, 8]  # Will create patches of size 256, 128, and 64
    
    # Translation settings (in HR space)
    max_hr_translation = 8  # Maximum pixels to translate in HR space
    num_samples = 16
    
    # Cropping settings
    border_crop = 20  # Pixels to remove from border (black edge)
    
    # Create base data directory
    base_save_folder = pathlib.Path("data")
    
    # Load and crop border from image
    img_path = pathlib.Path("images/hr_image.png")
    original_img = cv2.imread(str(img_path))
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    original_img = original_img[border_crop:-border_crop, border_crop:-border_crop]
    
    # Resize to target HR size
    # original_img = cv2.resize(original_img, (target_hr_size, target_hr_size))
    
    # Convert to torch tensor
    original_img = torch.from_numpy(original_img).float() / 255.0
    original_img = original_img.permute(2, 0, 1).unsqueeze(0).to(device)
    
    # Calculate center position for the base patch
    height, width = original_img.shape[-2:]
    center_y = (height // 2) - 100
    center_x = (width // 2) - 100
    half_patch = hr_patch_size // 2
    
    # Extract the center patch (this will be our HR ground truth)
    base_hr_patch = original_img[
        :,
        :,
        center_y - half_patch:center_y + half_patch,
        center_x - half_patch:center_x + half_patch
    ]
    
    # Generate translations in HR space
    translations = generate_random_translations(
        num_samples=num_samples,
        max_pixels_x=max_hr_translation,
        max_pixels_y=max_hr_translation
    )
    
    print(f"Generated {len(translations)} translations with max HR shift of {max_hr_translation} pixels")
    print(f"This will result in:")
    for factor in downsampling_factors:
        print(f"  {factor}x downsampling: max shift of {max_hr_translation/factor:.1f} pixels")
    
    # After generating translations
    print("Sample translations in HR space:")
    for i, (dx, dy) in enumerate(translations):
        print(f"Sample {i:02d}: dx={dx:.3f}, dy={dy:.3f}")
    
    # Process each downsampling factor
    for factor in downsampling_factors:
        print(f"\nProcessing downsampling factor: {factor}x")
        
        lr_patch_size = hr_patch_size // factor
        print(f"LR patch size: {lr_patch_size}x{lr_patch_size}")
        
        # Create directory for this factor
        save_folder = base_save_folder / f"lr_factor_{factor}x"
        save_folder.mkdir(parents=True, exist_ok=True)
        
        transform_log = {}
        
        for i, (dx, dy) in enumerate(translations):
            if i == 0:
                dx = 0
                dy = 0

            # Apply shift to full image
            shifted_full = apply_shift_torch(
                original_img,
                dx=dx,
                dy=dy
            )
            
            # Extract our patch from the center of the shifted image
            shifted_patch = shifted_full[
                :,
                :,
                center_y - half_patch:center_y + half_patch,
                center_x - half_patch:center_x + half_patch
            ]
            
            sample_name = f"sample_{i:02d}"
            
            # Save HR patch only for the reference (unshifted) sample
            if i == 0:  # This is sample_00
                hr_np = (shifted_patch[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                hr_np = cv2.cvtColor(hr_np, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(save_folder / "hr_ground_truth.png"), hr_np)
            
            # Downsample and save LR patch
            shifted_lr = downsample_torch(shifted_patch, (lr_patch_size, lr_patch_size))
            lr_np = (shifted_lr[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            lr_np = cv2.cvtColor(lr_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(save_folder / f"{sample_name}.png"), lr_np)
            
            # Log transformation
            transform_log[sample_name] = {
                'dx_pixels_hr': dx,
                'dy_pixels_hr': dy,
                'dx_pixels_lr': dx / factor,
                'dy_pixels_lr': dy / factor,
                'dx_percent': (dx/factor) / lr_patch_size,
                'dy_percent': (dy/factor) / lr_patch_size,
                'magnitude_pixels_hr': np.sqrt(dx**2 + dy**2),
                'magnitude_pixels_lr': np.sqrt((dx/factor)**2 + (dy/factor)**2),
                'shape': lr_np.shape,
                'path': f"{sample_name}.png"
            }
        
        # Save transform log for this factor
        with open(save_folder / "transform_log.json", 'w') as f:
            json.dump(transform_log, f, indent=2)

        # Create difference images
        diff_folder = save_folder / "differences"
        diff_folder.mkdir(parents=True, exist_ok=True)

        # Load reference image (sample_00)
        ref_img = cv2.imread(str(save_folder / "sample_00.png"))

        # Calculate differences with all other samples
        for i in range(1, num_samples):
            sample_path = save_folder / f"sample_{i:02d}.png"
            sample_img = cv2.imread(str(sample_path))
            
            diff = cv2.absdiff(ref_img, sample_img)
            diff = cv2.multiply(diff, 2.0)
            
            cv2.imwrite(str(diff_folder / f"diff_{i:02d}.png"), diff)
            comparison = np.hstack([ref_img, sample_img, diff])
            cv2.imwrite(str(diff_folder / f"comparison_{i:02d}.png"), comparison)

        # Inside the factor loop, after calculating transform_log
        print(f"\nFor factor {factor}x:")
        for sample_name, trans in transform_log.items():
            print(f"{sample_name}: dx_lr={trans['dx_pixels_lr']:.3f}, dy_lr={trans['dy_pixels_lr']:.3f}, "
                  f"dx_percent={trans['dx_percent']:.3f}, dy_percent={trans['dy_percent']:.3f}")

