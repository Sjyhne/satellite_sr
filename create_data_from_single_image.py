import cv2
import numpy as np
import pathlib
import random
import json
import torch
import torch.nn.functional as F
from utils import apply_shift_torch, downsample_torch
import torchvision.transforms.functional as TF
from torchvision import transforms

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
    for i in range(num_samples):
        if i == 0:  # First sample should be unshifted
            dx = 0.0
            dy = 0.0
        else:
            dx = random.uniform(-max_pixels_x, max_pixels_x)
            dy = random.uniform(-max_pixels_y, max_pixels_y)
        translations.append((dx, dy))
    return translations

def generate_fixed_lr_translations(num_samples, lr_pixel_shifts, factor):
    """
    Generate translations that correspond to specific pixel shifts in LR space.
    
    Args:
        num_samples: Number of translations per shift magnitude
        lr_pixel_shifts: List of desired pixel shifts in LR space (e.g., [0.5, 1.0, 2.0, 4.0])
        factor: Downsampling factor (to convert LR shifts to HR shifts)
    """
    translations = []
    
    # Always add the unshifted reference sample first
    translations.append((0.0, 0.0))
    
    # For each desired LR shift magnitude
    for lr_shift in lr_pixel_shifts:
        hr_shift = lr_shift * factor  # Convert LR pixels to HR pixels
        
        # Generate samples for this shift magnitude
        samples_per_shift = (num_samples - 1) // len(lr_pixel_shifts)
        for i in range(samples_per_shift):
            # Generate random angle
            angle = random.uniform(0, 2 * np.pi)
            # Calculate x and y components
            dx = hr_shift * np.cos(angle)
            dy = hr_shift * np.sin(angle)
            translations.append((dx, dy))
    
    return translations

def generate_fixed_magnitude_translations(num_samples, lr_pixel_shift, factor):
    """
    Generate translations with random x and y shifts within bounds.
    
    Args:
        num_samples: Number of translations to generate
        lr_pixel_shift: Maximum shift in LR space (for both x and y)
        factor: Downsampling factor (to convert LR shifts to HR shifts)
    """
    translations = []
    
    # Always add the unshifted reference sample first
    translations.append((0.0, 0.0))
    
    # Convert LR shift to HR shift
    hr_shift = lr_pixel_shift * factor
    
    # Generate remaining samples with random x and y shifts
    for i in range(num_samples - 1):
        # Independent random shifts for x and y, each between -hr_shift and +hr_shift
        dx = random.uniform(-hr_shift, hr_shift)
        dy = random.uniform(-hr_shift, hr_shift)
        translations.append((dx, dy))
    
    return translations

def apply_atmospheric_augmentations(image, seed, augment_params=None):
    """
    Apply color jitter augmentations to simulate atmospheric effects.
    
    Args:
        image: Torch tensor image [B, C, H, W] in range [0, 1]
        seed: Random seed for reproducibility
        augment_params: Dictionary of augmentation parameters, if None uses defaults
    """
    # Set seed for reproducibility
    torch.manual_seed(seed)
    
    if augment_params is None:
        augment_params = {
            'brightness': (0.8, 1.2),
            'contrast': (0.8, 1.2),
            'saturation': (0.8, 1.2),
            'hue': (-0.1, 0.1),
        }
    
    # Start with original image
    aug_image = image.clone()
    
    # Apply color jitter
    saturation_factor = torch.empty(1).uniform_(*augment_params['saturation']).item()
    aug_image = TF.adjust_saturation(aug_image, saturation_factor)
    
    hue_factor = torch.empty(1).uniform_(*augment_params['hue']).item()
    aug_image = TF.adjust_hue(aug_image, hue_factor)
    
    brightness_factor = torch.empty(1).uniform_(*augment_params['brightness']).item()
    aug_image = TF.adjust_brightness(aug_image, brightness_factor)
    
    contrast_factor = torch.empty(1).uniform_(*augment_params['contrast']).item()
    aug_image = TF.adjust_contrast(aug_image, contrast_factor)
    
    # Ensure values stay in [0, 1] range
    aug_image = torch.clamp(aug_image, 0, 1)
    
    return aug_image

if __name__ == "__main__":
    # Set seeds
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Ensures deterministic behavior in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    device = "cuda:0"
    
    # Image settings
    target_hr_size = 740  # Size to resize the full HR image to (after border crop)
    hr_patch_size = 256   # Size of HR patches to extract
    
    # Multiple downsampling factors to generate
    downsampling_factors = [1, 2, 4, 8]
    
    # Translation settings
    lr_pixel_shifts = [0.5, 1.0, 2.0, 4.0]  # Each will get its own dataset
    sample_counts = [1, 4, 8, 12, 16]  # Different numbers of samples to test
    
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
    
    # Augmentation settings
    augmentation_types = ['none', 'light', 'medium', 'heavy']
    augment_params = {
        'light': {
            'brightness': (0.9, 1.1),
            'contrast': (0.9, 1.1),
            'saturation': (0.9, 1.1),
            'hue': (-0.05, 0.05),
        },
        'medium': {
            'brightness': (0.8, 1.2),
            'contrast': (0.8, 1.2),
            'saturation': (0.8, 1.2),
            'hue': (-0.1, 0.1),
        },
        'heavy': {
            'brightness': (0.6, 1.4),
            'contrast': (0.6, 1.4),
            'saturation': (0.6, 1.4),
            'hue': (-0.2, 0.2),
        }
    }
    
    # Process each combination of factor, shift, sample count, and augmentation
    for factor in downsampling_factors:
        for lr_shift in lr_pixel_shifts:
            for num_samples in sample_counts:
                for aug_type in augmentation_types:
                    print(f"\nProcessing factor {factor}x with {lr_shift}px LR shift, "
                          f"{num_samples} samples, and {aug_type} augmentation")
                    
                    # Generate translations
                    translations = generate_fixed_magnitude_translations(
                        num_samples=num_samples,
                        lr_pixel_shift=lr_shift,
                        factor=factor
                    )
                    
                    # Calculate LR patch size
                    lr_patch_size = hr_patch_size // factor
                    
                    # Create directory for this combination
                    save_folder = base_save_folder / f"lr_factor_{factor}x_shift_{lr_shift:.1f}px_samples_{num_samples}_aug_{aug_type}"
                    save_folder.mkdir(parents=True, exist_ok=True)
                    
                    transform_log = {}
                    
                    for i, (dx, dy) in enumerate(translations):
                        if i == 0:
                            dx = 0
                            dy = 0
                        
                        # 1. First apply shift to full image
                        shifted_full = apply_shift_torch(
                            original_img,
                            dx=torch.tensor([dx], device=device),
                            dy=torch.tensor([dy], device=device)
                        )
                        
                        # 2. Extract HR patch
                        shifted_patch = shifted_full[
                            :,
                            :,
                            center_y - half_patch:center_y + half_patch,
                            center_x - half_patch:center_x + half_patch
                        ]
                        
                        # 3. Save HR patch for reference sample without augmentation
                        if i == 0:
                            hr_np = (shifted_patch[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                            hr_np = cv2.cvtColor(hr_np, cv2.COLOR_RGB2BGR)
                            cv2.imwrite(str(save_folder / "hr_ground_truth.png"), hr_np)
                        # Apply augmentations only to non-reference samples (i > 0)
                        elif aug_type != 'none':
                            aug_seed = seed * 1000 + i
                            shifted_patch = apply_atmospheric_augmentations(
                                shifted_patch,
                                seed=aug_seed,
                                augment_params=augment_params[aug_type]
                            )
                        
                        # 4. Downsample to LR
                        shifted_lr = downsample_torch(shifted_patch, (lr_patch_size, lr_patch_size))
                        
                        # 5. Save LR sample
                        lr_np = (shifted_lr[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                        lr_np = cv2.cvtColor(lr_np, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(str(save_folder / f"sample_{i:02d}.png"), lr_np)
                        
                        # Update transform log
                        transform_log[f"sample_{i:02d}"] = {
                            'dx_pixels_hr': dx,
                            'dy_pixels_hr': dy,
                            'dx_pixels_lr': dx / factor,
                            'dy_pixels_lr': dy / factor,
                            'dx_percent': (dx/factor) / lr_patch_size,
                            'dy_percent': (dy/factor) / lr_patch_size,
                            'magnitude_pixels_hr': np.sqrt(dx**2 + dy**2),
                            'magnitude_pixels_lr': np.sqrt((dx/factor)**2 + (dy/factor)**2),
                            'shape': lr_np.shape,
                            'path': f"sample_{i:02d}.png",
                            'augmentation': aug_type
                        }
                    
                    # Save transform log
                    with open(save_folder / "transform_log.json", 'w') as f:
                        json.dump(transform_log, f, indent=2)

                    # Print transform log summary
                    print(f"\nFor factor {factor}x with {lr_shift}px LR shift and {num_samples} samples:")
                    for sample_name, trans in transform_log.items():
                        print(f"{sample_name}: dx_lr={trans['dx_pixels_lr']:.3f}, dy_lr={trans['dy_pixels_lr']:.3f}, "
                              f"dx_percent={trans['dx_percent']:.3f}, dy_percent={trans['dy_percent']:.3f}")

