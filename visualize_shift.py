import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from utils import apply_shift_torch

def create_sampling_grid(h, w):
    # Create normalized coordinates (-1 to 1)
    x = torch.linspace(-1, 1, w)
    y = torch.linspace(-1, 1, h)
    
    # Create grid
    y_coords, x_coords = torch.meshgrid(y, x, indexing='ij')
    grid = torch.stack([x_coords, y_coords], dim=-1)
    
    return grid

def apply_shift(grid, dx, dy):
    # Convert pixel shifts to normalized coordinates (-1 to 1 scale)
    h, w = grid.shape[:2]
    # Negate the shifts because:
    # - When sampling grid moves right (positive dx), image appears to move left
    # - When sampling grid moves down (positive dy), image appears to move up
    dx_norm = -2 * dx / w  # Negate dx for correct visual movement
    dy_norm = -2 * dy / h  # Negate dy for correct visual movement
    
    # Apply shift
    shifted_grid = grid.clone()
    shifted_grid[..., 0] = shifted_grid[..., 0] + dx_norm
    shifted_grid[..., 1] = shifted_grid[..., 1] + dy_norm
    
    return shifted_grid

def create_test_image(size=8):
    """Create a test image with a clear pattern"""
    img = np.zeros((size, size, 3))
    # Add a cross pattern
    img[size//2, :, 0] = 1  # Red horizontal line
    img[:, size//2, 2] = 1  # Blue vertical line
    # Add corner markers
    img[0, 0, 1] = 1  # Green top-left
    img[0, -1, 0] = 1  # Red top-right
    img[-1, 0, 2] = 1  # Blue bottom-left
    img[-1, -1, 1] = 1  # Green bottom-right
    return img

def visualize_shift(dx, dy, size=8):
    # Create test image
    img = create_test_image(size)
    
    # Convert to torch tensor
    img_tensor = torch.from_numpy(img).float()
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    
    # Apply shift
    shifted = apply_shift_torch(img_tensor, 
                              torch.tensor([dx]), 
                              torch.tensor([dy]), 
                              normalize=True)
    
    # Convert back to numpy
    shifted = shifted[0].permute(1, 2, 0).numpy()
    
    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # Original image
    ax1.imshow(img)
    ax1.set_title('Original Image')
    ax1.grid(True)
    ax1.set_xticks(np.arange(-0.5, size, 1), minor=True)
    ax1.set_yticks(np.arange(-0.5, size, 1), minor=True)
    ax1.grid(True, which='minor', color='w', linewidth=0.5)
    
    # Shifted image
    ax2.imshow(shifted)
    ax2.set_title(f'Shifted Image (dx={dx}, dy={dy})')
    ax2.grid(True)
    ax2.set_xticks(np.arange(-0.5, size, 1), minor=True)
    ax2.set_yticks(np.arange(-0.5, size, 1), minor=True)
    ax2.grid(True, which='minor', color='w', linewidth=0.5)
    
    plt.suptitle('Visualization of apply_shift_torch')
    plt.tight_layout()
    plt.savefig(f'shift_vis_dx{dx}_dy{dy}.png')
    plt.close()

if __name__ == "__main__":
    # Test different shifts
    shifts = [
        (2, 0),   # Right shift
        (-2, 0),  # Left shift
        (0, 2),   # Down shift
        (0, -2),  # Up shift
        (2, 2),   # Diagonal shift
    ]
    
    for dx, dy in shifts:
        visualize_shift(dx, dy) 