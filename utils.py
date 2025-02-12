import torch
import torch.nn.functional as F
import cv2
import numpy as np

def apply_shift_cv2(image, dx, dy):
    """Apply translation using cv2 for data generation"""
    rows, cols = image.shape[:2]
    M = np.float32([[1, 0, dx],
                    [0, 1, dy]])
    return cv2.warpAffine(image, M, (cols, rows))

def apply_shift_torch(img, dx, dy, normalize=True):
    """Apply translation to image.
    
    Args:
        img: Input image tensor [B,C,H,W]
        dx: Shift in x direction (in pixels, positive = right) [B]
        dy: Shift in y direction (in pixels, positive = down) [B]
        normalize: If True, normalize shifts by image dimensions
    """
    B = img.shape[0]
    if normalize:
        _, _, H, W = img.shape
        dx = dx / W
        dy = dy / H
    
    # Create batched transformation matrix
    theta = torch.zeros(B, 2, 3, device=img.device)
    theta[:, 0, 0] = 1  # Set diagonal to 1
    theta[:, 1, 1] = 1
    theta[:, 0, 2] = dx  # Set translations
    theta[:, 1, 2] = dy
    
    grid = F.affine_grid(theta, img.size(), align_corners=False)
    return F.grid_sample(img, grid, mode='bilinear', align_corners=False)

def downsample_cv2(image, size):
    """Downsample using cv2 for data generation"""
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

def downsample_torch(image, size):
    """Downsample using torch for training"""
    return F.interpolate(image, size=size, mode='bilinear', align_corners=False)
 