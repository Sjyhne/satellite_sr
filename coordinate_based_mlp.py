import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import numpy as np

from utils import apply_shift_torch




# Fourier feature mapping
def input_mapping(x, B):
    if B is None:
        return x
    else:
        x_proj = (2. * np.pi * x) @ B.T
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

def get_B_gauss(mapping_size, coordinate_dim, scale=10):
    return torch.randn(mapping_size, coordinate_dim) * scale

# PyTorch network definition
class FourierNetwork(nn.Module):
    def __init__(self, input_dim, num_layers, num_channels, num_samples, coordinate_dim=2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, num_channels))
        for i in range(num_layers - 2):
            self.layers.append(nn.Linear(num_channels, num_channels))
        self.layers.append(nn.Linear(num_channels, 3))

        self.B = get_B_gauss(input_dim // 2, coordinate_dim)
        
        # Create spatial transform parameters for each sample
        self.transform_vectors = nn.ParameterList([
            nn.ParameterList([
                nn.Parameter(torch.zeros(1), requires_grad=(i != 0))
                for _ in range(coordinate_dim)
            ])
            for i in range(num_samples)
        ])

        # Create color scaling parameters for each sample (just 3 values per sample for RGB)
        self.color_scales = nn.ParameterList([
            nn.Parameter(torch.ones(3), requires_grad=(i != 0))  # [3] for RGB
            for i in range(num_samples)
        ])

    def apply_color_transform(self, x, sample_idx):
        """Apply per-channel color scaling."""
        x_reshaped = x.permute(0, 3, 1, 2)  # [B, 3, H, W]
        result = x_reshaped.clone()
        
        for i, idx in enumerate(sample_idx):
            if idx != 0:  # Skip reference sample
                # Get color scales for this sample [3]
                scales = self.color_scales[idx].view(3, 1, 1)  # reshape to [3, 1, 1] for broadcasting
                # Apply channel-wise scaling
                result[i] = x_reshaped[i] * scales
        
        return torch.clamp(result.permute(0, 2, 3, 1), 0, 1)  # Back to [B, H, W, 3]

    def forward(self, x, sample_idx=None, dx_percent=None, dy_percent=None, **kwargs):
        B, H, W, C = x.shape
        x = x.clone()

        dx_list = None
        dy_list = None
        
        if sample_idx is not None:
            dx_list = []
            dy_list = []
            for i, sample_id in enumerate(sample_idx):
                transform = self.transform_vectors[sample_id]
                if dx_percent is not None and dy_percent is not None:
                    dx = dx_percent[i].squeeze()
                    dy = dy_percent[i].squeeze()
                else:
                    dx = transform[0]
                    dy = transform[1]

                x[i, :, :, 0] += dx
                x[i, :, :, 1] += dy
                dx_list.append(dx)
                dy_list.append(dy)

            dx_list = torch.stack(dx_list)
            dy_list = torch.stack(dy_list)

        x = input_mapping(x, self.B.to(x.device))
        for i, layer in enumerate(self.layers[:-1]):
            x = torch.nn.functional.relu(layer(x))
        out = torch.sigmoid(self.layers[-1](x))

        if sample_idx is not None:
            out = self.apply_color_transform(out, sample_idx)

        transforms = [dx_list, dy_list] if dx_list is not None else None
        return out, transforms
    

# PyTorch network definition
class TransformFourierNetwork(nn.Module):
    def __init__(self, input_dim, num_layers, num_channels, num_samples, coordinate_dim=2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, num_channels))
        for i in range(num_layers - 2):
            self.layers.append(nn.Linear(num_channels, num_channels))
        self.layers.append(nn.Linear(num_channels, 3))

        self.learnable_transform_scale = nn.Parameter(torch.ones(1))

        self.B = get_B_gauss(input_dim // 2, coordinate_dim)

        # Create transform parameters for each sample
        self.transform_vectors = nn.ParameterList([
            nn.ParameterList([
                nn.Parameter(torch.zeros(1) if i == 0 else ((torch.randn(1) - 0.5) * 0.01), 
                            requires_grad=(i != 0))
                for _ in range(coordinate_dim)
            ])
            for i in range(num_samples)
        ])

    def create_translation_mask(self, shape, dx, dy):
        """Create a mask for valid pixels after translation.
        
        Args:
            shape: (H, W) tuple of image dimensions
            dx: translation in x (positive = right)
            dy: translation in y (positive = down)
        Returns:
            mask: Binary mask of valid pixels [B, H, W]
        """
        H, W = shape
        B = len(dx)  # Batch size
        mask = torch.ones((B, H, W), device=dx.device)
        
        # For each sample in batch
        for i in range(B):
            # Handle x-direction masking
            if dx[i] > 0:
                # Moving right - mask right edge
                width = int(abs(dx[i].item()))
                if width > 0:
                    mask[i, :, -width:] = 0
            elif dx[i] < 0:
                # Moving left - mask left edge
                width = int(abs(dx[i].item()))
                if width > 0:
                    mask[i, :, :width] = 0
            
            # Handle y-direction masking
            if dy[i] > 0:
                # Moving down - mask bottom edge
                height = int(abs(dy[i].item()))
                if height > 0:
                    mask[i, -height:, :] = 0
            elif dy[i] < 0:
                # Moving up - mask top edge
                height = int(abs(dy[i].item()))
                if height > 0:
                    mask[i, :height, :] = 0
        
        return mask

    def forward(self, x, sample_idx=None, dx_percent=None, dy_percent=None, dx_pixels_hr=None, dy_pixels_hr=None, lr_shape=None):
        x = input_mapping(x, self.B.to(x.device))
        B, H, W, _ = x.shape

        # Get base output from main network
        for i, layer in enumerate(self.layers[:-1]):
            x = torch.nn.functional.gelu(layer(x))
        out = torch.sigmoid(self.layers[-1](x))  # [B, H, W, 3]
        
        if sample_idx is not None:
            transforms = []
            
            for i, sample_id in enumerate(sample_idx):
                transform = self.transform_vectors[sample_id]
                if dx_pixels_hr is not None and dy_pixels_hr is not None:
                    # Convert HR pixel translations to LR pixel translations
                    dx = dx_pixels_hr[i].squeeze()
                    dy = dy_pixels_hr[i].squeeze()  # Scale to LR pixels
                else:
                    # Convert normalized predictions to LR pixel space
                    dx = transform[0] 
                    dy = transform[1] 
                transforms.append([dx, dy])
            
            # Convert transforms list to tensors
            dx_list = torch.stack([t[0] for t in transforms])
            dy_list = torch.stack([t[1] for t in transforms])

            # Create validity mask using LR pixel translations
            mask = self.create_translation_mask(lr_shape, dx_list, dy_list)
            mask = mask.unsqueeze(-1)  # Add channel dimension [B, H, W, 1]
            
            # Apply shift to output using pixel-space translations
            # apply_shift_torch will handle normalization internally
            out = out.permute(0, 3, 1, 2)  # [B, 3, H, W]
            out = apply_shift_torch(out, dx_list, dy_list)
            out = out.permute(0, 2, 3, 1)  # [B, H, W, 3]
            
            # Return output and mask separately for loss computation
            out = (out, mask)  # Return tuple of (unmasked output, mask)
            transforms = [dx_list, dy_list]
            
            return out, transforms
        
        return out, None
    
    
