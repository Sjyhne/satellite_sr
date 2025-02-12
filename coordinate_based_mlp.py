import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np




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
        self.layers.append(nn.Linear(num_channels, 3))  # Comment says "+1 for transform value" but we removed that feature

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

    def forward(self, x, sample_idx=None, true_dx=None, true_dy=None):
        # Ensure transform vectors are broadcastable
        B, H, W, C = x.shape  # Assuming x has shape [Batch, Height, Width, Channels]

        transforms = None
        x = x.clone()  # Create a copy to avoid modifying the input
        
        if sample_idx is not None:
            transforms = []
            for i, sample_id in enumerate(sample_idx):
                transform = self.transform_vectors[sample_id]

                if true_dx is not None and true_dy is not None:
                    # Use true values for coordinate shifting
                    dx_for_coords = true_dx[i].view(1, 1, 1)
                    dy_for_coords = true_dy[i].view(1, 1, 1)
                    
                    # Apply true transforms to coordinates
                    x[i, :, :, 0] -= (dx_for_coords.squeeze(0))
                    x[i, :, :, 1] -= (dy_for_coords.squeeze(0))
                    
                    # Store true transforms for loss
                    transforms.append([dx_for_coords.squeeze(), dy_for_coords.squeeze()])
                else:
                    # Use predicted transforms
                    dx = transform[0].view(1, 1, 1) 
                    dy = transform[1].view(1, 1, 1)
                    
                    # Apply predicted transforms
                    x[i, :, :, 0] += dx.squeeze(0)
                    x[i, :, :, 1] += dy.squeeze(0)
                    
                    # Store predicted transforms
                    transforms.append([dx.squeeze(), dy.squeeze()])

            # Convert transforms list to tensors
            dx_list = torch.stack([t[0] for t in transforms])
            dy_list = torch.stack([t[1] for t in transforms])
            transforms = [dx_list, dy_list]

        x = input_mapping(x, self.B.to(x.device))

        # Process through layers
        for i, layer in enumerate(self.layers[:-1]):
            x = torch.nn.functional.relu(layer(x))
        
        x = torch.sigmoid(self.layers[-1](x))

        return x, transforms
    

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

        # Create transform parameters for each sample
        self.transform_vectors = nn.ParameterList([
            nn.ParameterList([
                nn.Parameter(torch.zeros(1) if i == 0 else (torch.randn(1) - 0.5), 
                            requires_grad=(i != 0))
                for _ in range(coordinate_dim)
            ])
            for i in range(num_samples)
        ])

    def forward(self, x, sample_idx=None):
        # Get base output from main network
        for i, layer in enumerate(self.layers[:-1]):
            if i == 0:
                self.last_pred = layer(x)
            x = torch.nn.functional.gelu(layer(x))
        out = torch.sigmoid(self.layers[-1](x))

        if sample_idx is not None:
            transforms = []
            for sample_id in sample_idx:
                # Get learned translations for this sample
                transform = self.transform_vectors[sample_id]
                transforms.append([t * self.learnable_transform_scale for t in transform])
            return out, transforms
        
        return out, None
    
    
