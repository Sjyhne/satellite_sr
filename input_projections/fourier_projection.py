import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class FourierProjection(nn.Module):
    """
    Projects input coordinates to a higher dimensional space using random Fourier features.
    
    Fourier features map low-dimensional inputs to a higher dimensional space by
    projecting them with random frequencies and computing sine and cosine activations.
    This approach helps models learn high-frequency functions more effectively.
    """
    def __init__(self, input_dim=2, output_dim=256, scale=10.0, device=None):
        """
        Initialize the Fourier feature projection.
        
        Args:
            input_dim: Dimension of input coordinates (typically 2 for 2D coordinates)
            output_dim: Number of Fourier features (output will be 2*output_dim)
            scale: Scaling factor for the frequency matrix (higher values = higher frequencies)
        """
        super(FourierProjection, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim // 2 # Beacuse we are using sin and cos
        self.scale = scale
        self.device = device
    
        # Create a random matrix for Fourier features
        # Shape: [output_dim, input_dim]
        B = self.get_B_gauss()
        
        # Move B to the specified device if provided
        if device is not None:
            B = B.to(device)
        
        # Register B as a buffer (not a parameter)
        self.register_buffer('B', B)

    def input_mapping(self, x, B):
        """
        Apply Fourier feature mapping to input coordinates.
        
        Args:
            x: Input coordinates tensor of shape [..., input_dim]
            B: Random Fourier feature matrix of shape [output_dim, input_dim]
            
        Returns:
            Tensor of shape [..., output_dim * 2] with Fourier features
        """
        # Move B to the same device as x
        B = B.to(x.device)
        
        # Project input coordinates
        x_proj = (2. * np.pi * x) @ B.T
        
        # Apply sine and cosine to get Fourier features
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

    def get_B_gauss(self):
        """
        Generate a random frequency matrix from a Gaussian distribution.
        
        Returns:
            Tensor of shape [output_dim, input_dim] with random frequencies
        """
        # Sample from a Gaussian distribution and scale by self.scale
        # Higher scales lead to higher frequency components
        return torch.randn(self.output_dim, self.input_dim, device=self.device) * self.scale

    def forward(self, x):
        """
        Map input coordinates to Fourier features.
        
        Args:
            x: Input tensor of shape [..., input_dim]
            
        Returns:
            Tensor of shape [..., 2*output_dim] with Fourier features
        """

        # Apply Fourier feature mapping
        x = self.input_mapping(x, self.B)

        return x

