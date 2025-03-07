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
    def __init__(self, project_dim, scale=10, input_dim=2, activation=None, device="cuda"):
        """
        Initialize the Fourier feature projection.
        
        Args:
            project_dim: Number of Fourier features (output will be 2*project_dim)
            scale: Scaling factor for the frequency matrix (higher values = higher frequencies)
            input_dim: Dimension of input coordinates (typically 2 for 2D coordinates)
            activation: Optional activation function to apply after projection
            device: Device to use for computation
        """
        super(FourierProjection, self).__init__()

        self.project_dim = project_dim    # Number of random frequencies
        self.scale = scale                # Scaling factor for frequencies
        self.input_dim = input_dim        # Input dimension
        self.activation = activation      # Optional activation function
        self.device = device              # Computation device
    
    def input_mapping(self, x, B):
        """
        Apply Fourier feature mapping to input coordinates.
        
        Args:
            x: Input tensor of shape [..., input_dim]
            B: Random frequency matrix of shape [project_dim, input_dim]
            
        Returns:
            Tensor of shape [..., 2*project_dim] with sine and cosine features
        """
        if B is None:
            # If no frequency matrix is provided, return inputs unchanged
            return x
        else:
            # Project inputs to a higher dimension using random frequencies
            # Scale inputs by 2π and multiply by frequency matrix
            x_proj = (2. * np.pi * x) @ B.T
            
            # Concatenate sine and cosine features
            # This creates 2*project_dim output features
            return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

    def get_B_gauss(self):
        """
        Generate a random frequency matrix from a Gaussian distribution.
        
        Returns:
            Tensor of shape [project_dim, input_dim] with random frequencies
        """
        # Sample from a Gaussian distribution and scale by self.scale
        # Higher scales lead to higher frequency components
        return torch.randn(self.project_dim, self.input_dim, device=self.device) * self.scale

    def forward(self, x):
        """
        Map input coordinates to Fourier features.
        
        Args:
            x: Input tensor of shape [..., input_dim]
            
        Returns:
            Tensor of shape [..., 2*project_dim] with Fourier features
        """
        # Generate random frequency matrix
        B = self.get_B_gauss()
        
        # Apply Fourier feature mapping
        x = self.input_mapping(x, B)
        
        # Apply optional activation function
        if self.activation is not None:
            x = self.activation(x)
            
        return x

