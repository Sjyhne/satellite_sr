import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearProjection(nn.Module):
    """
    Projects input coordinates to a higher dimensional space using a linear layer.
    
    This is a simpler projection method that applies a fixed linear transformation
    followed by an activation function. The linear weights are frozen after initialization.
    """
    def __init__(self, project_dim, activation=F.relu, input_dim=2, device=None):
        """
        Initialize the linear projection.
        
        Args:
            project_dim: Output dimension after projection
            scale: Not used in this projection but kept for interface consistency
            input_dim: Dimension of input coordinates (typically 2 for 2D coordinates)
            activation: Activation function to apply after linear projection
        """
        super(LinearProjection, self).__init__()

        # Create a linear layer that maps from input_dim to project_dim
        self.linear = nn.Linear(input_dim, project_dim)
        
        # Freeze the weights of the linear layer (no training)
        self.linear.requires_grad = False
        
        # Activation function to apply after linear transformation
        self.activation = activation
        self.device = device

        if self.activation is None:
            raise ValueError("Activation function must be provided for LinearProjection")

    def forward(self, x):
        """
        Map input coordinates to higher dimensions using linear projection.
        
        Args:
            x: Input tensor of shape [..., input_dim]
            
        Returns:
            Tensor of shape [..., project_dim] with projected features
        """
        # Apply linear transformation
        x = self.linear(x)
        
        # Apply activation function
        x = self.activation(x)
        
        return x
