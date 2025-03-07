import torch
import torch.nn as nn
import numpy as np

class LegendreProjection(nn.Module):
    """
    Projects input coordinates to a higher dimensional space using Legendre polynomials.
    
    Legendre polynomials are orthogonal polynomials defined on the interval [-1,1]. 
    This projection maps input coordinates to features representing evaluations of 
    Legendre polynomials of different degrees.
    """
    def __init__(self, max_degree=10, input_dim=2, activation=None, device="cuda"):
        """
        Initialize the Legendre polynomial projection.
        
        Args:
            max_degree: Maximum degree of Legendre polynomials to use
            input_dim: Dimension of input coordinates (typically 2 for 2D coordinates)
            device: Device to use for computation
        """
        super(LegendreProjection, self).__init__()
        
        self.max_degree = max_degree
        self.input_dim = input_dim
        self.device = device
        self.output_dim = (max_degree + 1) * input_dim
        
        # Pre-compute coefficients for each degree of Legendre polynomial
        # Each row contains coefficients for one degree (e.g., eye(4) gives coefficients for degrees 0,1,2,3)
        self.register_buffer('coefficients', torch.from_numpy(np.eye(max_degree + 1)).float())
    
    def get_output_dim(self):
        return self.output_dim
        
    def legendre_polynomial(self, x):
        """
        Evaluate Legendre polynomials at positions x.
        
        Args:
            x: Input tensor of shape [..., input_dim] with values in range [-1, 1]
            
        Returns:
            Tensor of shape [..., input_dim * (max_degree + 1)] containing Legendre polynomial evaluations
        """
        # Initialize with P_0(x) = 1 and P_1(x) = x
        results = [torch.ones_like(x), x]
        
        # Recurrence relation for Legendre polynomials:
        # (n+1)P_{n+1}(x) = (2n+1)xP_n(x) - nP_{n-1}(x)
        for n in range(1, self.max_degree):
            # Compute P_{n+1} using the recurrence relation
            p_next = ((2 * n + 1) * x * results[n] - n * results[n-1]) / (n + 1)
            results.append(p_next)
            
        # Stack all polynomials into one tensor
        return torch.stack(results, dim=-1)
        
    def forward(self, x):
        """
        Map input coordinates to Legendre polynomial features.
        
        Args:
            x: Input tensor of shape [..., input_dim] with values in range [0, 1]
            
        Returns:
            Tensor of shape [..., input_dim * (max_degree + 1)]
        """
        # Input is typically in [0, 1], rescale to [-1, 1] for Legendre polynomials
        x_scaled = 2.0 * x - 1.0
        
        # Process each dimension separately
        results = []
        for i in range(self.input_dim):
            # Get coordinates for this dimension
            x_i = x_scaled[..., i:i+1]
            
            # Evaluate Legendre polynomials for this dimension
            poly_values = self.legendre_polynomial(x_i)
            
            # Flatten the result for this dimension
            results.append(poly_values)
            
        # Concatenate results from all dimensions
        # Shape: [..., input_dim * (max_degree + 1)]
        output = torch.cat(results, dim=-1)

        if self.activation is not None:
            output = self.activation(output)
        
        return output
