import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import numpy as np

from utils import apply_shift_torch

def nystrom_mapping(x, Z, kernel_fn):
    """
    Nyström feature mapping.
    
    Parameters:
      x: torch.Tensor of shape (n_samples, input_dim)
      Z: torch.Tensor of landmark points with shape (n_landmarks, input_dim)
         If Z is None, the function returns x unchanged.
      kernel_fn: function to compute the kernel between two sets of points.
                 It should accept two arguments (e.g., kernel_fn(x, Z)) and return a tensor.
    
    Returns:
      phi_x: torch.Tensor of shape (n_samples, n_landmarks) representing the approximate feature mapping.
    """
    if Z is None:
        return x
    else:
        # Compute kernel between x and landmark points (n_samples x n_landmarks)
        K_xZ = kernel_fn(x, Z)
        # Compute kernel matrix among landmarks (n_landmarks x n_landmarks)
        K_ZZ = kernel_fn(Z, Z)
        # Compute SVD of the landmark kernel matrix
        U, S, _ = torch.svd(K_ZZ)
        # Compute the inverse square root of singular values for stability (add epsilon)
        eps = 1e-6
        S_inv_sqrt = torch.diag(1.0 / torch.sqrt(S + eps))
        # Form the Nyström mapping: phi(x) = K(x, Z) * U * S^{-1/2}
        phi_x = K_xZ @ U @ S_inv_sqrt
        return phi_x


def random_maclaurin_mapping(x, D=128, degree=2, c=1):
    """
    Random Maclaurin feature mapping for approximating polynomial kernels.
    
    Approximates the kernel (x^T y + c)^degree using D random features.
    
    Parameters:
      x: torch.Tensor of shape (n_samples, input_dim)
      D: int, number of random features to generate.
         If D is None, the function returns x unchanged.
      degree: int, degree of the polynomial kernel.
      c: float, constant offset in the polynomial kernel (default is 0.0).
      
    Returns:
      phi_x: torch.Tensor of shape (n_samples, D) representing the approximate feature mapping.
    """
    if D is None or degree is None:
        return x
    else:
        n, input_dim = x.shape
        features = []
        for i in range(D):
            # Randomly sample indices for this monomial (sampled with replacement)
            indices = torch.randint(low=0, high=input_dim, size=(degree,))
            # Randomly assign signs (+1 or -1) for each coordinate
            signs = 2 * torch.randint(0, 2, (degree,), dtype=torch.float32) - 1.0
            # Select the corresponding coordinates: shape (n, degree)
            x_selected = x[:, indices]
            # Apply the random signs
            x_weighted = x_selected * signs
            # Optionally incorporate constant offset c.
            if c != 0.0:
                # A simple way is to add a constant offset to each term.
                # Here we add sqrt(c/degree) to each term as a rough heuristic.
                offset = np.sqrt(c / degree)
                x_weighted = x_weighted + offset
            # Multiply the selected coordinates to form one feature per sample
            feature = torch.prod(x_weighted, dim=1, keepdim=True)
            features.append(feature)
        # Concatenate all D features (resulting shape: n x D)
        phi_x = torch.cat(features, dim=1)
        # Optionally normalize the features
        phi_x = phi_x / torch.sqrt(torch.tensor(D, dtype=phi_x.dtype))
        return phi_x





# Fourier feature mapping
def input_mapping(x, B):
    if B is None:
        return x
    else:
        x_proj = (2. * np.pi * x) @ B.T
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

def get_B_gauss(mapping_size, coordinate_dim, scale=10):
    return torch.randn(mapping_size, coordinate_dim) * scale

## v1
# def get_learnable_transforms(num_samples, coordinate_dim=2):
#     # we freeze the first transform to be zero
#     params = nn.ParameterList([
#         nn.ParameterList([
#             nn.Parameter(torch.zeros(1), requires_grad=(i != 0))
#             for _ in range(coordinate_dim)
#         ])
#         for i in range(num_samples)
#     ])
#     return params

## v2 it should be the same as above, however this implementation changes the logging of transform loss. The reconstruction loss is the same and the shifts are also the same.
# def get_learnable_transforms(num_samples, coordinate_dim=2):
#     # loop through each sample set requires_grad to False for the first sample
#     params = nn.ParameterList([
#         nn.Parameter(torch.zeros(coordinate_dim), requires_grad=True) for i in range(num_samples)
#     ])
#     params[0].requires_grad = False
#     return params

def get_learnable_transforms(num_samples, coordinate_dim=2, zeros=True):
    # Create a list of learnable tensors
    if zeros:
        params = [nn.Parameter(torch.zeros(coordinate_dim), requires_grad=(i != 0)) for i in range(num_samples)]
    else:
        params = [nn.Parameter(torch.ones(coordinate_dim), requires_grad=(i != 0)) for i in range(num_samples)]
    # Store them in an nn.ParameterList to register them as model parameters
    return nn.ParameterList(params)  # List of [D] tensors, length B


def get_one_hot_encoding(num_classes):
    return torch.eye(num_classes).cuda()


class FourierNetwork(nn.Module):
    def __init__(self, input_dim, num_layers, num_channels, num_samples, coordinate_dim=2, code_dim=0, rggb=False):
        super().__init__()

        self.code_dim = code_dim
        self.rggb = rggb

        # create decoder MLP
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim + code_dim, num_channels))
        for i in range(num_layers - 2):
            self.layers.append(nn.Linear(num_channels, num_channels))
        if self.rggb:
            self.layers.append(nn.Linear(num_channels, 5))
        else:
            self.layers.append(nn.Linear(num_channels, 4))

        self.B = get_B_gauss(input_dim // 2, coordinate_dim)

        # Create transform parameters for each sample
        self.transform_vectors = get_learnable_transforms(num_samples=num_samples, coordinate_dim=coordinate_dim)

        # learnable codes for each frame
        if self.code_dim > 0:
            self.frame_codes = get_learnable_transforms(num_samples, coordinate_dim=code_dim).cuda()
        else:
            self.frame_codes = nn.ParameterList([])
        if self.rggb:
            self.color_shifts = get_learnable_transforms(num_samples, coordinate_dim=4)
            self.color_scales = get_learnable_transforms(num_samples, coordinate_dim=4, zeros=False)
        else:
            self.color_shifts = get_learnable_transforms(num_samples, coordinate_dim=3)
            self.color_scales = get_learnable_transforms(num_samples, coordinate_dim=3, zeros=False)




    def apply_color_transform(self, x, sample_idx):
        """Apply per-channel color scaling."""
        x_reshaped = x.permute(0, 3, 1, 2)  # [B, 3, H, W]
        result = x_reshaped.clone()
        
        for i, idx in enumerate(sample_idx):
            if idx != 0:  # Skip reference sample
                # Get color scales for this sample [3]
                if self.rggb:
                    scales = self.color_scales[idx].view(4, 1, 1)  # reshape to [3, 1, 1] for broadcasting
                    shifts = self.color_shifts[idx].view(4, 1, 1)  # reshape to [3, 1, 1] for broadcasting
                else:
                    scales = self.color_scales[idx].view(3, 1, 1)  # reshape to [3, 1, 1] for broadcasting
                    shifts = self.color_shifts[idx].view(3, 1, 1)  # reshape to [3, 1, 1] for broadcasting
                # Apply channel-wise scaling
                result[i] = x_reshaped[i] * scales + shifts
        
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
                
                # Apply transforms to coordinates
                x[i, :, :, 0] += dx
                x[i, :, :, 1] += dy
                with torch.no_grad():
                    dx_list.append(dx)
                    dy_list.append(dy)
            
            with torch.no_grad():
                # Convert transforms list to tensors
                dx_list = torch.stack(dx_list)
                dy_list = torch.stack(dy_list)

        # # NOTE: Alternative to sum the self.transform_vectors to the input x
        # B, H, W, _ = x.shape
        # transform_vectors = torch.stack(list(self.transform_vectors))  # Shape [B, D]
        # transform_vectors = transform_vectors[:B, None, None, :].expand(-1, H, W, -1)  # Shape [B, H, W, D]
        # x += transform_vectors

        # Apply Fourier feature mapping
        x = input_mapping(x, self.B.to(x.device))  # ([16, 256, 256, 256])

        # # NOTE: concatenate learned frame_codes to x (maybe useful in combination with transform_vectors to adjust for atmospheric shifts?)   
        if self.code_dim > 0:
            B, H, W, F = x.shape
            frame_codes = torch.stack(list(self.frame_codes))  # Shape [B, D]
            frame_codes = frame_codes[:B, None, None, :].expand(-1, H, W, -1)  # Shape [B, H, W, D]
            # concatenate frame codes to x
            x = torch.cat([x, frame_codes], dim=-1)        


        # Process through layers
        for i, layer in enumerate(self.layers[:-1]):
            x = torch.nn.functional.relu(layer(x))

        rgbv = self.layers[-1](x)

        rgb = torch.sigmoid(rgbv[..., :-1])

        variance = torch.relu(rgbv[..., -1:])

        if sample_idx is not None:
            rgb = self.apply_color_transform(rgb, sample_idx)
        
        out = torch.cat([rgb, variance], dim=-1)

        transforms = [dx_list, dy_list] if dx_list is not None else None

        return out, transforms
