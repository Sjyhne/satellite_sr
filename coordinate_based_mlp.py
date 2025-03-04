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

def get_B_gauss(mapping_size, coordinate_dim, scale=3, device=None):
    return torch.randn(mapping_size, coordinate_dim, device=device) * scale

def get_learnable_transforms(num_samples, coordinate_dim=2, zeros=True, freeze_first=True):
    # Create a list of learnable tensors
    if zeros:
        if freeze_first:
            params = [nn.Parameter(torch.zeros(coordinate_dim), requires_grad=(i != 0)) for i in range(num_samples)]
        else:
            params = [nn.Parameter(torch.zeros(coordinate_dim), requires_grad=True) for i in range(num_samples)]
    else:
        if freeze_first:
            params = [nn.Parameter(torch.ones(coordinate_dim), requires_grad=(i != 0)) for i in range(num_samples)]
        else:
            params = [nn.Parameter(torch.ones(coordinate_dim), requires_grad=True) for i in range(num_samples)]
    # Store them in an nn.ParameterList to register them as model parameters
    return nn.ParameterList(params)  # List of [D] tensors, length B


def get_one_hot_encoding(num_classes):
    return torch.eye(num_classes).cuda()


class FourierNetwork(nn.Module):
    def __init__(self, input_dim, num_layers, num_channels, num_samples, 
                 coordinate_dim=2, code_dim=0, rggb=False, rotation=False, best_sigma=10):
        super().__init__()

        self.code_dim = code_dim
        self.rggb = rggb

        self.input_dim = input_dim
        self.coordinate_dim = coordinate_dim

        self.sigma_opt = best_sigma

        # create decoder MLP
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim + code_dim, num_channels))

        for i in range(num_layers - 2):
            self.layers.append(nn.Linear(num_channels, num_channels))

        self.layers.append(nn.Linear(num_channels, 3))

        self.B = get_B_gauss(input_dim // 2, coordinate_dim, scale=self.sigma_opt)

        # Create transform parameters for each sample
        self.transform_vectors = get_learnable_transforms(num_samples=num_samples, coordinate_dim=coordinate_dim)
        self.transform_angles = get_learnable_transforms(num_samples=num_samples, coordinate_dim=1) if rotation else torch.zeros(num_samples, 1).cuda()

        self.log_variance = get_learnable_transforms(num_samples=num_samples, coordinate_dim=(48, 48, 3), freeze_first=False)

        # learnable codes for each frame
        if self.code_dim > 0:
            self.frame_codes = get_learnable_transforms(num_samples, coordinate_dim=code_dim)
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
                scales = self.color_scales[idx].view(3, 1, 1)  # reshape to [3, 1, 1] for broadcasting
                shifts = self.color_shifts[idx].view(3, 1, 1)  # reshape to [3, 1, 1] for broadcasting

                # Apply channel-wise scaling
                result[i] = x_reshaped[i] * scales + shifts
        
        return result.permute(0, 2, 3, 1)  # Back to [B, H, W, 3]

    def forward(self, x, sample_idx=None, dx_percent=None, dy_percent=None, **kwargs):
        B, H, W, C = x.shape
        x = x.clone()

        dx_list = None
        dy_list = None
        
        if sample_idx is not None:
            dx_list = []
            dy_list = []
            log_variance = []

            for i, sample_id in enumerate(sample_idx):
                if dx_percent is not None and dy_percent is not None:
                    dx = dx_percent[i].squeeze()
                    dy = dy_percent[i].squeeze()
                else:
                    transform = self.transform_vectors[sample_id]
                    dx = transform[0]
                    dy = transform[1]
                    angle = self.transform_angles[sample_id]
                
                log_variance.append(self.log_variance[sample_id])
                
                # create affine transformation matrix
                self.theta = torch.stack([
                    torch.stack([torch.cos(angle), -torch.sin(angle), dx.unsqueeze(0)]),  # Rotation and translation
                    torch.stack([torch.sin(angle), torch.cos(angle), dy.unsqueeze(0)])    # Rotation and translation
                ]).to(x.device)

                # reshape x to [B, H*W, 2]
                x_reshaped = x[i].reshape(-1, 2)
                # convert to homogeneous coordinates: add ones to the last column
                x_reshaped = torch.cat([x_reshaped, torch.ones(x_reshaped.shape[0], 1, device=x.device)], dim=1)
                # apply affine transformation
                x_reshaped = torch.matmul(x_reshaped, self.theta.T)
                # reshape back to [B, H, W, 2]
                x[i] = x_reshaped.reshape(H, W, 2)

                dx_list.append(dx)
                dy_list.append(dy)
            
            dx_list = torch.stack(dx_list)
            dy_list = torch.stack(dy_list)
            
            log_variance = torch.stack(log_variance)

        x = input_mapping(x, self.B.to(x.device))

        # # NOTE: concatenate learned frame_codes to x (maybe useful in combination with transform_vectors to adjust for atmospheric shifts?)   
        if self.code_dim > 0:
            B, H, W, F = x.shape
            frame_codes = torch.stack(list(self.frame_codes))  # Shape [B, D]
            frame_codes = frame_codes[:B, None, None, :].expand(-1, H, W, -1)  # Shape [B, H, W, D]
            # concatenate frame codes to x
            x = torch.cat([x, frame_codes], dim=-1)        


        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = torch.nn.functional.relu(x)
        
        out = self.layers[-1](x)
        #out = torch.nn.functional.sigmoid(out)

        if sample_idx is not None:
            out = self.apply_color_transform(out, sample_idx)
        
        transforms = [dx_list, dy_list] if dx_list is not None else None

        # add noise to the output
        variance = torch.exp(log_variance)

        return out, transforms, variance




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
