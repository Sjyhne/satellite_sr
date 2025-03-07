import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from itertools import product

def legendre_polynomial_recursive(n, x):
    """
    Compute the Legendre polynomial P_n(x) using Bonnet's recursion formula.
    """
    if n == 0:
        return torch.ones_like(x)
    elif n == 1:
        return x
    else:
        return ((2 * n - 1) * x * legendre_polynomial_recursive(n - 1, x) - (n - 1) * legendre_polynomial_recursive(n - 2, x)) / n

def polynomial_features_torch(coords, degree):
    """
    Generate Legendre polynomial features up to a given degree for 2D coordinates using PyTorch.
    
    Args:
        coords (torch.Tensor): Tensor of shape (B, H, W, 2) containing (x, y) coordinates.
        degree (int): Maximum degree of polynomial expansion.

    Returns:
        torch.Tensor: Expanded feature tensor of shape (B, H, W, num_features).
    """
    B, H, W, _ = coords.shape
    
    # Normalize coordinates to range [-1, 1]
    x, y = 2 * coords[..., 0] - 1, 2 * coords[..., 1] - 1
    
    num_features = (degree + 1) * (degree + 2) // 2  # Calculate number of polynomial features
    features = [torch.ones((B, H, W), device=coords.device)]  # Bias term (degree 0)
    
    # Generate Legendre polynomial terms using recursion
    for d in range(1, degree + 1):
        for i, j in product(range(d + 1), repeat=2):
            if i + j <= d:
                legendre_x = legendre_polynomial_recursive(i, x)
                legendre_y = legendre_polynomial_recursive(j, y)
                features.append(legendre_x * legendre_y)
    
    return torch.stack(features, dim=-1)  # Stack features and return num_features

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

class FourierFeatures(nn.Module):
    def __init__(self, input_dim, num_frequencies, max_frequency, annealing_steps):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.annealing_steps = annealing_steps
        self.register_buffer("B", torch.randn(num_frequencies, input_dim) * max_frequency)
        self.current_step = 0  # Track training step

    def annealing_factor(self):
        # Linear schedule (adjust as needed)
        return min(1.0, self.current_step / self.annealing_steps)

    def forward(self, x):
        # Apply annealing to frequencies
        sigma_t = self.annealing_factor()
        scaled_B = self.B * sigma_t  # Scale frequency matrix

        # Center input around 0 by shifting from [0,1] to [-0.5,0.5]
        x_centered = x - 0.5  

        # Compute Fourier features
        x_proj = (2.0 * np.pi * x_centered) @ scaled_B.T  # Shape: (batch, num_frequencies)
        fourier_feats = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

        return fourier_feats

    def step(self):
        """ Call this at each training step to update annealing. """
        self.current_step += 1

import math
class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                             1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                             np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
    
class sin_fr_layer(nn.Module):
    def __init__(self, in_features, out_features, high_freq_num,low_freq_num,phi_num,alpha,omega_0=30.0):
        super().__init__()
        super(sin_fr_layer,self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.high_freq_num =high_freq_num
        self.low_freq_num = low_freq_num
        self.phi_num = phi_num
        self.alpha=alpha
        self.omega_0=omega_0
        self.bases=self.init_bases()
        self.lamb=self.init_lamb()
        self.bias=nn.Parameter(torch.Tensor(self.out_features,1),requires_grad=True)
        self.init_bias()

    def init_bases(self):
        phi_set=np.array([2*math.pi*i/self.phi_num for i in range(self.phi_num)])
        high_freq=np.array([i+1 for i in range(self.high_freq_num)])
        low_freq=np.array([(i+1)/self.low_freq_num for i in range(self.low_freq_num)])
        if len(low_freq)!=0:
            T_max=2*math.pi/low_freq[0]
        else:
            T_max=2*math.pi/min(high_freq) # 取最大周期作为取点区间
        points=np.linspace(-T_max/2,T_max/2,self.in_features)
        bases=torch.Tensor((self.high_freq_num+self.low_freq_num)*self.phi_num,self.in_features)
        i=0
        for freq in low_freq:
            for phi in phi_set:
                base=torch.tensor([math.cos(freq*x+phi) for x in points])
                bases[i,:]=base
                i+=1
        for freq in high_freq:
            for phi in phi_set:
                base=torch.tensor([math.cos(freq*x+phi) for x in points])
                bases[i,:]=base
                i+=1
        bases=self.alpha*bases
        bases=nn.Parameter(bases,requires_grad=False)
        return bases

    
    def init_lamb(self):
        self.lamb=torch.Tensor(self.out_features,(self.high_freq_num+self.low_freq_num)*self.phi_num)
        with torch.no_grad():
            m=(self.low_freq_num+self.high_freq_num)*self.phi_num
            for i in range(m):
                dominator=torch.norm(self.bases[i,:],p=2)
                self.lamb[:,i]=nn.init.uniform_(self.lamb[:,i],-np.sqrt(6/m)/dominator/self.omega_0,np.sqrt(6/m)/dominator/self.omega_0)
        self.lamb=nn.Parameter(self.lamb,requires_grad=True)
        return self.lamb

    def init_bias(self):
        with torch.no_grad():
            nn.init.zeros_(self.bias)
        
    def forward(self,x):
        weight=torch.matmul(self.lamb,self.bases)
        output=torch.matmul(x,weight.transpose(0,1))
        output=output+self.bias.T
        return torch.sin(self.omega_0*output)


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

        # self.layers.append(SineLayer(2 + code_dim, num_channels))

        for i in range(num_layers - 2):
            # self.layers.append(sin_fr_layer(num_channels, num_channels, 128, 128, 32, 0.05, 30.0))
            self.layers.append(nn.Linear(num_channels, num_channels))

        final_linear = nn.Linear(num_channels, 3)
        # with torch.no_grad():
        #     final_linear.weight.uniform_(-np.sqrt(6/ num_channels)/30.0,np.sqrt(6 / num_channels)/30.0)
        
        self.layers.append(final_linear)
        # self.layers.append(nn.Linear(num_channels, 3))

        self.B = get_B_gauss(input_dim // 2, coordinate_dim, scale=self.sigma_opt)

        self.random_linear = nn.Linear(2, input_dim)


        # self.polynomial_features = polynomial_features_torch

        self.fourier_features = FourierFeatures(input_dim=coordinate_dim, num_frequencies=128, max_frequency=10, annealing_steps=1000)

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

        
        # x = input_mapping(x, self.B.to(x.device))
        with torch.no_grad():
            x = torch.sigmoid(self.random_linear(x))

        # x = self.polynomial_features(x, 8)
        # x = self.fourier_features(x)
        # self.fourier_features.step()

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
