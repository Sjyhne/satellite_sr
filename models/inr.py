import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from utils import bilinear_resize_torch

from models.utils import get_learnable_transforms
from models.nir import apply_affine, Affine
import einops

class INR(nn.Module):
    def __init__(self,
                 input_projection,
                 decoder,
                 num_samples,
                 coordinate_dim=2,
                 use_gnll=False,
                 use_base_frame=True,
                 use_direct_param_T=True):
        super(INR, self).__init__()

        self.input_projection = input_projection
        self.decoder = decoder

        self.num_samples = num_samples
        self.time_vectors = torch.FloatTensor(np.linspace(-1, 1, self.num_samples))
        
        self.use_gnll = use_gnll

        if use_direct_param_T:
            self.shift_vectors = get_learnable_transforms(num_samples=num_samples, coordinate_dim=coordinate_dim, zeros=True, freeze_first=use_base_frame)
            self.rotation_angle = get_learnable_transforms(num_samples=num_samples, coordinate_dim=1, zeros=True, freeze_first=use_base_frame)
        else:
            self.affine_mlp = Affine(hidden_features=256, hidden_layers=2)

        self.use_direct_param_T = use_direct_param_T
        
        #ct = nn.ModuleList([nn.Linear(1, 1) for _ in range(3)])
        #self.color_transforms = nn.ModuleList([ct for _ in range(num_samples)])

        #self.color_transforms[0].requires_grad = False

        # Initialize all biases to 0
        #for color_transform in self.color_transforms:
        #    for ct in color_transform:
        #        ct.bias.data.zero_()

        # Initialize all weights to 1
        #for color_transform in self.color_transforms:
        #    for ct in color_transform:
        #        ct.weight.data.fill_(1)

        if self.use_gnll:

            self.variance_predictor = nn.Sequential(
                nn.Linear(self.decoder.output_dim * 2, 128),
                nn.ReLU(),
                nn.Linear(128, self.decoder.output_dim)
            )

    def apply_color_transform(self, x, sample_idx):
        """Apply per-channel color scaling."""
        result = x.clone()

        for i, idx in enumerate(sample_idx):
            if idx != 0:  # Skip reference sample
                for channel in range(3):
                    transformed = self.color_transforms[idx][channel](x[i, :, :, channel].unsqueeze(-1))
                    result[i, :, :, channel] = transformed.squeeze(-1)

        return result
    
    def get_affine_transform(self, sample_id):

        if self.time_vectors.device != sample_id.device:
            self.time_vectors = self.time_vectors.to(sample_id.device)

        if self.use_direct_param_T:
            return self.get_direct_affine(sample_id)
        else:
            return self.get_neural_affine(self.time_vectors[sample_id])
    
    def get_direct_affine(self, sample_id):

        B = sample_id.shape[0]

        # Handle batch indexing of ParameterList
        shifts = []
        angles = []
        
        for i, idx in enumerate(sample_id):
            shifts.append(self.shift_vectors[idx.item()])
            angles.append(self.rotation_angle[idx.item()])
        
        shift = torch.stack(shifts)  # [B, 2]
        angle = torch.stack(angles)  # [B, 1]

        a1 = torch.stack([torch.cos(angle), -torch.sin(angle), shift[:, 0].unsqueeze(-1)], dim=1).squeeze(-1)
        a2 = torch.stack([torch.sin(angle), torch.cos(angle), shift[:, 1].unsqueeze(-1)], dim=1).squeeze(-1)

        A = torch.stack([a1, a2], dim=1)

        assert A.shape == (B, 2, 3), f"A.shape: {A.shape}"

        return A


    def get_neural_affine(self, time):
        
        time = time.unsqueeze(-1)


        affine_params = self.affine_mlp(time) # [B, 6]

        B, C = affine_params.shape

        A = affine_params.reshape(B, 2, 3) # [B, 2, 3]

        assert A.shape == (B, 2, 3)

        return A

    def apply_affine(self, coords, A):

        B, H, W, C = coords.shape
        
        coords = coords.reshape(B, -1, C) # [B, H*W, C]

        homogenous_coords = torch.cat([coords, torch.ones(B, coords.shape[1], 1, device=coords.device)], dim=2) # B, HW, 3 - Homoegenous coordinates

        transformed_coords = torch.matmul(homogenous_coords, A.mT) # B, HW, 2

        return transformed_coords.reshape(B, H, W, C)


    def forward(self, coords, sample_idx=None, scale_factor=None, training=True, lr_frames=None):
        B, H, W, C = coords.shape

        # Initialize shift lists
        dx_list = None
        dy_list = None

        if training:

            A = self.get_affine_transform(sample_idx) # [B, 2, 3]

            dx_list = A[:, 0, 2]
            dy_list = A[:, 1, 2]

            coords = self.apply_affine(coords, A)


        if self.input_projection is not None:
            coords = self.input_projection(coords)

        output = self.decoder(coords)

        shifts = [dx_list, dy_list] if dx_list is not None else None

        if training: # pool the supersampled output to the LR resolution

            if scale_factor.unique().shape[0] == 1:
                scale_factor = scale_factor.unique().item()
            else:
                raise ValueError("Not implemented functionality that supports multiple scale factors in the same batch")

            output = F.interpolate(output.permute(0, 3, 1, 2), 
                             scale_factor=scale_factor, 
                             mode='area').permute(0, 2, 3, 1)

        if self.use_gnll:
            variances = []
            if lr_frames is not None:
                for i, sample_id in enumerate(sample_idx):
                    variances.append(torch.exp(self.variance_predictor(torch.cat([output[i], lr_frames[i]], dim=-1))))
                variances = torch.stack(variances, dim=0)

            return output, shifts, variances
        else:
            return output, shifts
