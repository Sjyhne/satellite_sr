import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import bilinear_resize_torch

from models.utils import get_learnable_transforms

class INR(nn.Module):
    def __init__(self,
                 input_projection,
                 decoder,
                 num_samples,
                 coordinate_dim=2,
                 use_gnll=False):
        super(INR, self).__init__()

        self.input_projection = input_projection
        self.decoder = decoder
        
        self.use_gnll = use_gnll

        self.shift_vectors = get_learnable_transforms(num_samples=num_samples, coordinate_dim=coordinate_dim, zeros=True)
        self.rotation_angle = get_learnable_transforms(num_samples=num_samples, coordinate_dim=1, zeros=True)
        
        ct = nn.ModuleList([nn.Linear(1, 1) for _ in range(3)])
        self.color_transforms = nn.ModuleList([ct for _ in range(num_samples)])

        self.color_transforms[0].requires_grad = False

        # Initialize all biases to 0
        for color_transform in self.color_transforms:
            for ct in color_transform:
                ct.bias.data.zero_()

        # Initialize all weights to 1
        for color_transform in self.color_transforms:
            for ct in color_transform:
                ct.weight.data.fill_(1)

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

    def forward(self, x, sample_idx=None, scale_factor=None, training=True, lr_frames=None):
        B, H, W, C = x.shape

        if sample_idx is not None:
            dx_list = []
            dy_list = []

            for i, sample_id in enumerate(sample_idx):
                shift = self.shift_vectors[sample_id]
                angle = self.rotation_angle[sample_id]

                dx = shift[0]
                dy = shift[1]

                # create affine transformation matrix
                self.theta = torch.stack([
                    torch.stack([torch.cos(angle), -torch.sin(angle), dx.unsqueeze(0)]),  # Rotation and translation
                    torch.stack([torch.sin(angle), torch.cos(angle), dy.unsqueeze(0)])    # Rotation and translation
                ]).to(x.device)

                x_reshaped = x[i].reshape(-1, 2)
                x_reshaped = torch.cat([x_reshaped, torch.ones(x_reshaped.shape[0], 1, device=x.device)], dim=1)
                x_reshaped = torch.matmul(x_reshaped, self.theta.squeeze(-1).mT)
                x[i] = x_reshaped.reshape(H, W, 2)  

                dx_list.append(dx)
                dy_list.append(dy)
            
            dx_list = torch.stack(dx_list)
            dy_list = torch.stack(dy_list)

        coords = x
        
        if self.input_projection is not None:
            x = self.input_projection(x)
        
        try:
            x = self.decoder(coords, x, scale_factor)
            print("scale_factor", scale_factor)
        except Exception as e:
            x = self.decoder(x)

        if sample_idx is not None:
            x = self.apply_color_transform(x, sample_idx)

        shifts = [dx_list, dy_list] if dx_list is not None else None


        if training:
            x = bilinear_resize_torch(x.permute(0, 3, 1, 2), (int(H * scale_factor), int(W * scale_factor))).permute(0, 2, 3, 1)

        if self.use_gnll:
            variances = []
            if lr_frames is not None:
                for i, sample_id in enumerate(sample_idx):
                    variances.append(torch.exp(self.variance_predictor(torch.cat([x[i], lr_frames[i]], dim=-1))))
                variances = torch.stack(variances, dim=0)

            return x, shifts, variances
        else:
            return x, shifts
