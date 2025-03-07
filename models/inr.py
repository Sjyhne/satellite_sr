import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils import get_learnable_transforms

class INR(nn.Module):
    def __init__(self, input_dim,
                 input_transform,
                 decoder,
                 num_samples,
                 coordinate_dim=2):
        super(INR, self).__init__()

        self.input_transform = input_transform
        self.decoder = decoder

        self.shift_vectors = get_learnable_transforms(num_samples=num_samples, coordinate_dim=coordinate_dim)
        self.rotation_angle = get_learnable_transforms(num_samples=num_samples, coordinate_dim=1)

        self.color_shifts = get_learnable_transforms(num_samples=num_samples, coordinate_dim=coordinate_dim)
        self.color_scales = get_learnable_transforms(num_samples=num_samples, coordinate_dim=coordinate_dim)
    

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

    def forward(self, x, sample_idx=None):
        B, H, W, C = x.shape

        if self.input_transform is not None:
            x = self.input_transform(x)
        

        if sample_idx is not None:
            dx_list = []
            dy_list = []

            for i, sample_id in enumerate(sample_idx):
                shift = self.shift_vectors[sample_id]
                angle = self.rotation_angle[sample_id]

                dy = shift[0]
                dx = shift[1]

                # create affine transformation matrix
                self.theta = torch.stack([
                    torch.stack([torch.cos(angle), -torch.sin(angle), dx.unsqueeze(0)]),  # Rotation and translation
                    torch.stack([torch.sin(angle), torch.cos(angle), dy.unsqueeze(0)])    # Rotation and translation
                ]).to(x.device)

                x_reshaped = x[i].reshape(-1, 2)
                x_reshaped = torch.cat([x_reshaped, torch.ones(x_reshaped.shape[0], 1, device=x.device)], dim=1)
                x_reshaped = torch.matmul(x_reshaped, self.theta.T)
                x[i] = x_reshaped.reshape(H, W, 2)  

                dx_list.append(dx)
                dy_list.append(dy)
            
            dx_list = torch.stack(dx_list)
            dy_list = torch.stack(dy_list)

        x = self.decoder(x)

        if sample_idx is not None:
            x = self.apply_color_transform(x, sample_idx)

        shifts = [dx_list, dy_list] if dx_list is not None else None

        return x, shifts
