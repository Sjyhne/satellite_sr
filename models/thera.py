import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Thermal(nn.Module):
    def __init__(self, w0_scale=1.):
        super().__init__()
        self.w0_scale = w0_scale

    def forward(self, x, phase, t, norm, k):
        return torch.sin(self.w0_scale * x + phase) * \
               torch.exp(-(self.w0_scale * norm)**2 * k * t)


class TheraField(nn.Module):
    def __init__(self, dim_hidden, dim_out, w0=1., c=6.):
        super().__init__()
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out
        self.w0 = w0
        self.c = c
        self.thermal = Thermal(w0)

    def forward(self, coords, t, k, components, phase, linear_weight):
        x = coords @ components  # (B, hidden_dim)
        norm = components.norm(dim=0, keepdim=True)  # (1, hidden_dim)
        x = self.thermal(x, phase, t, norm, k)
        x = torch.einsum('bh,bho->bo', x, linear_weight)
        return x


class Hypernetwork(nn.Module):
    def __init__(self, feature_dim, thera_hidden_dim, output_dim, depth=4):
        super().__init__()

        self.layers = nn.ModuleList()
        for d in range(depth - 1):
            self.layers.append(nn.Sequential(
                nn.Linear(feature_dim, thera_hidden_dim),
                nn.ReLU(),
            ))
        
        self.pre_out = nn.Sequential(*self.layers)

        self.phase_head = nn.Linear(thera_hidden_dim, thera_hidden_dim)
        self.linear_head = nn.Linear(thera_hidden_dim, thera_hidden_dim * output_dim)

        self.thera_hidden_dim = thera_hidden_dim
        self.output_dim = output_dim

    def forward(self, features, coords):
        features = features.permute(0, 3, 1, 2)
        B, _, H, W = features.shape
        N_coords = H * W

        sampled_features = F.grid_sample(features, coords, align_corners=True, mode='bilinear')
        sampled_features = sampled_features.view(B, -1, N_coords).unsqueeze(2)  # (B, feature_dim, 1, N_coords)

        sampled_features = sampled_features.permute(0, 3, 2, 1)

        pre_out = self.pre_out(sampled_features)
        phase = self.phase_head(pre_out)
        linear_weight = self.linear_head(pre_out)

        phase = phase.view(B, N_coords, self.thera_hidden_dim)
        linear_weight = linear_weight.view(B, N_coords, self.thera_hidden_dim, self.output_dim)

        return phase, linear_weight




class Thera(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=3, depth=4):
        super().__init__()
        self.hypernet = Hypernetwork(input_dim, hidden_dim, output_dim, depth)
        self.k = nn.Parameter(torch.sqrt(torch.log(torch.tensor(4.))) / (2 * torch.pi**2))
        self.components = nn.Parameter(torch.randn(2, hidden_dim) * 0.1) # This is W_1
        self.thera_field = TheraField(hidden_dim, output_dim)
        self.output_dim = output_dim


    def forward(self, coords, source_features, scale_factor=4):
        B, H, W, _ = coords.shape
        N = H * W

        phase, linear_weight = self.hypernet(source_features, coords)

        t = scale_factor**2

        coords_flat = coords.reshape(B*N, -1) 
        phase_flat = phase.reshape(B*N, -1) # B1
        linear_weight_flat = linear_weight.reshape(B*N, self.thera_field.dim_hidden, self.thera_field.dim_out) # W2

        preds_flat = self.thera_field(coords_flat, t, self.k, self.components, phase_flat, linear_weight_flat)
        preds = preds_flat.view(B, H, W, -1)

        return preds
    
    # 64x64 LR t = 16

    # 256x256 HR --> t = 1



