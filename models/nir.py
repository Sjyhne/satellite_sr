import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torch.nn import functional as F

import numpy as np
import os
from PIL import Image

import einops

from models.mlp import MLP


class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
    # hyperparameter.
    
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    
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
    
    def forward_with_intermediate(self, input): 
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate
    
    
class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, 
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        output = self.net(coords)
        return output


class Homography(nn.Module):
    def __init__(self, in_features=1, hidden_features=256, hidden_layers=1):
        super().__init__()
        out_features = 8
        
        self.net = []
        self.net.append(nn.Linear(in_features, hidden_features))
        self.net.append(nn.ReLU(inplace=True))
        for i in range(hidden_layers):
            self.net.append(nn.Linear(hidden_features, hidden_features))
            self.net.append(nn.ReLU(inplace=True))
        self.net.append(nn.Linear(hidden_features, out_features))     
        self.net = nn.Sequential(*self.net)
        
        self.init_weights()
        
    def init_weights(self):
        with torch.no_grad():
            self.net[-1].bias.copy_(torch.Tensor([1., 0., 0., 0., 1., 0., 0., 0.]))
    
    def forward(self, coords):
        output = self.net(coords)
        return output
    

class Affine(nn.Module):
    def __init__(self, in_features=1, hidden_features=256, hidden_layers=1):
        super().__init__()
        out_features = 6  # 6 parameters for affine transformation
        
        self.net = []
        self.net.append(nn.Linear(in_features, hidden_features))
        self.net.append(nn.ReLU(inplace=True))
        for i in range(hidden_layers):
            self.net.append(nn.Linear(hidden_features, hidden_features))
            self.net.append(nn.ReLU(inplace=True))
        self.net.append(nn.Linear(hidden_features, out_features))     
        self.net = nn.Sequential(*self.net)
        
        self.init_weights()
        
    def init_weights(self):
        with torch.no_grad():
            # Initialize to identity transformation: [1, 0, 0, 1, 0, 0]
            self.net[-1].bias.copy_(torch.Tensor([1., 0., 0., 1., 0., 0.]))
    
    def forward(self, coords):
        output = self.net(coords)
        return output



# UTILS SECTION

def get_mgrid(sidelen, vmin=-1, vmax=1):
    if type(vmin) is not list:
        vmin = [vmin for _ in range(len(sidelen))]
    if type(vmax) is not list:
        vmax = [vmax for _ in range(len(sidelen))]
    tensors = tuple([torch.linspace(vmin[i], vmax[i], steps=sidelen[i]) for i in range(len(sidelen))])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, len(sidelen))
    return mgrid

def apply_homography(x, h):
    h = torch.cat([h, torch.ones_like(h[:, [0]])], -1)
    h = h.view(-1, 3, 3)
    x = torch.cat([x, torch.ones_like(x[:, 0]).unsqueeze(-1)], -1).unsqueeze(-1)
    o = torch.bmm(h, x).squeeze(-1)
    o = o[:, :-1] / o[:, [-1]]
    return o

def apply_affine(x, a):
    """
    Apply affine transformation to coordinates.
    
    Args:
        x: Input coordinates of shape [N, 2] (x, y coordinates)
        a: Affine transformation parameters of shape [N, 6] 
           [a11, a12, a21, a22, tx, ty] where:
           - a11, a12, a21, a22 form the 2x2 transformation matrix
           - tx, ty are the translation components
    
    Returns:
        Transformed coordinates of shape [N, 2]
    """
    # Reshape affine parameters to separate rotation/scaling and translation
    A = a[:, :4].view(-1, 2, 2)  # 2x2 transformation matrix
    t = a[:, 4:6].unsqueeze(-1)  # Translation vector [N, 2, 1]
    
    # Apply transformation: x' = A * x + t
    x_transformed = torch.bmm(A, x.unsqueeze(-1)) + t
    return x_transformed.squeeze(-1)

def jacobian(y, x):
    B, N = y.shape
    jacobian = list()
    for i in range(N):
        v = torch.zeros_like(y)
        v[:, i] = 1.
        dy_i_dx = torch.autograd.grad(y,
                                      x,
                                      grad_outputs=v,
                                      retain_graph=True,
                                      create_graph=True)[0]  # shape [B, N]
        jacobian.append(dy_i_dx)
    jacobian = torch.stack(jacobian, dim=1).requires_grad_()
    return jacobian



class VideoFitting(Dataset):
    def __init__(self, path, transform=None):
        super().__init__()

        self.path = path
        if transform is None:
            self.transform = ToTensor()
        else:
            self.transform = transform

        self.video = self.get_video_tensor()
        self.num_frames, _, self.H, self.W = self.video.size()
        self.pixels = self.video.permute(2, 3, 0, 1).contiguous().view(-1, 3)
        self.coords = get_mgrid([self.H, self.W, self.num_frames])

        shuffle = torch.randperm(len(self.pixels))
        self.pixels = self.pixels[shuffle]
        self.coords = self.coords[shuffle]

    def get_video_tensor(self):
        frames = sorted(os.listdir(self.path))
        video = []
        for i in range(len(frames)):
            img = Image.open(os.path.join(self.path, frames[i]))
            img = self.transform(img)
            video.append(img)
        return torch.stack(video, 0)

    def __len__(self):
        return 1

    def __getitem__(self, idx):    
        if idx > 0: raise IndexError
            
        return self.coords, self.pixels
    

def nir_loss(o, ground_truth):
    
    loss_recon = ((o - ground_truth) ** 2).mean()

    loss = loss_recon

    return loss

class NIR(nn.Module):
    def __init__(self, input_projection, decoder, num_samples, coordinate_dim=2, use_gnll=False):
        super(NIR, self).__init__()

        self.input_projection = input_projection
        self.decoder = decoder

        self.use_gnll = use_gnll

        g = Affine(hidden_features=256, hidden_layers=2)
        # f1 = Siren(in_features=2, out_features=3, hidden_features=256, hidden_layers=4, outermost_linear=True)
        f1 = MLP(input_dim=256, output_dim=3, hidden_dim=256, depth=4)
        # f2 = Siren(in_features=3, out_features=3, hidden_features=256, hidden_layers=4, outermost_linear=True)

        self.g = g
        self.f1 = f1

    def forward(self, x, time_vector, sample_idx=None, scale_factor=None, training=True, lr_frames=None):
        B, H, W, C = x.shape

        x = einops.rearrange(x, 'b h w c -> b (h w) c')

        time_vector = time_vector.repeat(1, H * W, 1)

        xy = x
        t = time_vector

        a = self.g(t)

        # Extract dx and dy from the affine transformation parameters
        # a has shape [B, num_pixels, 6] for affine: [a11, a12, a21, a22, tx, ty]
        # For affine transformations, tx and ty are the translation components
        pred_dx = a[:, 0, 4]  # tx component (index 4)
        pred_dy = a[:, 0, 5]  # ty component (index 5)

        xy_ = apply_affine(xy.squeeze(0), a.squeeze(0))

        if self.input_projection is not None:
            xy_ = self.input_projection(xy_)

        o = self.f1(xy_)

        # Match INR pattern: return shifts as [dx_list, dy_list]
        shifts = [pred_dx, pred_dy]

        return o, shifts