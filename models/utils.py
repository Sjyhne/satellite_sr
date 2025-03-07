import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


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


