import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from models.mlp import MLP
from models.siren import Siren
from models.wire import Wire
from models.linear import SingleLinear
from models.conv import Conv
from models.thera import Thera

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


def get_decoder(network_name, network_depth, input_dim, network_hidden_dim, output_dim=3):
    if network_name == "mlp":
        return MLP(input_dim=input_dim, hidden_dim=network_hidden_dim, depth=network_depth, output_dim=output_dim)
    elif network_name == "siren":
        return Siren(input_dim=2, hidden_dim=network_hidden_dim, depth=network_depth, output_dim=output_dim)
    elif network_name == "wire":
        return Wire(input_dim=2, hidden_dim=network_hidden_dim, depth=network_depth, output_dim=output_dim)
    elif network_name == "linear":
        return SingleLinear(input_dim=input_dim, output_dim=output_dim)
    elif network_name == "conv":
        return Conv(input_dim=input_dim, hidden_dim=network_hidden_dim, depth=network_depth, output_dim=output_dim)
    elif network_name == "thera":
        return Thera(input_dim=input_dim, hidden_dim=network_hidden_dim, depth=network_depth, output_dim=output_dim)
    elif network_name == "nir":
        return NIR(input_dim=input_dim, hidden_dim=network_hidden_dim, depth=network_depth, output_dim=output_dim)
    else:
        raise ValueError(f"Network name {network_name} not recognized")

