import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleLinear(nn.Module):
    def __init__(self, input_dim, output_dim, sigmoid_output=False):
        super(SingleLinear, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.sigmoid_output = sigmoid_output

        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.linear(x)

        if self.sigmoid_output:
            x = torch.sigmoid(x)
        
        return x
