import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, activation=F.relu, num_layers=4, output_dim=3, sigmoid_output=False):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.num_layers = num_layers
        self.output_dim = output_dim

        first_layer = nn.Linear(input_dim, hidden_dim)
        layers = [first_layer]
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.ModuleList(layers)
    
    def forward(self, x):
        for layer in self.net:
            x = self.activation(layer(x))
        
        if self.sigmoid_output:
            x = torch.sigmoid(x)
        
        return x
