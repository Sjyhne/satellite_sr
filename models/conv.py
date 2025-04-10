import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv(nn.Module):
    def __init__(self, input_dim, hidden_dim, depth=4, activation=F.relu, output_dim=3, sigmoid_output=False):
        super(Conv, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.depth = depth
        self.output_dim = output_dim
        self.sigmoid_output = sigmoid_output

        # First layer: input_dim -> hidden_dim
        self.layers = nn.ModuleList([nn.Conv2d(input_dim, hidden_dim, kernel_size=3, padding=1)])
        self.bns = nn.ModuleList([nn.BatchNorm2d(hidden_dim)])
        
        # Hidden layers: hidden_dim -> hidden_dim
        for _ in range(depth - 2):
            self.layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1))
            self.bns.append(nn.BatchNorm2d(hidden_dim))
        # Output layer: hidden_dim -> output_dim
        self.layers.append(nn.Conv2d(hidden_dim, output_dim, kernel_size=3, padding=1))
        
        # Initialize weights (optional)
        self._init_weights()
    
    def _init_weights(self):
        # Optional weight initialization
        for m in self.layers:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)

        # Apply activation to all layers except the last one
        for i, layer in enumerate(self.layers[:-1]):
            x = self.activation(self.bns[i](layer(x)))
        
        # Apply the last layer without activation
        x = self.layers[-1](x)
        
        # Apply sigmoid to output if specified
        if self.sigmoid_output:
            x = torch.sigmoid(x)
        
        x = x.permute(0, 2, 3, 1)

        return x
