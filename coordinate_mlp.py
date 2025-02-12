import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

torch.manual_seed(0)

# Fourier feature mapping
def input_mapping(x, B):
    if B is None:
        return x
    else:
        # Ensure B is a NumPy array for matrix multiplication
        B_np = B.detach().cpu().numpy() if isinstance(B, torch.Tensor) else B
        x_proj = (2. * np.pi * x) @ B_np.T
        return np.concatenate([np.sin(x_proj), np.cos(x_proj)], axis=-1)

# PyTorch network definition
class MLP(nn.Module):
    def __init__(self, input_size, num_layers, num_channels):
        super(MLP, self).__init__()
        layers = [nn.Linear(input_size, num_channels), nn.ReLU()]
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(num_channels, num_channels))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(num_channels, 3))
        layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Train model with given hyperparameters and data
def train_model(network_size, learning_rate, iters, B, train_data, test_data):
    train_x, train_y = train_data
    test_x, test_y = test_data

    # Transform inputs
    train_x = torch.tensor(input_mapping(train_x, B), dtype=torch.float32)
    test_x = torch.tensor(input_mapping(test_x, B), dtype=torch.float32)

    # Get input size from transformed data
    input_size = train_x.shape[-1]

    num_layers, num_channels = network_size
    model = MLP(input_size, num_layers, num_channels)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    train_psnrs = []
    test_psnrs = []
    pred_imgs = []
    xs = []

    train_y = torch.tensor(train_y, dtype=torch.float32)
    test_y = torch.tensor(test_y, dtype=torch.float32)

    for i in tqdm(range(iters), desc='train iter', leave=False):
        model.train()
        optimizer.zero_grad()
        outputs = model(train_x)
        loss = criterion(outputs, train_y)
        loss.backward()
        optimizer.step()

        if i % 25 == 0:
            model.eval()
            with torch.no_grad():
                train_loss = criterion(model(train_x), train_y).item()
                test_loss = criterion(model(test_x), test_y).item()
                train_psnr = -10 * np.log10(2. * train_loss)
                test_psnr = -10 * np.log10(2. * test_loss)
                train_psnrs.append(train_psnr)
                test_psnrs.append(test_psnr)
                pred_imgs.append(model(test_x).numpy())
                xs.append(i)

    return {
        'state': model.state_dict(),
        'train_psnrs': train_psnrs,
        'test_psnrs': test_psnrs,
        'pred_imgs': np.stack(pred_imgs),
        'xs': xs,
    }
