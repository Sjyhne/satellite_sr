import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import imageio.v2 as imageio

# Set random seed for reproducibility
torch.manual_seed(0)
device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')

# Download image, take a square crop from the center
image_url = 'https://live.staticflickr.com/7492/15677707699_d9d67acf9d_b.jpg'
# img = imageio.imread(image_url)[..., :3] / 255.

filename = "hr_image.png"
img = imageio.imread(f"images/{filename}")[..., :3] / 255
c = [img.shape[0]//2, img.shape[1]//2]
r = 256
img = img[c[0]-r:c[0]+r, c[1]-r:c[1]+r]

every_other = True


# Create input pixel coordinates in the unit square
coords = np.linspace(0, 1, img.shape[0], endpoint=False)
x_test = np.stack(np.meshgrid(coords, coords), -1)
x_test = torch.FloatTensor(x_test).to(device)
test_data = [x_test, torch.FloatTensor(img).to(device)]
if every_other:
    train_data = [x_test[::2, ::2], torch.FloatTensor(img[::2, ::2]).to(device)]
else:
    train_data = [x_test, torch.FloatTensor(img).to(device)]

# Fourier feature mapping
def input_mapping(x, B):
    if B is None:
        return x
    else:
        x_proj = (2. * np.pi * x) @ B.T
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

# PyTorch network definition
class FourierNetwork(nn.Module):
    def __init__(self, input_dim, num_layers, num_channels):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # First layer
        self.layers.append(nn.Linear(input_dim, num_channels))
        
        # Hidden layers
        for i in range(num_layers - 2):
            self.layers.append(nn.Linear(num_channels, num_channels))
            
        # Output layer
        self.layers.append(nn.Linear(num_channels, 3))
        
    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = torch.nn.functional.relu(layer(x))
        return torch.sigmoid(self.layers[-1](x))

# Train model with given hyperparameters and data
def train_model(network_size, learning_rate, iters, B, train_data, test_data):
    input_dim = input_mapping(train_data[0], B).shape[-1]
    model = FourierNetwork(input_dim, *network_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    def model_pred(x):
        return model(input_mapping(x, B))
    
    def model_loss(x, y):
        return 0.5 * torch.mean((model_pred(x) - y) ** 2)
    
    def model_psnr(x, y):
        return -10 * torch.log10(2. * model_loss(x, y))
    
    train_psnrs = []
    test_psnrs = []
    pred_imgs = []
    xs = []
    
    for i in tqdm(range(iters), desc='train iter', leave=False):
        optimizer.zero_grad()
        loss = model_loss(train_data[0], train_data[1])
        loss.backward()
        optimizer.step()
        
        if i % 25 == 0:
            with torch.no_grad():
                train_psnrs.append(model_psnr(train_data[0], train_data[1]).item())
                test_psnrs.append(model_psnr(test_data[0], test_data[1]).item())
                pred_imgs.append(model_pred(test_data[0]).cpu().numpy())
                xs.append(i)
    
    return {
        'model': model,
        'train_psnrs': train_psnrs,
        'test_psnrs': test_psnrs,
        'pred_imgs': np.stack(pred_imgs),
        'xs': xs,
    }

network_size = (4, 256)
learning_rate = 1e-4
iters = 5000
mapping_size = 256

B_dict = {}
# Standard network - no mapping
# B_dict['none'] = None
# Basic mapping
# B_dict['basic'] = torch.eye(2).to(device)
# Three different scales of Gaussian Fourier feature mappings
B_gauss = torch.randn(mapping_size, 2).to(device)
for scale in [1, 5, 10]:
    B_dict[f'gauss_{scale}'] = B_gauss * scale

# Train all models
outputs = {}
for k in tqdm(B_dict):
    outputs[k] = train_model(network_size, learning_rate, iters, B_dict[k], train_data, test_data)

# Find best performing model based on final test PSNR
best_psnr = -float('inf')
best_model = None
for k in outputs:
    final_psnr = outputs[k]['test_psnrs'][-1]
    if final_psnr > best_psnr:
        best_psnr = final_psnr
        best_model = k

# Calculate PSNR for downsampled+upsampled baseline
img_tensor = torch.FloatTensor(img).to(device)
downsampled = img_tensor[::2, ::2]  # Downsample by factor of 2
upsampled = torch.nn.functional.interpolate(
    downsampled.permute(2, 0, 1).unsqueeze(0),
    size=img_tensor.shape[:2],
    mode='bilinear',
    align_corners=False
)[0].permute(1, 2, 0)
baseline_mse = torch.mean((upsampled - img_tensor) ** 2)
baseline_psnr = -10 * torch.log10(2 * baseline_mse)

# Show final network outputs
N = len(outputs)  # Number of models
num_cols = 5  # Number of plots per row
num_rows = (N + 4) // num_cols + 1  # +2 for train and GT, round up
plt.figure(figsize=(24, 8 * num_rows))

# Plot all model outputs
for i, k in enumerate(outputs):
    plt.subplot(num_rows, num_cols, i + 1)
    plt.imshow(outputs[k]['pred_imgs'][-1])
    final_psnr = outputs[k]['test_psnrs'][-1]
    plt.title(f'{k}\nPSNR: {final_psnr:.2f}')
    plt.axis('off')

# Add downsampled+upsampled baseline image
plt.subplot(num_rows, num_cols, N + 1)
plt.imshow(upsampled.cpu())
plt.title(f'Downsampled+Upsampled\nPSNR: {baseline_psnr:.2f}')
plt.axis('off')

# Add ground truth
plt.subplot(num_rows, num_cols, N + 2)
plt.imshow(img)
plt.title('GT\n(Full Resolution)')
plt.axis('off')

plt.suptitle(f'Best Model: {best_model} (PSNR: {best_psnr:.2f})\nBaseline PSNR: {baseline_psnr:.2f}', fontsize=16, y=0.95)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for suptitle
plt.subplots_adjust(wspace=0.3, hspace=0.4)  # Add more space between subplots
plt.savefig(f'outputs_{filename}_every_other_{every_other}.png', bbox_inches='tight', dpi=300)
plt.show()

# Plot train/test error curves
plt.figure(figsize=(16, 6))

plt.subplot(121)
for i, k in enumerate(outputs):
    plt.plot(outputs[k]['xs'], outputs[k]['train_psnrs'], label=k)
plt.title('Train error')
plt.ylabel('PSNR')
plt.xlabel('Training iter')
plt.legend()

plt.subplot(122)
for i, k in enumerate(outputs):
    plt.plot(outputs[k]['xs'], outputs[k]['test_psnrs'], label=k)
plt.title('Test error')
plt.ylabel('PSNR')
plt.xlabel('Training iter')
plt.legend()

plt.savefig(f'training_convergence_{filename}_every_other_{every_other}.png')