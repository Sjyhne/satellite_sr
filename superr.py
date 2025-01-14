import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

import pathlib

# Fourier feature mapping
def input_mapping(x, mapping_size):
    B = torch.normal(mean=0.0, std=1.0, size=(mapping_size, 2)) * 10
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
    

def main():
    img = cv2.imread("images/hr.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.

    coords = np.linspace(0, 1, img.shape[0], endpoint=False)
    x_test = np.stack(np.meshgrid(coords, coords), -1)

    transformed_lr_folder = pathlib.Path("data")
    transformed_lr_filepaths = list(transformed_lr_folder.glob("*.png"))
    transformed_lrs = [cv2.imread(str(filepath)) for filepath in transformed_lr_filepaths]
    transformed_lrs = [cv2.cvtColor(lr, cv2.COLOR_BGR2RGB) for lr in transformed_lrs]
    transformed_lrs = [lr.astype(np.float32) / 255. for lr in transformed_lrs]

    iters = 26
    learning_rate = 1e-4
    mapping_size = 256

    test_data = [input_mapping(x_test, mapping_size), img]
    train_data = [(input_mapping(x_test, mapping_size), transformed_lrs[i]) for i in range(len(transformed_lrs))]

    input_size = train_data[0][0].shape

    num_channels = 256
    num_layers = 4
    input_size = input_size[-1]
    network = MLP(input_size, num_layers, num_channels)
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)

    print(network)

    train_psnrs = []
    test_psnrs = []
    pred_imgs = []
    xs = []

    for _, epoch in tqdm(enumerate(range(iters)), desc="Training", total=iters):
        for x, y in train_data:
            network.train()
            optimizer.zero_grad()

            x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
            y = torch.tensor(y, dtype=torch.float32).unsqueeze(0)

            out = network(x)

            out = out.permute(0, 3, 1, 2)
            out = F.avg_pool2d(out, 2, 2).permute(0, 2, 3, 1).squeeze(0)

            # Ensure y has the same shape as out
            y = y.squeeze(0)

            loss = F.mse_loss(out, y)

            loss.backward()
            optimizer.step()
        
        # Store the last prediction with the gt image in a visualization folder here (every epoch)
        pred_img = network(torch.tensor(train_data[0][0], dtype=torch.float32).unsqueeze(0))
        pred_img = pred_img.permute(0, 3, 1, 2)
        pred_img = F.max_pool2d(pred_img, 2, 2).permute(0, 2, 3, 1).squeeze(0)
        pred_img = pred_img.squeeze(0)  # Remove the batch dimension
        gt_img = train_data[0][1]
        pred_img = pred_img.detach().cpu().numpy()
        gt_img = gt_img
        # Convert images back to [0, 255] for saving
        pred_img_uint8 = (pred_img * 255).astype(np.uint8)
        gt_img_uint8 = (gt_img * 255).astype(np.uint8)
        cv2.imwrite(f"visualization/epoch_{epoch}.png", np.concatenate([pred_img_uint8, gt_img_uint8], axis=1))

        if epoch % 25 == 0:
            network.eval()
            with torch.no_grad():
                train_loss = torch.tensor(0.)
                for x, y in train_data:
                    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
                    y = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
                    train_loss += F.mse_loss(F.avg_pool2d(network(x).permute(0, 3, 1, 2), 2, 2).permute(0, 2, 3, 1), y).item()
                train_loss /= len(train_data)
                test_x, test_y = test_data
                test_x = torch.tensor(test_x, dtype=torch.float32).unsqueeze(0)
                test_y = torch.tensor(test_y, dtype=torch.float32).unsqueeze(0)
                pred_x = network(test_x)
                test_loss = F.mse_loss(pred_x, test_y).item()
                train_psnr = -10 * np.log10(2. * train_loss)
                test_psnr = -10 * np.log10(2. * test_loss)
                train_psnrs.append(train_psnr)
                test_psnrs.append(test_psnr)
                pred_imgs.append(network(torch.tensor(test_data[0], dtype=torch.float32).unsqueeze(0)).numpy())
                xs.append(epoch)

    output = {
        'state': network.state_dict(),
        'train_psnrs': train_psnrs,
        'test_psnrs': test_psnrs,
        'pred_imgs': np.stack(pred_imgs),
        'xs': xs,
    }

    outputs = [output]

    plt.figure(figsize=(24, 4))
    N = len(outputs)
    for i, output in enumerate(outputs):
        plt.subplot(1, N+1, i+1)
        plt.imshow(output['pred_imgs'][-1].squeeze(0))
        plt.title(f'Output {i}')
    plt.subplot(1, N+1, N+1)
    plt.imshow(img)
    plt.title('GT')
    plt.show()
            
if __name__ == "__main__":
    main()