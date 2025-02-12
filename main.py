import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
torch.manual_seed(0)

from coordinate_mlp import train_model


img = cv2.imread("images/hr_2.JPG")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img / 255.

c = [img.shape[0] // 2, img.shape[1] // 2]
r = 256

img = img[c[0] - r:c[0] + r, c[1] - r:c[1] + r]

plt.imshow(img)
plt.show()

# Create input pixel coordinates in the unit square
coords = np.linspace(0, 1, img.shape[0], endpoint=False)
x_test = np.stack(np.meshgrid(coords, coords), -1)
test_data = [x_test, img]
train_data = [x_test[::2, ::2], img[::2, ::2]]


network_size = (4, 256)
learning_rate = 1e-4
iters = 6000

mapping_size = 256

B_dict = {}

B_gauss = torch.normal(mean=0.0, std=1.0, size=(mapping_size, 2))

print(B_gauss.min(), B_gauss.max())
exit("")

for scale in [10., 100.]:
    B_dict[f"gauss_{scale}"] = B_gauss * scale


outputs = {}
for k in tqdm(B_dict):
    outputs[k] = train_model(network_size, learning_rate, iters, B_dict[k], train_data, test_data)


# Show final network outputs

plt.figure(figsize=(24, 4))
N = len(outputs)
for i, k in enumerate(outputs):
    plt.subplot(1, N+1, i+1)
    plt.imshow(outputs[k]['pred_imgs'][-1])
    plt.title(k)
plt.subplot(1, N+1, N+1)
plt.imshow(img)
plt.title('GT')
plt.savefig("fourier_features.png")

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

plt.savefig("fourier_features_error.png")