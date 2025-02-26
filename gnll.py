import numpy as np
import matplotlib.pyplot as plt

# Define the dimensions of the image (for example, 128x128)
H, W = 128, 128

# Create a coordinate grid
x = np.linspace(-1, 1, W)
y = np.linspace(-1, 1, H)
xx, yy = np.meshgrid(x, y)

# Define the standard deviation of the Gaussian (controls the spread)
sigma = 0.3

# Create the Gaussian weighting mask centered at (0, 0)
gaussian_weights = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))

# Visualize the weighting mask
plt.figure(figsize=(6, 6))
plt.imshow(gaussian_weights, cmap='hot', extent=[-1,1,-1,1])
plt.colorbar(label='Pixel Weight')
plt.title('Gaussian Pixel Weighting Mask')
plt.xlabel('x-coordinate')
plt.ylabel('y-coordinate')
plt.savefig('gaussian_weights.png')
