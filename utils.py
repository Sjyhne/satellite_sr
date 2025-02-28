import torch
import torch.nn.functional as F
import cv2
import numpy as np
from typing import Tuple


def apply_shift_cv2(image, dx, dy):
    """Apply translation using cv2 for data generation"""
    rows, cols = image.shape[:2]
    M = np.float32([[1, 0, dx],
                    [0, 1, dy]])
    return cv2.warpAffine(image, M, (cols, rows))

def apply_shift_torch(img, dx, dy):
    """Apply translation to image.
    
    Args:
        img: Input image tensor [B,C,H,W]
        dx: Shift in x direction (in pixels, positive = right) [B]
        dy: Shift in y direction (in pixels, positive = down) [B]
    """
    # Convert pixel shifts to normalized coordinates (-1 to 1)
    dx_norm = 2 * dx / img.shape[3]  # Normalize by width
    dy_norm = 2 * dy / img.shape[2]  # Normalize by height
    
    theta = torch.zeros(img.shape[0], 2, 3, device=img.device)
    theta[:, 0, 0] = 1  # Set diagonal to 1
    theta[:, 1, 1] = 1
    theta[:, 0, 2] = dx_norm  # Set translations
    theta[:, 1, 2] = dy_norm
    
    grid = F.affine_grid(theta, img.size(), align_corners=True)
    output = F.grid_sample(img, grid, mode='bilinear', align_corners=True)
    
    return output

def downsample_cv2(image, size):
    """Downsample using cv2 for data generation"""
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

def bilinear_resize_torch(image, size, antialiasing=True):
    return F.interpolate(image, size=size, mode='bilinear', align_corners=False, antialias=antialiasing)

# from: https://gist.github.com/ProGamerGov/d032aa6780d8ef234f3ce67b177f3c14
def color_transfer(
    input: torch.Tensor,
    source: torch.Tensor,
    mode: str = "pca",
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Transfer the colors from one image tensor to another, so that the target image's
    histogram matches the source image's histogram. Applications for image histogram
    matching includes neural style transfer and astronomy.
    The source image is not required to have the same height and width as the target
    image. Batch and channel dimensions are required to be the same for both inputs.
    Gatys, et al., "Controlling Perceptual Factors in Neural Style Transfer", arXiv, 2017.
    https://arxiv.org/abs/1611.07865
    Args:
        input (torch.Tensor): The NCHW or CHW image to transfer colors from source
            image to from the source image.
        source (torch.Tensor): The NCHW or CHW image to transfer colors from to the
            input image.
        mode (str): The color transfer mode to use. One of 'pca', 'cholesky', or 'sym'.
            Default: "pca"
        eps (float): The desired epsilon value to use.
            Default: 1e-5
    Returns:
        matched_image (torch.tensor): The NCHW input image with the colors of source
            image. Outputs should ideally be clamped to the desired value range to
            avoid artifacts.
    """

    assert input.dim() == 3 or input.dim() == 4
    assert source.dim() == 3 or source.dim() == 4
    input = input.unsqueeze(0) if input.dim() == 3 else input
    source = source.unsqueeze(0) if source.dim() == 3 else source
    assert input.shape[:2] == source.shape[:2]

    # Handle older versions of PyTorch
    torch_cholesky = (
        torch.linalg.cholesky if torch.__version__ >= "1.9.0" else torch.cholesky
    )

    def torch_symeig_eigh(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        torch.symeig() was deprecated in favor of torch.linalg.eigh()
        """
        if torch.__version__ >= "1.9.0":
            L, V = torch.linalg.eigh(x, UPLO="U")
        else:
            L, V = torch.symeig(x, eigenvectors=True, upper=True)
        return L, V

    def get_mean_vec_and_cov(
        x_input: torch.Tensor, eps: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert input images into a vector, subtract the mean, and calculate the
        covariance matrix of colors.
        """
        x_mean = x_input.mean(3).mean(2)[:, :, None, None]

        # Subtract the color mean and convert to a vector
        B, C = x_input.shape[:2]
        x_vec = (x_input - x_mean).reshape(B, C, -1)

        # Calculate covariance matrix
        x_cov = torch.bmm(x_vec, x_vec.permute(0, 2, 1)) / x_vec.shape[2]

        # This line is only important if you get artifacts in the output image
        x_cov = x_cov + (eps * torch.eye(C, device=x_input.device)[None, :])
        return x_mean, x_vec, x_cov

    def pca(x: torch.Tensor) -> torch.Tensor:
        """Perform principal component analysis"""
        eigenvalues, eigenvectors = torch_symeig_eigh(x)
        e = torch.sqrt(torch.diag_embed(eigenvalues.reshape(eigenvalues.size(0), -1)))
        # Remove any NaN values if they occur
        if torch.isnan(e).any():
            e = torch.where(torch.isnan(e), torch.zeros_like(e), e)
        return torch.bmm(torch.bmm(eigenvectors, e), eigenvectors.permute(0, 2, 1))

    # Collect & calculate required values
    _, input_vec, input_cov = get_mean_vec_and_cov(input, eps)
    source_mean, _, source_cov = get_mean_vec_and_cov(source, eps)

    # Calculate new cov matrix for input
    if mode == "pca":
        new_cov = torch.bmm(pca(source_cov), torch.inverse(pca(input_cov)))
    elif mode == "cholesky":
        new_cov = torch.bmm(
            torch_cholesky(source_cov), torch.inverse(torch_cholesky(input_cov))
        )
    elif mode == "sym":
        p = pca(input_cov)
        pca_out = pca(torch.bmm(torch.bmm(p, source_cov), p))
        new_cov = torch.bmm(torch.bmm(torch.inverse(p), pca_out), torch.inverse(p))
    else:
        raise ValueError(
            "mode has to be one of 'pca', 'cholesky', or 'sym'."
            + " Received '{}'.".format(mode)
        )

    # Multiply input vector by new cov matrix
    new_vec = torch.bmm(new_cov, input_vec)

    # Reshape output vector back to input's shape &
    # add the source mean to our output vector
    return new_vec.reshape(input.shape) + source_mean





