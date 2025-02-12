import torch
import numpy as np
import cv2
import json
from pathlib import Path

from utils import downsample_torch


class FourierData(torch.utils.data.Dataset):
    def __init__(self, hr_img, coords, every_other=False):
        self.hr_img = hr_img
        self.original_coords = coords
        self.coords = coords.clone()
        self.scale = 1.0

        self.every_other = every_other

        if self.every_other:
            self.coords = self.coords[::2, ::2]
            self.hr_img = self.hr_img[::2, ::2]

        # No reshaping needed - keep original image dimensions
        # The MLP will handle [H, W, features] -> [H, W, 3]

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.coords, self.hr_img

    def scale_features(self, scale):
        self.scale = scale
        self.coords = self.original_coords.clone()
        if self.every_other:
            self.coords = self.coords[::2, ::2]
        self.coords = self.coords * scale

    def reset_scaling(self):
        self.scale = 1.0
        self.coords = self.original_coords.clone()
        if self.every_other:
            self.coords = self.coords[::2, ::2]



class SRData(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        """
        Initialize SR dataset from generated data directory.
        
        Args:
            data_dir: Base path to data directory
            mode: 'lr' or 'hr' - which dataset to load
        """
        self.data_dir = Path(data_dir)
        
        # Load transformation log
        with open(self.data_dir / "transform_log.json", 'r') as f:
            self.transform_log = json.load(f)
            
        # Get list of sample names
        self.samples = sorted(list(self.transform_log.keys()))

        # Load original image for reference
        self.original = cv2.imread(str(self.data_dir / "hr_ground_truth.png"))
        self.original = cv2.cvtColor(self.original, cv2.COLOR_BGR2RGB)
        self.original = torch.from_numpy(self.original).float() / 255.0

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample_name = self.samples[idx]
        sample_info = self.transform_log[sample_name]
        sample_id = int(sample_name.split("_")[-1])
        
        # Load transformed image
        img_path = self.data_dir / sample_info['path']
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).float() / 255.0
        
        return {
            'image': img,
            'sample_id': sample_id,
            'transform': {
                'dx_lr': sample_info['dx_pixels_lr'],
                'dy_lr': sample_info['dy_pixels_lr'],
                'dx_hr': sample_info['dx_pixels_hr'],
                'dy_hr': sample_info['dy_pixels_hr'],
                'dx_percent': sample_info['dx_percent'],
                'dy_percent': sample_info['dy_percent']
            }
        }
    
    def get_original_hr(self):
        """Return the original image (before any transformations)"""
        return self.original
    
    def get_random_lr_sample(self):
        """Return a random sample from the dataset"""
        sample_name = self.samples[np.random.randint(len(self))]
        sample_info = self.transform_log[sample_name]
        img = cv2.imread(str(self.data_dir / sample_info['path']))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).float() / 255.0
        return img
    
    def get_downsampled_original_hr(self):
        """Return the downsampled original image"""
        return downsample_torch(self.original, (self.original.shape[1] // 4, self.original.shape[2] // 4))
