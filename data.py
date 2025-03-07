import os
import torch
import numpy as np
import cv2
import json
from pathlib import Path
import rawpy
import pickle as pkl
import tifffile
import random


def get_and_standardize_image(image):
    """Get and standardize image to have zero mean and unit std for each channel"""
    # Check if image has a channel dimension
    if image.dim() >= 3:
        # Calculate mean and std along spatial dimensions only (not across channels)
        # For [C, H, W] format
        if image.dim() == 4:
            mean = image.mean(dim=(1, 2, 3), keepdim=True)
            std = image.std(dim=(1, 2, 3), keepdim=True)
        # For [H, W, C] format
        else:
            mean = image.mean(dim=(0, 1, 2), keepdim=True)
            std = image.std(dim=(0, 1, 2), keepdim=True)
    else:
        # For grayscale without channel dimension
        mean = image.mean()
        std = image.std()
    
    # Avoid division by zero
    std = torch.clamp(std, min=1e-8)
    
    return (image - mean) / std, mean, std

def get_dataset(args, name='satburst', keep_in_memory=True):
    """ Returns the dataset object based on the name """
    if name == 'satburst_synth':
        return SRData(data_dir=args.root_satburst_synth, keep_in_memory=keep_in_memory)
    elif name == 'burst_synth':
        return SyntheticBurstVal(data_dir=args.root_burst_synth, 
                                 sample_id=args.sample_id, keep_in_memory=keep_in_memory)
    elif name == 'worldstrat':
        return WorldStratDatasetFrame(data_dir=args.root_worldstrat, 
                                      area_name=args.area_name, hr_size=args.worldstrat_hr_size)
    else:
        raise ValueError(f"Invalid daaset name: {name}")


class SRData(torch.utils.data.Dataset):
    def __init__(self, data_dir, keep_in_memory=True):
        """
        Initialize SR dataset from generated data directory.
        
        Args:
            data_dir: Base path to data directory
            mode: 'lr' or 'hr' - which dataset to load
        """
        self.data_dir = Path(data_dir)
        self.keep_in_memory = keep_in_memory
        
        # Load transformation log
        with open(self.data_dir / "transform_log.json", 'r') as f:
            self.transform_log = json.load(f)
            
        # Get list of sample names
        self.samples = sorted(list(self.transform_log.keys()))

        self.means = list()
        self.stds = list()
        self.lr_image_sizes = list()

        if self.keep_in_memory:
            self.images = {}
            for sample in self.samples:
                img_path = self.data_dir / self.transform_log[sample]['path']
                img = cv2.imread(str(img_path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = torch.from_numpy(img).float() / 255.0
                img, mean, std = get_and_standardize_image(img)
                self.lr_image_sizes.append(img.shape[1:3])
                self.images[sample] = {
                    "image": img,
                    "mean": mean,
                    "std": std
                }
        
        # Load original image for reference
        self.original = cv2.imread(str(self.data_dir / "hr_ground_truth.png"))
        self.original = cv2.cvtColor(self.original, cv2.COLOR_BGR2RGB)
        self.original = (torch.from_numpy(self.original).float() / 255.0).cuda()
        # Standardize original image to have zero mean and no bias

        self.hr_coords = np.linspace(0, 1, self.original.shape[0], endpoint=False)
        self.hr_coords = np.stack(np.meshgrid(self.hr_coords, self.hr_coords), -1)
        self.hr_coords = torch.FloatTensor(self.hr_coords).cuda()

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample_name = self.samples[idx]
        sample_info = self.transform_log[sample_name]
        sample_id = int(sample_name.split("_")[-1])

        input_coordinates = self.hr_coords

        if self.keep_in_memory:
            img = self.images[sample_name]["image"]
            mean = self.images[sample_name]["mean"]
            std = self.images[sample_name]["std"]
        else:
            # Load transformed image
            img_path = self.data_dir / sample_info['path']
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = torch.from_numpy(img).float() / 255.0
            img, mean, std = get_and_standardize_image(img)
        
        return {
            'input': input_coordinates,
            'lr_target': img,
            'mean': mean,
            'std': std,
            'sample_id': sample_id,
            'shifts': {
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
    

    def get_lr_sample(self, index):
        """Get a specific LR sample by index.
        
        Args:
            index: Sample index (0 is the reference sample)
            
        Returns:
            Tensor of shape [C, H, W] with values in [0, 1]
        """

        if self.keep_in_memory:
            img = self.images[self.samples[index]]["image"]
            mean = self.images[self.samples[index]]["mean"]
            std = self.images[self.samples[index]]["std"]
            # Unstandardize the image
            img = img * std + mean
        else:
            sample_path = self.data_dir / f"sample_{index:02d}.png"
            img = cv2.imread(str(sample_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = torch.from_numpy(img).float() / 255.0

        return img

    def get_lr_mean(self, index):
        return self.images[self.samples[index]]["mean"]

    def get_lr_std(self, index):
        return self.images[self.samples[index]]["std"]

    def get_hr_coordinates(self):
        """Return the high-resolution coordinates."""
        return self.hr_coords


class SyntheticBurstVal(torch.utils.data.Dataset):
    def __init__(self, data_dir, sample_id, keep_in_memory=True):
        """
        Initialize SyntheticBurstVal dataset.
        
        Args:
            data_dir: Base path to SyntheticBurstVal directory
            sample_id: ID of the burst to use (0-299)
            keep_in_memory: Whether to load all images into memory
        """
        self.data_dir = Path(data_dir)
        self.keep_in_memory = keep_in_memory
        self.sample_id = sample_id

        self.rggb = True
        
        # Format sample_id as a 4-digit string with leading zeros
        self.sample_id_str = f"{int(sample_id):04d}"
        
        # Set up paths
        self.gt_dir = self.data_dir / "gt" / self.sample_id_str
        self.burst_dir = self.data_dir / "bursts" / self.sample_id_str
        
        # Find all burst images
        self.burst_paths = sorted(list(self.burst_dir.glob('im_raw_*.png')))
        self.burst_size = len(self.burst_paths)
        
        # Extract frame indices from filenames
        self.frame_indices = []
        for path in self.burst_paths:
            # Extract the frame index from the filename (im_raw_XX.png)
            frame_idx = int(path.stem.split('_')[-1])
            self.frame_indices.append(frame_idx)
        
        # Load ground truth image
        if self.keep_in_memory:
            self.gt_image = self._read_gt_image()
            self.burst_images = {}
            for idx in self.frame_indices:
                img = self._read_burst_image(idx)
                img_std, mean, std = get_and_standardize_image(img)
                self.burst_images[idx] = {
                    "image": img_std,
                    "mean": mean,
                    "std": std
                }
        else:
            self.gt_image = None
            self.burst_images = None
        
        # Create coordinate grid for HR image
        if self.keep_in_memory:
            h, w = self.gt_image.shape[:-1]
            coords_h = np.linspace(0, 1, h, endpoint=False)
            coords_w = np.linspace(0, 1, w, endpoint=False)
            coords = np.stack(np.meshgrid(coords_h, coords_w), -1)
            self.hr_coords = torch.FloatTensor(coords)
        else:
            self.hr_coords = None
        
    def __len__(self):
        return self.burst_size
    
    def _read_burst_image(self, frame_idx):
        """Read a single raw burst image"""
        path = self.burst_dir / f"im_raw_{frame_idx:02d}.png"
        im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        # Convert from 16-bit to float and normalize
        im_t = im.astype(np.float32) / (2**14)

        # Extract RGGB channels
        R = im_t[..., 0]
        G1 = im_t[..., 1]
        G2 = im_t[..., 2]
        B = im_t[..., 3]
        
        # Average the two green channels
        G = (G1 + G2) / 2
        
        # Create RGB image
        rgb = np.stack([R, G, B], axis=-1)
        
        # Apply white balance (example values, actual values might differ)
        wb_gains = np.array([2.0, 1.0, 1.5])  # R, G, B gains
        rgb = rgb * wb_gains
        
        # Apply gamma correction
        gamma = 2.2
        rgb = np.power(np.maximum(rgb, 0), 1.0/gamma)
        
        # Clip values to [0, 1]
        rgb = np.clip(rgb, 0, 1)

        rgb = torch.from_numpy(rgb).float()
        
        return rgb
    
    def _read_gt_image(self):
        """Read the ground truth RGB image"""
        path = self.gt_dir / "im_rgb.png"
        gt = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        # Convert from 16-bit to float and normalize
        gt_t = gt.astype(np.float32) / (2**14)

        wb_gains = np.array([2.0, 1.0, 1.5])  # R, G, B gains
        gt_t = gt_t * wb_gains

        # Apply gamma correction
        gamma = 2.2
        gt_t = np.power(np.maximum(gt_t, 0), 1.0/gamma)

        gt_t = np.clip(gt_t, 0, 1)

        gt_t = torch.from_numpy(gt_t).float()
        
        return gt_t
    
    def __getitem__(self, idx):
        """Get a specific frame from the burst"""
        # Get the frame index for this position
        frame_idx = self.frame_indices[idx]
        
        # Load the burst image (or get from cache)
        if self.keep_in_memory and self.burst_images is not None:
            img = self.burst_images[frame_idx]["image"]
            mean = self.burst_images[frame_idx]["mean"]
            std = self.burst_images[frame_idx]["std"]
        else:
            # Load and standardize on demand
            img = self._read_burst_image(frame_idx)
            img, mean, std = get_and_standardize_image(img)

        # Return in a format similar to SRData
        return {
            'input': self.get_hr_coordinates(),
            'lr_target': img,
            'mean': mean,
            'std': std,
            'sample_id': idx,
            'burst_id': self.sample_id,
            'shifts': {
                'dx_percent': 0.0,  # Placeholder
                'dy_percent': 0.0   # Placeholder
            }
        }
    
    def get_burst(self):
        """Get all frames from the burst as a tensor [N, C, H, W]"""
        if self.keep_in_memory and self.burst_images is not None:
            # Use cached images
            burst = [self.burst_images[idx]["image"] for idx in self.frame_indices]
        else:
            # Load images on demand
            burst = []
            for idx in self.frame_indices:
                img = self._read_burst_image(idx)
                img_std, _, _ = get_and_standardize_image(img)
                burst.append(img_std)
        return torch.stack(burst, 0)
    
    def get_original_hr(self):
        """Return the ground truth image"""
        if self.keep_in_memory and self.gt_image is not None:
            return self.gt_image
        else:
            return self._read_gt_image()
    
    def get_lr_sample(self, frame_idx=0):
        """Get a specific LR frame from the burst"""
        if self.keep_in_memory and self.burst_images is not None:
            # Make sure frame_idx is in range
            if frame_idx >= len(self.frame_indices):
                frame_idx = 0
            idx = self.frame_indices[frame_idx]
            img = self.burst_images[idx]["image"]
            mean = self.burst_images[idx]["mean"]
            std = self.burst_images[idx]["std"]
            # Unstandardize the image
            img = img * std + mean
            return img
        else:
            return self._read_burst_image(self.frame_indices[frame_idx])
    
    def get_lr_mean(self, frame_idx=0):
        return self.burst_images[self.frame_indices[frame_idx]]["mean"]

    def get_lr_std(self, frame_idx=0):
        return self.burst_images[self.frame_indices[frame_idx]]["std"]
    
    def get_hr_coordinates(self):
        """Return coordinates for the HR image"""
        if self.hr_coords is not None:
            return self.hr_coords
            
        # Create on demand if not cached
        gt = self._read_gt_image()
        h, w = gt.shape[1:]
        coords_h = np.linspace(0, 1, h, endpoint=False)
        coords_w = np.linspace(0, 1, w, endpoint=False)
        coords = np.stack(np.meshgrid(coords_h, coords_w), -1)
        return torch.FloatTensor(coords)



class WorldStratDatasetFrame(torch.utils.data.Dataset):
    """ Returns single LR frames in getitem """
    def __init__(self, data_dir, area_name="UNHCR-LBNs006446", num_frames=8, hr_size=None):
        """
        Args:
            data_dir (str): Path to the dataset.
            area_name (str): area name.
        """

        self.dataset_root = '/home/nlang/data/worldstrat_kaggle'
        self.hr_dataset = "{}/hr_dataset/12bit".format(data_dir)
        self.lr_dataset = "{}/lr_dataset".format(data_dir)
        #self.metadata_df = pd.read_csv("{}/metadata.csv".format(dataset_root))

        self.area_name = area_name
        self.num_frames = num_frames    
        self.hr_size = hr_size

        # Load high-resolution image
        self.hr_image = self.get_hr()   # Shape: (hr_img_size, hr_img_size, 3)
        if self.hr_size is not None:
            self.hr_image = cv2.resize(self.hr_image, (self.hr_size, self.hr_size), interpolation=cv2.INTER_AREA)
        self.hr_image = torch.tensor(self.hr_image)
        

        # Create input coordinate grid that matches the HR image
        self.hr_coords = np.linspace(0, 1, self.hr_image.shape[0], endpoint=False)
        self.hr_coords = np.stack(np.meshgrid(self.hr_coords, self.hr_coords), -1)
        self.hr_coords = torch.FloatTensor(self.hr_coords)
    
    def __len__(self):
        # TODO
        return self.num_frames
    
    def get_hr(self):
        """Loads and processes the high-resolution image."""
        hr_rgb_path = os.path.join(self.hr_dataset, self.area_name, f"{self.area_name}_rgb.png")
        print(hr_rgb_path)
        hr_rgb_img = cv2.imread(hr_rgb_path)
        print(hr_rgb_img.shape)
        hr_rgb_img = cv2.cvtColor(hr_rgb_img, cv2.COLOR_BGR2RGB)
        return hr_rgb_img.astype(np.float32) / 255.0  # Normalize
    
    def get_lr(self, frame_id):
        """Loads a single LR frame."""

        # files start with index 1 (not 0)
        frame_id+=1

        lr_sample_path = os.path.join(self.lr_dataset, self.area_name, "L2A")
        lr_rgb_path = os.path.join(lr_sample_path, f"{self.area_name}-{frame_id}-L2A_data.tiff")
        lr_rgb_img = tifffile.imread(lr_rgb_path)[:, :, 4:1:-1].copy()  # Select RGB bands and reverse order
        lr_rgb_img = torch.tensor(lr_rgb_img, dtype=torch.float32).clip(0, 1)  # Data is already normalized, but needs to be clipped

        return lr_rgb_img
    
    def __getitem__(self, idx):
        lr_image = self.get_lr(frame_id=idx)  # Shape: (8, lr_img_size, lr_img_size, 3)
        
        # Convert to torch tensors
        lr_image = torch.tensor(lr_image)
        
        return {
            'input': self.hr_coords,
            'lr_target': lr_image,
            'sample_id': idx,
            # note: the true shifts are unknown, set to default 0
            'shifts': {
                'dx_lr': 0,
                'dy_lr': 0,
                'dx_hr': 0,
                'dy_hr': 0,
                'dx_percent': 0,
                'dy_percent': 0
            }
        }
    
    def get_original_hr(self):
        """Return the original image (before any transformations)"""
        return self.hr_image
    
    def get_hr_coordinates(self):
        """Return the high-resolution coordinates."""
        return self.hr_coords
    
    def get_lr_sample(self, index):
        """Get a specific LR sample by index.
        
        Args:
            index: Sample index (0 is the reference sample)
            
        Returns:
            Tensor of shape [C, H, W] with values in [0, 1]
        """
        return self.get_lr(index).permute(2, 0, 1)
    

if __name__ == "__main__":
    dataset = SyntheticBurstVal("SyntheticBurstVal", 0)


    for i in range(len(dataset)):
        print(dataset[i]['image'].shape)
        exit("")