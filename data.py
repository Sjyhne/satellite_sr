import os
import torch
import numpy as np
import cv2
import json
from pathlib import Path
import rawpy
import pickle as pkl
import tifffile

from utils import bilinear_resize_torch


def get_dataset(args, name='satburst', keep_in_memory=True):
    """ Returns the dataset object based on the name """
    if name == 'satburst_synth':
        return SRData(data_dir=args.root_satburst_synth, keep_in_memory=keep_in_memory)
    elif name == 'burst_synth':
        return SyntheticBurstVal(data_dir=args.root_burst_synth, sample_id=args.sample_id, keep_in_memory=keep_in_memory)
    elif name == 'worldstrat':
        return WorldStratDatasetFrame(data_dir=args.root_worldstrat, area_name=args.area_name)
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

        if self.keep_in_memory:
            self.images = {}
            for sample in self.samples:
                img_path = self.data_dir / self.transform_log[sample]['path']
                img = cv2.imread(str(img_path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.images[sample] = (torch.from_numpy(img).float() / 255.0).cuda()

        # Load original image for reference
        self.original = cv2.imread(str(self.data_dir / "hr_ground_truth.png"))
        self.original = cv2.cvtColor(self.original, cv2.COLOR_BGR2RGB)
        self.original = (torch.from_numpy(self.original).float() / 255.0).cuda()

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
            img = self.images[sample_name]
        else:
            # Load transformed image
            img_path = self.data_dir / sample_info['path']
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = torch.from_numpy(img).float() / 255.0
        
        return {
            'input': input_coordinates,
            'lr_target': img,
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
        sample_path = self.data_dir / f"sample_{index:02d}.png"
        img = cv2.imread(str(sample_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).float() / 255.0
        img = img.permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
        return img

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
        self.sample_id_str = f"{sample_id:04d}"
        
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
            
            # Pre-load all burst images
            self.burst_images = {}
            for idx in self.frame_indices:
                self.burst_images[idx] = self._read_burst_image(idx)
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
        rgb = np.power(rgb, 1.0/gamma)
        
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
        gt_t = np.power(gt_t, 1.0/gamma)

        gt_t = np.clip(gt_t, 0, 1)

        gt_t = torch.from_numpy(gt_t).float()
        
        return gt_t
    
    def __getitem__(self, idx):
        """Get a specific frame from the burst"""
        # Get the frame index for this position
        frame_idx = self.frame_indices[idx]
        
        # Load the burst image (or get from cache)
        if self.keep_in_memory and self.burst_images is not None:
            burst_img = self.burst_images[frame_idx]
        else:
            burst_img = self._read_burst_image(frame_idx)

        # Return in a format similar to SRData
        return {
            'input': self.get_hr_coordinates(),
            'lr_target': burst_img,  # Keep as [4, H, W]
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
            burst = [self.burst_images[idx] for idx in self.frame_indices]
        else:
            # Load images on demand
            burst = [self._read_burst_image(idx) for idx in self.frame_indices]
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
            return self.burst_images[self.frame_indices[frame_idx]].permute(2, 0, 1)
        else:
            return self._read_burst_image(self.frame_indices[frame_idx]).permute(2, 0, 1)
    
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
    def __init__(self, data_dir, area_name="UNHCR-LBNs006446", num_frames=8):
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
        self.hr_img_size = 512

        # Load high-resolution image
        self.hr_image = self.get_hr()   # Shape: (hr_img_size, hr_img_size, 3)
        self.hr_image = cv2.resize(self.hr_image, (self.hr_img_size, self.hr_img_size))
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
    
    def get_downsampled_original_hr(self):
        """Return the downsampled original image"""
        return downsample_torch(self.hr_image, (self.hr_image.shape[1] // 4, self.hr_image.shape[2] // 4))

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