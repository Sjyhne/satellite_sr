import argparse
import os
import numpy as np
from pathlib import Path
import rawpy
import matplotlib.pyplot as plt
import shutil
import cv2
import json
from utils import downsample_torch
import torch
import tifffile
import subprocess
import sys

def check_exiftool():
    """Check if ExifTool is installed."""
    try:
        subprocess.run(['exiftool', '-ver'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except FileNotFoundError:
        return False

def tiff_to_dng(tiff_path, output_path=None, copy_metadata_from=None):
    """
    Convert a TIFF file to DNG format, preserving as much metadata as possible.
    
    Args:
        tiff_path: Path to the TIFF file
        output_path: Path to save the DNG file (default: same as TIFF but with .dng extension)
        copy_metadata_from: Path to a DNG file to copy metadata from (optional)
        
    Returns:
        Path to the created DNG file or None if conversion failed
    """
    tiff_path = Path(tiff_path)
    
    # Default output path
    if output_path is None:
        output_path = tiff_path.with_suffix('.dng')
    else:
        output_path = Path(output_path)
    
    # Check if ExifTool is installed
    if not check_exiftool():
        print("Error: ExifTool is not installed. Please install it first.")
        print("Installation options:")
        print("  - Using conda: conda install -c conda-forge exiftool")
        print("  - On Ubuntu/Debian: sudo apt-get install libimage-exiftool-perl")
        print("  - On macOS: brew install exiftool")
        print("  - On Windows: Download from https://exiftool.org/")
        return None
    
    # Load metadata from JSON
    json_path = tiff_path.with_suffix('.json')
    if not json_path.exists():
        print(f"Error: Metadata file {json_path} not found.")
        return None
    
    with open(json_path, 'r') as f:
        metadata = json.load(f)
    
    # Load TIFF data
    raw_data = tifffile.imread(str(tiff_path))
    
    # Check if output file already exists and remove it if it does
    if output_path.exists():
        try:
            output_path.unlink()
            print(f"Removed existing file: {output_path}")
        except Exception as e:
            print(f"Warning: Could not remove existing file {output_path}: {e}")
            # Try using a different filename
            output_path = output_path.with_name(f"new_{output_path.name}")
            print(f"Using alternative output path: {output_path}")
    
    # Create a temporary TIFF file with the correct orientation and metadata
    temp_tiff = tiff_path.with_name(f"temp_{tiff_path.name}")
    tifffile.imwrite(str(temp_tiff), raw_data, photometric='minisblack')
    
    # Convert TIFF to DNG using ExifTool
    cmd = ['exiftool', '-DNGVersion=1.4.0.0', '-PhotometricInterpretation=BlackIsZero']
    
    # Add basic metadata
    if 'camera_make' in metadata:
        cmd.extend([f'-Make={metadata["camera_make"]}'])
    if 'camera_model' in metadata:
        cmd.extend([f'-Model={metadata["camera_model"]}'])
    
    # Add color-related metadata if available
    if 'camera_white_balance' in metadata and metadata['camera_white_balance']:
        wb = metadata['camera_white_balance']
        if isinstance(wb, list) and len(wb) >= 3:
            try:
                cmd.extend([f'-AsShotNeutral={float(wb[0]):.6f} {float(wb[1]):.6f} {float(wb[2]):.6f}'])
            except (ValueError, TypeError):
                print(f"Warning: Could not convert white balance values to float: {wb}")
    
    # Add black and white levels
    if 'black_level' in metadata:
        try:
            cmd.extend([f'-BlackLevel={int(metadata["black_level"])}'])
        except (ValueError, TypeError):
            print(f"Warning: Could not convert black level to integer: {metadata['black_level']}")
    if 'white_level' in metadata:
        try:
            cmd.extend([f'-WhiteLevel={int(metadata["white_level"])}'])
        except (ValueError, TypeError):
            print(f"Warning: Could not convert white level to integer: {metadata['white_level']}")
    
    # Convert to DNG
    cmd.extend(['-o', str(output_path), str(temp_tiff)])
    
    try:
        # Run with verbose output to help diagnose issues
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error converting to DNG. ExifTool output:")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return None
        print(f"Created DNG file: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error executing ExifTool command: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error during DNG conversion: {e}")
        return None
    finally:
        # Clean up temporary file
        if temp_tiff.exists():
            temp_tiff.unlink()
    
    # If we have a reference DNG, copy additional metadata from it
    if copy_metadata_from and Path(copy_metadata_from).exists():
        try:
            # Copy metadata tags from the reference DNG
            cmd = [
                'exiftool', '-TagsFromFile', str(copy_metadata_from),
                '-ColorMatrix1', '-ColorMatrix2', '-ForwardMatrix1', '-ForwardMatrix2',
                '-CameraCalibration1', '-CameraCalibration2', '-ReductionMatrix1', '-ReductionMatrix2',
                '-AnalogBalance', '-AsShotNeutral', '-BaselineExposure', '-BaselineNoise', '-BaselineSharpness',
                '-LinearResponseLimit', '-NoiseProfile', '-OpcodeList1', '-OpcodeList2', '-OpcodeList3',
                str(output_path)
            ]
            subprocess.run(cmd, check=False, capture_output=True, text=True)
            print(f"Copied metadata from reference DNG: {copy_metadata_from}")
        except Exception as e:
            print(f"Warning: Could not copy metadata from reference DNG: {e}")
    
    return output_path

def extract_region_from_dng(dng_path, output_dir, region, downsample_factors=None, save_png=True, convert_to_dng=False):
    """
    Extract a region from a DNG file and save it as PNG and TIFF files.
    
    Args:
        dng_path: Path to the source DNG file
        output_dir: Directory to save output files
        region: Tuple of (x_start, y_start, width, height) defining the region to extract
        downsample_factors: List of factors to downsample the image (e.g., [1, 2, 4])
        save_png: Whether to save PNG previews
        convert_to_dng: Whether to convert TIFF files to DNG format
    """
    x_start, y_start, width, height = region
    
    # Default to no downsampling if not specified
    if downsample_factors is None:
        downsample_factors = [1, 2, 4, 8]
    
    # Create subdirectories
    preview_dir = output_dir / "preview"
    raw_dir = output_dir / "raw"
    metadata_dir = output_dir / "metadata"
    preview_dir.mkdir(exist_ok=True)
    raw_dir.mkdir(exist_ok=True)
    metadata_dir.mkdir(exist_ok=True)
    
    # Define output path for metadata
    metadata_path = metadata_dir / Path(dng_path).with_suffix('.json').name
    
    # Read the original DNG
    with rawpy.imread(str(dng_path)) as raw:
        # Get the raw data
        raw_data = raw.raw_image_visible
        
        # Extract the region
        region_data = raw_data[y_start:y_start+height, x_start:x_start+width].copy()
        
        # Process the image for PNG preview
        rgb_full = raw.postprocess(use_camera_wb=True)
        region_rgb = rgb_full[y_start:y_start+height, x_start:x_start+width]
        
        # Helper function to safely convert to list
        def safe_to_list(value):
            if value is None:
                return None
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, list):
                return value
            return value
        
        # Extract as much noise profile information as possible
        noise_profile = {}
        
        # Try to extract noise profile data
        if hasattr(raw, 'noise_profile'):
            noise_profile['noise_profile'] = safe_to_list(raw.noise_profile)
        
        # Extract ISO information if available
        if hasattr(raw, 'metadata') and hasattr(raw.metadata, 'iso'):
            noise_profile['iso'] = raw.metadata.iso
        
        # Extract other potentially useful noise-related attributes
        for attr in ['black_level_per_channel', 'white_level', 'camera_whitebalance', 
                    'color_matrix', 'rgb_xyz_matrix', 'raw_pattern', 'daylight_whitebalance']:
            if hasattr(raw, attr):
                noise_profile[attr] = safe_to_list(getattr(raw, attr))
        
        # Get metadata
        metadata = {
            "black_level": safe_to_list(raw.black_level_per_channel[0] if hasattr(raw, 'black_level_per_channel') else 0),
            "white_level": raw.white_level if hasattr(raw, 'white_level') else 65535,
            "camera_white_balance": safe_to_list(raw.camera_whitebalance if hasattr(raw, 'camera_whitebalance') else None),
            "color_matrix": safe_to_list(raw.color_matrix if hasattr(raw, 'color_matrix') else None),
            "original_file": str(dng_path.name),
            "region": {
                "x": x_start,
                "y": y_start,
                "width": width,
                "height": height
            },
            "noise_profile": noise_profile
        }
        
        # Try to extract camera make and model
        if hasattr(raw, 'metadata'):
            if hasattr(raw.metadata, 'make'):
                metadata['camera_make'] = raw.metadata.make
            if hasattr(raw.metadata, 'model'):
                metadata['camera_model'] = raw.metadata.model
    
    # Save PNG previews and raw TIFF files for each factor
    png_paths = []
    tiff_paths = []
    dng_paths = []
    
    if save_png:
        # Convert to torch tensor for downsampling
        region_tensor = torch.from_numpy(region_rgb).float().permute(2, 0, 1).unsqueeze(0)
        
        for factor in downsample_factors:
            # Create subdirectories for this factor
            factor_preview_dir = preview_dir / f"factor_{factor}x"
            factor_raw_dir = raw_dir / f"factor_{factor}x"
            factor_preview_dir.mkdir(exist_ok=True)
            factor_raw_dir.mkdir(exist_ok=True, parents=True)
            
            # Define output paths for this factor
            png_output_path = factor_preview_dir / Path(dng_path).with_suffix('.png').name
            tiff_output_path = factor_raw_dir / Path(dng_path).with_suffix('.tiff').name
            json_output_path = factor_raw_dir / Path(dng_path).with_suffix('.json').name
            
            if factor == 1:
                # No downsampling needed for preview
                output_rgb = region_rgb
                
                # Save the raw region data as TIFF
                # Use 32-bit float to preserve full precision
                tifffile.imwrite(str(tiff_output_path), region_data, photometric='minisblack')
                
                # Save metadata for this factor
                factor_metadata = metadata.copy()
                factor_metadata["downsampling_factor"] = factor
                
                with open(json_output_path, 'w') as f:
                    json.dump(factor_metadata, f, indent=2)
                
                # Also copy the original DNG to this directory for reference
                dng_copy_path = factor_raw_dir / Path(dng_path).name
                shutil.copy(dng_path, dng_copy_path)
                
                # Convert TIFF to DNG if requested
                if convert_to_dng:
                    dng_output_path = factor_raw_dir / Path(dng_path).with_suffix('.dng').name
                    try:
                        converted_dng_path = tiff_to_dng(tiff_output_path, dng_output_path, copy_metadata_from=dng_path)
                        if converted_dng_path:
                            dng_paths.append(converted_dng_path)
                    except Exception as e:
                        print(f"Error converting TIFF to DNG: {e}")
                        # Continue without DNG conversion
            else:
                # Downsample preview using torch function
                h, w = region_tensor.shape[2], region_tensor.shape[3]
                downsampled = downsample_torch(region_tensor, (h // factor, w // factor))
                
                # Convert back to numpy for preview
                output_rgb = downsampled[0].permute(1, 2, 0).numpy().astype(np.uint8)
                
                # Downsample raw data
                # For raw data, we need to be careful with downsampling to preserve the Bayer pattern
                # Here we use a simple approach - take every Nth pixel in both dimensions
                # This preserves the Bayer pattern but might not be optimal for all sensors
                downsampled_raw = region_data[::factor, ::factor]
                
                # Save the downsampled raw data as TIFF
                tifffile.imwrite(str(tiff_output_path), downsampled_raw, photometric='minisblack')
                
                # Save metadata for this factor
                factor_metadata = metadata.copy()
                factor_metadata["downsampling_factor"] = factor
                factor_metadata["downsampling_method"] = "pixel_skipping"
                factor_metadata["region"]["width"] = downsampled_raw.shape[1]
                factor_metadata["region"]["height"] = downsampled_raw.shape[0]
                
                with open(json_output_path, 'w') as f:
                    json.dump(factor_metadata, f, indent=2)
                
                # Convert TIFF to DNG if requested
                if convert_to_dng:
                    dng_output_path = factor_raw_dir / Path(dng_path).with_suffix('.dng').name
                    try:
                        converted_dng_path = tiff_to_dng(tiff_output_path, dng_output_path, copy_metadata_from=dng_path)
                        if converted_dng_path:
                            dng_paths.append(converted_dng_path)
                    except Exception as e:
                        print(f"Error converting TIFF to DNG: {e}")
                        # Continue without DNG conversion
            
            # Convert from RGB to BGR for OpenCV
            output_bgr = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)
            
            # Save the preview image
            cv2.imwrite(str(png_output_path), output_bgr)
            png_paths.append(png_output_path)
            tiff_paths.append(tiff_output_path)
    
    # Save the overall metadata
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return region_rgb, png_paths, tiff_paths, dng_paths

def process_burst_folder(input_folder, output_folder, region, downsample_factors=None, convert_to_dng=False):
    """
    Process all DNG files in a folder, extract a region from each, and save to output folder.
    
    Args:
        input_folder: Path to folder containing DNG files
        output_folder: Path to save extracted regions
        region: Tuple of (x_start, y_start, width, height)
        downsample_factors: List of factors to downsample the image (e.g., [1, 2, 4])
        convert_to_dng: Whether to convert TIFF files to DNG format
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Default to no downsampling if not specified
    if downsample_factors is None:
        downsample_factors = [1]  # Original resolution only
    
    # Create subdirectories
    preview_dir = output_path / "preview"
    raw_dir = output_path / "raw"
    metadata_dir = output_path / "metadata"
    preview_dir.mkdir(exist_ok=True)
    raw_dir.mkdir(exist_ok=True)
    metadata_dir.mkdir(exist_ok=True)
    
    # Create raw directories for each factor
    for factor in downsample_factors:
        factor_raw_dir = raw_dir / f"factor_{factor}x"
        factor_raw_dir.mkdir(exist_ok=True, parents=True)
    
    # Find all DNG files
    dng_files = sorted(list(input_path.glob('*.dng')))
    
    if not dng_files:
        print(f"No DNG files found in {input_folder}")
        return
    
    print(f"Found {len(dng_files)} DNG files")
    print(f"Applying downsampling factors: {downsample_factors}")
    
    # Check if ExifTool is installed if we need to convert to DNG
    if convert_to_dng and not check_exiftool():
        print("Warning: ExifTool is not installed. Cannot convert TIFF files to DNG.")
        print("  - On Ubuntu/Debian: sudo apt-get install libimage-exiftool-perl")
        print("  - On macOS: brew install exiftool")
        print("  - On Windows: Download from https://exiftool.org/")
        convert_to_dng = False
    
    # Extract the region from each DNG
    all_png_paths = []
    all_tiff_paths = []
    all_dng_paths = []
    for i, dng_file in enumerate(dng_files):
        print(f"Processing {i+1}/{len(dng_files)}: {dng_file.name}")
        
        # Extract and save the region
        _, png_paths, tiff_paths, dng_paths = extract_region_from_dng(
            dng_file, 
            output_path, 
            region, 
            downsample_factors=downsample_factors, 
            save_png=True,
            convert_to_dng=convert_to_dng
        )
        all_png_paths.extend(png_paths)
        all_tiff_paths.extend(tiff_paths)
        all_dng_paths.extend(dng_paths)
    
    # Save the region information to a JSON file
    region_info = {
        "x_start": region[0],
        "y_start": region[1],
        "width": region[2],
        "height": region[3],
        "downsample_factors": downsample_factors,
        "files_processed": [f.name for f in dng_files]
    }
    
    with open(output_path / "region_info.json", 'w') as f:
        json.dump(region_info, f, indent=2)
    
    # Create a montage for each downsampling factor
    for factor in downsample_factors:
        factor_dir = preview_dir / f"factor_{factor}x"
        factor_png_paths = [p for p in all_png_paths if f"factor_{factor}x" in str(p)]
        
        if factor_png_paths:
            montage_path = factor_dir / "montage.png"
            create_montage(factor_png_paths, montage_path)
            print(f"Montage for {factor}x downsampling saved to {montage_path}")
    
    # Create helper script for loading TIFF files
    create_helper_script(output_path)
    
    print(f"Extracted regions saved to:")
    for factor in downsample_factors:
        print(f"  - Raw TIFF files ({factor}x): {output_path / 'raw' / f'factor_{factor}x'}")
        if convert_to_dng:
            print(f"  - DNG files ({factor}x): {output_path / 'raw' / f'factor_{factor}x'} (converted from TIFF)")
    print(f"  - Previews: {preview_dir} (with subdirectories for each downsampling factor)")
    print(f"  - Metadata: {metadata_dir}")
    print(f"Region information saved to {output_path / 'region_info.json'}")
    print(f"Helper script created: {output_path / 'load_raw_tiff.py'} (for loading and visualizing TIFF files)")

def create_helper_script(output_dir):
    """Create a helper script to load the TIFF files."""
    script_content = '''
import numpy as np
import json
import tifffile
from pathlib import Path
import matplotlib.pyplot as plt

def load_raw_tiff(tiff_path):
    """
    Load a raw TIFF file and its accompanying metadata.
    
    Args:
        tiff_path: Path to the TIFF file
        
    Returns:
        raw_data: The raw data from the TIFF file
        metadata: The metadata from the JSON file
    """
    tiff_path = Path(tiff_path)
    json_path = tiff_path.with_suffix('.json')
    
    # Load raw data
    raw_data = tifffile.imread(str(tiff_path))
    
    # Load metadata
    with open(json_path, 'r') as f:
        metadata = json.load(f)
    
    return raw_data, metadata

def simple_postprocess(raw_data, metadata):
    """
    A very simple postprocessing function to visualize raw data.
    This is not a proper demosaicing algorithm, just for visualization.
    
    Args:
        raw_data: Raw data from the TIFF file
        metadata: Metadata from the JSON file
        
    Returns:
        rgb_image: A simple RGB visualization of the raw data
    """
    # Normalize the data
    black_level = metadata.get("black_level", 0)
    white_level = metadata.get("white_level", 65535)
    
    # Clip to valid range
    raw_data = np.clip(raw_data, black_level, white_level)
    
    # Normalize to [0, 1]
    normalized = (raw_data - black_level) / (white_level - black_level)
    
    # Simple visualization (not proper demosaicing)
    # Just duplicate the raw data to all channels
    rgb_image = np.stack([normalized, normalized, normalized], axis=2)
    
    return rgb_image

# Example usage:
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python load_raw_tiff.py path/to/tiff/file.tiff")
        sys.exit(1)
    
    tiff_path = sys.argv[1]
    
    # Load raw data and metadata
    raw_data, metadata = load_raw_tiff(tiff_path)
    
    # Simple visualization
    rgb_image = simple_postprocess(raw_data, metadata)
    
    # Display the image
    plt.figure(figsize=(10, 8))
    plt.imshow(rgb_image)
    plt.title(f"Raw data from {Path(tiff_path).name} - Factor: {metadata.get('downsampling_factor', 1)}x")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    print(f"Loaded raw data with shape: {raw_data.shape}")
    print(f"Downsampling factor: {metadata.get('downsampling_factor', 1)}x")
    print(f"Original file: {metadata.get('original_file', 'unknown')}")
    
    # Print noise profile information if available
    if 'noise_profile' in metadata:
        print("\nNoise Profile Information:")
        for key, value in metadata['noise_profile'].items():
            print(f"  {key}: {value}")
'''
    
    # Write the helper script
    with open(output_dir / 'load_raw_tiff.py', 'w') as f:
        f.write(script_content)

def create_montage(image_paths, output_path, max_images_per_row=5):
    """
    Create a montage of images for easy visual inspection.
    
    Args:
        image_paths: List of paths to images
        output_path: Path to save the montage
        max_images_per_row: Maximum number of images per row
    """
    # Read all images
    images = [cv2.imread(str(path)) for path in image_paths]
    
    if not images:
        return
    
    # Determine grid size
    n_images = len(images)
    n_cols = min(n_images, max_images_per_row)
    n_rows = (n_images + n_cols - 1) // n_cols  # Ceiling division
    
    # Get image dimensions (assuming all images have the same size)
    h, w, c = images[0].shape
    
    # Create montage
    montage = np.zeros((h * n_rows, w * n_cols, c), dtype=np.uint8)
    
    # Fill montage
    for i, img in enumerate(images):
        row = i // n_cols
        col = i % n_cols
        montage[row*h:(row+1)*h, col*w:(col+1)*w] = img
    
    # Save montage
    cv2.imwrite(str(output_path), montage)

def main():
    parser = argparse.ArgumentParser(description="Extract regions from burst DNG files")
    parser.add_argument("--input", type=str, required=True, help="Input folder containing DNG files")
    parser.add_argument("--output", type=str, required=True, help="Output folder for extracted regions")
    parser.add_argument("--x", type=int, required=True, help="X coordinate of region start")
    parser.add_argument("--y", type=int, required=True, help="Y coordinate of region start")
    parser.add_argument("--width", type=int, required=True, help="Width of region")
    parser.add_argument("--height", type=int, required=True, help="Height of region")
    parser.add_argument("--factors", type=str, default="1,2,4,8", 
                       help="Comma-separated list of downsampling factors (e.g., '1,2,4,8')")
    parser.add_argument("--convert-to-dng", action="store_true",
                       help="Convert TIFF files to DNG format (requires ExifTool)")
    
    args = parser.parse_args()
    
    # Define the region
    region = (args.x, args.y, args.width, args.height)
    
    # Parse downsampling factors
    downsample_factors = [int(f) for f in args.factors.split(',')]
    
    process_burst_folder(args.input, args.output, region, 
                        downsample_factors=downsample_factors,
                        convert_to_dng=args.convert_to_dng)

if __name__ == "__main__":
    main()
