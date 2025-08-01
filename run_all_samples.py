#!/usr/bin/env python3
"""
Script to run optimization for all samples in the data directory.
This script iterates over all sample folders and runs the optimization for each one.
"""

import os
import subprocess
import argparse
import json
from pathlib import Path
from datetime import datetime
import glob
import time
import sys
import numpy as np
import matplotlib.pyplot as plt

# Get the absolute path to optimize.py
OPTIMIZE_PY_PATH = str(Path(__file__).parent / "optimize.py")

def parse_psnr_results(psnr_file_path):
    """Parse PSNR results from the text file created by optimize.py."""
    try:
        with open(psnr_file_path, 'r') as f:
            content = f.read()
        
        # Extract values using simple parsing
        lines = content.split('\n')
        results = {}
        
        # Parse PSNR section
        in_psnr_section = False
        in_ssim_section = False
        in_lpips_section = False
        
        for line in lines:
            line = line.strip()
            
            if 'PSNR Results:' in line:
                in_psnr_section = True
                in_ssim_section = False
                in_lpips_section = False
            elif 'SSIM Results:' in line:
                in_psnr_section = False
                in_ssim_section = True
                in_lpips_section = False
            elif 'LPIPS Results:' in line:
                in_psnr_section = False
                in_ssim_section = False
                in_lpips_section = True
            elif 'Training Results:' in line:
                in_psnr_section = False
                in_ssim_section = False
                in_lpips_section = False
            
            # Parse PSNR metrics
            if in_psnr_section and 'Model Output:' in line and 'dB' in line:
                results['model_psnr'] = float(line.split(':')[1].strip().replace(' dB', ''))
            elif in_psnr_section and 'Bilinear Interpolation:' in line and 'dB' in line:
                results['bilinear_psnr'] = float(line.split(':')[1].strip().replace(' dB', ''))
            elif in_psnr_section and 'PSNR Improvement:' in line:
                results['psnr_improvement'] = float(line.split(':')[1].strip().replace(' dB', ''))
            
            # Parse SSIM metrics
            elif in_ssim_section and 'Model Output:' in line:
                results['model_ssim'] = float(line.split(':')[1].strip())
            elif in_ssim_section and 'Bilinear Interpolation:' in line:
                results['bilinear_ssim'] = float(line.split(':')[1].strip())
            elif in_ssim_section and 'SSIM Improvement:' in line:
                results['ssim_improvement'] = float(line.split(':')[1].strip())
            
            # Parse LPIPS metrics
            elif in_lpips_section and 'Model Output:' in line:
                results['model_lpips'] = float(line.split(':')[1].strip())
            elif in_lpips_section and 'Bilinear Interpolation:' in line:
                results['bilinear_lpips'] = float(line.split(':')[1].strip())
            elif in_lpips_section and 'LPIPS Improvement:' in line:
                results['lpips_improvement'] = float(line.split(':')[1].strip())
            
            # Parse training metrics (not in any section)
            elif 'Final Test Loss:' in line:
                results['final_loss'] = float(line.split(':')[1].strip())
            elif 'Final Test PSNR:' in line:
                results['final_psnr'] = float(line.split(':')[1].strip().replace(' dB', ''))
            elif 'Final Reconstruction Loss:' in line:
                results['final_recon_loss'] = float(line.split(':')[1].strip())
            elif 'Final Transformation Loss:' in line:
                results['final_trans_loss'] = float(line.split(':')[1].strip())
            elif 'Final Total Loss:' in line:
                results['final_total_loss'] = float(line.split(':')[1].strip())
        
        return results
    except Exception as e:
        print(f"Warning: Could not parse PSNR results from {psnr_file_path}: {e}")
        return None

def collect_aggregate_results(output_base_dir):
    """Collect and aggregate results from all processed samples."""
    all_results = []
    failed_samples = []
    
    # Find all sample directories
    for sample_dir in output_base_dir.iterdir():
        if not sample_dir.is_dir():
            continue
            
        sample_name = sample_dir.name
        if not sample_name.endswith('_rgb'):
            continue
            
        # Find scale directories
        for scale_dir in sample_dir.iterdir():
            if not scale_dir.is_dir():
                continue
                
            scale_name = scale_dir.name
            if not scale_name.startswith('scale_'):
                continue
                
            # Check if this sample was successfully processed
            psnr_file = scale_dir / "psnr_results.txt"
            run_info_file = scale_dir / "run_info.json"
            
            if psnr_file.exists() and run_info_file.exists():
                # Parse the results
                results = parse_psnr_results(psnr_file)
                if results:
                    # Add metadata
                    results['sample_name'] = sample_name
                    results['scale_name'] = scale_name
                    results['output_dir'] = str(scale_dir)
                    all_results.append(results)
                else:
                    failed_samples.append(f"{sample_name}/{scale_name}")
            else:
                failed_samples.append(f"{sample_name}/{scale_name}")
    
    # Calculate aggregate statistics
    if all_results:
        psnr_values = [r['model_psnr'] for r in all_results]
        bilinear_psnr_values = [r['bilinear_psnr'] for r in all_results]
        psnr_improvements = [r['psnr_improvement'] for r in all_results]
        final_losses = [r['final_loss'] for r in all_results]
        final_psnrs = [r['final_psnr'] for r in all_results]
        
        # Extract separate losses if available
        final_recon_losses = [r.get('final_recon_loss', 0) for r in all_results]
        final_trans_losses = [r.get('final_trans_loss', 0) for r in all_results]
        final_total_losses = [r.get('final_total_loss', 0) for r in all_results]
        
        # Extract SSIM and LPIPS if available
        model_ssim_values = [r.get('model_ssim', 0) for r in all_results]
        bilinear_ssim_values = [r.get('bilinear_ssim', 0) for r in all_results]
        ssim_improvements = [r.get('ssim_improvement', 0) for r in all_results]
        
        model_lpips_values = [r.get('model_lpips', 0) for r in all_results]
        bilinear_lpips_values = [r.get('bilinear_lpips', 0) for r in all_results]
        lpips_improvements = [r.get('lpips_improvement', 0) for r in all_results]
        
        aggregate_stats = {
            'total_samples': len(all_results),
            'successful_samples': len(all_results),
            'failed_samples': len(failed_samples),
            'failed_sample_list': failed_samples,
            
            # PSNR Statistics
            'model_psnr': {
                'mean': np.mean(psnr_values),
                'std': np.std(psnr_values),
                'min': np.min(psnr_values),
                'max': np.max(psnr_values),
                'median': np.median(psnr_values)
            },
            'bilinear_psnr': {
                'mean': np.mean(bilinear_psnr_values),
                'std': np.std(bilinear_psnr_values),
                'min': np.min(bilinear_psnr_values),
                'max': np.max(bilinear_psnr_values),
                'median': np.median(bilinear_psnr_values)
            },
            'psnr_improvement': {
                'mean': np.mean(psnr_improvements),
                'std': np.std(psnr_improvements),
                'min': np.min(psnr_improvements),
                'max': np.max(psnr_improvements),
                'median': np.median(psnr_improvements)
            },
            'final_loss': {
                'mean': np.mean(final_losses),
                'std': np.std(final_losses),
                'min': np.min(final_losses),
                'max': np.max(final_losses),
                'median': np.median(final_losses)
            },
            'final_psnr': {
                'mean': np.mean(final_psnrs),
                'std': np.std(final_psnrs),
                'min': np.min(final_psnrs),
                'max': np.max(final_psnrs),
                'median': np.median(final_psnrs)
            },
            'final_recon_loss': {
                'mean': np.mean(final_recon_losses),
                'std': np.std(final_recon_losses),
                'min': np.min(final_recon_losses),
                'max': np.max(final_recon_losses),
                'median': np.median(final_recon_losses)
            },
            'final_trans_loss': {
                'mean': np.mean(final_trans_losses),
                'std': np.std(final_trans_losses),
                'min': np.min(final_trans_losses),
                'max': np.max(final_trans_losses),
                'median': np.median(final_trans_losses)
            },
            'final_total_loss': {
                'mean': np.mean(final_total_losses),
                'std': np.std(final_total_losses),
                'min': np.min(final_total_losses),
                'max': np.max(final_total_losses),
                'median': np.median(final_total_losses)
            },
            
            # SSIM Statistics
            'model_ssim': {
                'mean': np.mean(model_ssim_values),
                'std': np.std(model_ssim_values),
                'min': np.min(model_ssim_values),
                'max': np.max(model_ssim_values),
                'median': np.median(model_ssim_values)
            },
            'bilinear_ssim': {
                'mean': np.mean(bilinear_ssim_values),
                'std': np.std(bilinear_ssim_values),
                'min': np.min(bilinear_ssim_values),
                'max': np.max(bilinear_ssim_values),
                'median': np.median(bilinear_ssim_values)
            },
            'ssim_improvement': {
                'mean': np.mean(ssim_improvements),
                'std': np.std(ssim_improvements),
                'min': np.min(ssim_improvements),
                'max': np.max(ssim_improvements),
                'median': np.median(ssim_improvements)
            },
            
            # LPIPS Statistics
            'model_lpips': {
                'mean': np.mean(model_lpips_values),
                'std': np.std(model_lpips_values),
                'min': np.min(model_lpips_values),
                'max': np.max(model_lpips_values),
                'median': np.median(model_lpips_values)
            },
            'bilinear_lpips': {
                'mean': np.mean(bilinear_lpips_values),
                'std': np.std(bilinear_lpips_values),
                'min': np.min(bilinear_lpips_values),
                'max': np.max(bilinear_lpips_values),
                'median': np.median(bilinear_lpips_values)
            },
            'lpips_improvement': {
                'mean': np.mean(lpips_improvements),
                'std': np.std(lpips_improvements),
                'min': np.min(lpips_improvements),
                'max': np.max(lpips_improvements),
                'median': np.median(lpips_improvements)
            },
            
            # Individual results
            'individual_results': all_results
        }
    else:
        aggregate_stats = {
            'total_samples': 0,
            'successful_samples': 0,
            'failed_samples': len(failed_samples),
            'failed_sample_list': failed_samples,
            'individual_results': []
        }
    
    return aggregate_stats

def create_aggregate_plots(aggregate_stats, output_base_dir):
    """Create plots showing aggregate statistics across all samples."""
    if aggregate_stats['successful_samples'] == 0:
        print("No successful samples to plot")
        return
    
    individual_results = aggregate_stats['individual_results']
    
    # Create a comprehensive plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. PSNR Distribution
    psnr_values = [r['model_psnr'] for r in individual_results]
    axes[0, 0].hist(psnr_values, bins=20, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].axvline(aggregate_stats['model_psnr']['mean'], color='red', linestyle='--', 
                       label=f"Mean: {aggregate_stats['model_psnr']['mean']:.2f} dB")
    axes[0, 0].set_title('Model PSNR Distribution', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('PSNR (dB)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. PSNR Improvement Distribution
    improvement_values = [r['psnr_improvement'] for r in individual_results]
    axes[0, 1].hist(improvement_values, bins=20, alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].axvline(aggregate_stats['psnr_improvement']['mean'], color='red', linestyle='--',
                       label=f"Mean: {aggregate_stats['psnr_improvement']['mean']:.2f} dB")
    axes[0, 1].set_title('PSNR Improvement Distribution', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('PSNR Improvement (dB)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Model vs Bilinear PSNR Scatter
    model_psnr = [r['model_psnr'] for r in individual_results]
    bilinear_psnr = [r['bilinear_psnr'] for r in individual_results]
    axes[0, 2].scatter(bilinear_psnr, model_psnr, alpha=0.6, color='purple')
    axes[0, 2].plot([min(bilinear_psnr), max(bilinear_psnr)], [min(bilinear_psnr), max(bilinear_psnr)], 
                    'r--', label='y=x (no improvement)')
    axes[0, 2].set_title('Model PSNR vs Bilinear PSNR', fontsize=14, fontweight='bold')
    axes[0, 2].set_xlabel('Bilinear PSNR (dB)')
    axes[0, 2].set_ylabel('Model PSNR (dB)')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Sample Performance Ranking
    sample_names = [r['sample_name'] for r in individual_results]
    psnr_values = [r['model_psnr'] for r in individual_results]
    
    # Sort by PSNR
    sorted_indices = np.argsort(psnr_values)[::-1]  # Descending order
    sorted_names = [sample_names[i] for i in sorted_indices]
    sorted_psnr = [psnr_values[i] for i in sorted_indices]
    
    axes[1, 0].bar(range(len(sorted_psnr)), sorted_psnr, color='skyblue', alpha=0.7)
    axes[1, 0].set_title('Sample Performance Ranking', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Sample Rank')
    axes[1, 0].set_ylabel('Model PSNR (dB)')
    axes[1, 0].set_xticks(range(len(sorted_names)))
    axes[1, 0].set_xticklabels([name.replace('_rgb', '') for name in sorted_names], rotation=45, ha='right')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Loss Distribution (all three types)
    recon_loss_values = [r.get('final_recon_loss', 0) for r in individual_results]
    trans_loss_values = [r.get('final_trans_loss', 0) for r in individual_results]
    total_loss_values = [r.get('final_total_loss', 0) for r in individual_results]
    
    axes[1, 1].hist(recon_loss_values, bins=15, alpha=0.6, color='red', edgecolor='black', label='Reconstruction')
    axes[1, 1].hist(trans_loss_values, bins=15, alpha=0.6, color='green', edgecolor='black', label='Transformation')
    axes[1, 1].hist(total_loss_values, bins=15, alpha=0.6, color='purple', edgecolor='black', label='Total')
    axes[1, 1].set_title('Final Loss Distribution', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Loss Value')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Summary Statistics Table
    axes[1, 2].axis('off')
    summary_text = f"""Aggregate Statistics
==================
Total Samples: {aggregate_stats['total_samples']}
Successful: {aggregate_stats['successful_samples']}
Failed: {aggregate_stats['failed_samples']}

Model PSNR:
  Mean: {aggregate_stats['model_psnr']['mean']:.2f} ± {aggregate_stats['model_psnr']['std']:.2f} dB
  Range: {aggregate_stats['model_psnr']['min']:.2f} - {aggregate_stats['model_psnr']['max']:.2f} dB

PSNR Improvement:
  Mean: {aggregate_stats['psnr_improvement']['mean']:.2f} ± {aggregate_stats['psnr_improvement']['std']:.2f} dB
  Range: {aggregate_stats['psnr_improvement']['min']:.2f} - {aggregate_stats['psnr_improvement']['max']:.2f} dB

Model SSIM:
  Mean: {aggregate_stats['model_ssim']['mean']:.4f} ± {aggregate_stats['model_ssim']['std']:.4f}
  Range: {aggregate_stats['model_ssim']['min']:.4f} - {aggregate_stats['model_ssim']['max']:.4f}

SSIM Improvement:
  Mean: {aggregate_stats['ssim_improvement']['mean']:.4f} ± {aggregate_stats['ssim_improvement']['std']:.4f}
  Range: {aggregate_stats['ssim_improvement']['min']:.4f} - {aggregate_stats['ssim_improvement']['max']:.4f}

Model LPIPS:
  Mean: {aggregate_stats['model_lpips']['mean']:.4f} ± {aggregate_stats['model_lpips']['std']:.4f}
  Range: {aggregate_stats['model_lpips']['min']:.4f} - {aggregate_stats['model_lpips']['max']:.4f}

LPIPS Improvement:
  Mean: {aggregate_stats['lpips_improvement']['mean']:.4f} ± {aggregate_stats['lpips_improvement']['std']:.4f}
  Range: {aggregate_stats['lpips_improvement']['min']:.4f} - {aggregate_stats['lpips_improvement']['max']:.4f}

Final Losses:
  Recon: {aggregate_stats['final_recon_loss']['mean']:.6f} ± {aggregate_stats['final_recon_loss']['std']:.6f}
  Trans: {aggregate_stats['final_trans_loss']['mean']:.6f} ± {aggregate_stats['final_trans_loss']['std']:.6f}
  Total: {aggregate_stats['final_total_loss']['mean']:.6f} ± {aggregate_stats['final_total_loss']['std']:.6f}

Best Sample: {sorted_names[0] if sorted_names else 'N/A'}
Worst Sample: {sorted_names[-1] if sorted_names else 'N/A'}"""
    
    axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes, 
                    fontsize=12, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plot_path = output_base_dir / "aggregate_results_analysis.png"
    plt.savefig(plot_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close()
    
    print(f"Aggregate analysis plots saved to {plot_path}")
    return plot_path

def get_all_sample_folders(data_dir="data"):
    """Get all sample folders from the data directory."""
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"Error: Data directory {data_dir} does not exist!")
        return []
    
    # Get all sample folders (directories that end with _rgb)
    sample_folders = []
    for item in data_path.iterdir():
        if item.is_dir() and item.name.endswith('_rgb'):
            sample_folders.append(item)
    
    return sorted(sample_folders)

def get_scale_folders(sample_folder, scale_factor=None, lr_shift=None, aug=None):
    """Get scale folders for a given sample that match the specified parameters."""
    scale_folders = []
    
    for item in sample_folder.iterdir():
        if not item.is_dir():
            continue
            
        # Parse folder name: scale_X_shift_Y.Zpx_aug_Z
        folder_name = item.name
        if not folder_name.startswith('scale_'):
            continue
            
        try:
            # Extract parameters from folder name
            parts = folder_name.split('_')
            if len(parts) < 4:
                continue
                
            folder_scale = int(parts[1])
            folder_shift = float(parts[3].replace('px', ''))
            folder_aug = parts[5] if len(parts) > 5 else 'none'
            
            # Check if this folder matches our criteria
            if scale_factor is not None and folder_scale != scale_factor:
                continue
            if lr_shift is not None and abs(folder_shift - lr_shift) > 0.01:
                continue
            if aug is not None and folder_aug != aug:
                continue
                
            scale_folders.append(item)
            
        except (ValueError, IndexError):
            continue
    
    return sorted(scale_folders)

def run_optimization_for_sample(sample_folder, scale_folder, output_base_dir, args):
    """Run optimization for a specific sample and scale configuration."""
    
    # Create output directory for this sample
    sample_name = sample_folder.name
    scale_name = scale_folder.name
    output_dir = output_base_dir / sample_name / scale_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if this sample has already been processed
    # The reverted optimize.py saves results to the current directory
    if (output_dir / "super_resolution_comparison.png").exists():
        print(f"  ⏭️  Skipping {sample_name}/{scale_name} - already processed")
        return True
    
    print(f"  🚀 Processing {sample_name}/{scale_name}")
    
    # Get absolute paths
    project_root = Path(__file__).parent
    data_dir_absolute = str(project_root / "data" / sample_name / scale_name)
    
    # Build the command for the reverted optimize.py (uses --sample_id)
    cmd = [
        sys.executable, OPTIMIZE_PY_PATH,
        "--dataset", "satburst_synth",
        "--sample_id", sample_name,
        "--df", str(args.df),
        "--scale_factor", str(args.scale_factor),
        "--lr_shift", str(args.lr_shift),
        "--num_samples", str(args.num_samples),
        "--aug", args.aug,
        "--model", args.model,
        "--network_depth", str(args.network_depth),
        "--network_hidden_dim", str(args.network_hidden_dim),
        "--projection_dim", str(args.projection_dim),
        "--input_projection", args.input_projection,
        "--seed", str(args.seed),
        "--iters", str(args.iters),
        "--learning_rate", str(args.learning_rate),
        "--weight_decay", str(args.weight_decay),
        "--device", str(args.device)
    ]
    
    if args.use_gnll:
        cmd.append("--use_gnll")
    
    # Run the optimization
    try:
        start_time = time.time()
        
        # Change to the output directory before running optimize.py
        # This ensures all output files are saved in the correct location
        original_cwd = os.getcwd()
        os.chdir(output_dir)
        
        # Set environment variable to override the data path
        env = os.environ.copy()
        env['DATA_DIR_ABSOLUTE'] = data_dir_absolute
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout, env=env)
        
        # Change back to original directory
        os.chdir(original_cwd)
        
        end_time = time.time()
        
        # Save the command and output for reference
        run_info = {
            'sample_name': sample_name,
            'scale_name': scale_name,
            'command': ' '.join(cmd),
            'data_dir_absolute': data_dir_absolute,
            'return_code': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'start_time': datetime.fromtimestamp(start_time).isoformat(),
            'end_time': datetime.fromtimestamp(end_time).isoformat(),
            'duration_seconds': end_time - start_time
        }
        
        with open(output_dir / "run_info.json", 'w') as f:
            json.dump(run_info, f, indent=2)
        
        if result.returncode == 0:
            print(f"  ✅ Completed {sample_name}/{scale_name} in {end_time - start_time:.1f}s")
            return True
        else:
            print(f"  ❌ Failed {sample_name}/{scale_name} (return code: {result.returncode})")
            print(f"     Error: {result.stderr.strip()}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"  ⏰ Timeout {sample_name}/{scale_name} after {args.timeout}s")
        # Change back to original directory in case of timeout
        os.chdir(original_cwd)
        return False
    except Exception as e:
        print(f"  💥 Exception {sample_name}/{scale_name}: {str(e)}")
        # Change back to original directory in case of exception
        os.chdir(original_cwd)
        return False

def main():
    parser = argparse.ArgumentParser(description="Run optimization for all samples in the data directory")
    
    # Data parameters
    parser.add_argument("--data_dir", type=str, default="data", help="Base data directory")
    parser.add_argument("--df", type=int, default=4, help="Downsampling factor")
    parser.add_argument("--scale_factor", type=float, default=4, help="Scale factor for input training grid")
    parser.add_argument("--lr_shift", type=float, default=1.0, help="LR shift value")
    parser.add_argument("--aug", type=str, default="none", choices=['none', 'light', 'medium', 'heavy'], help="Augmentation level")
    parser.add_argument("--num_samples", type=int, default=16, help="Number of samples to use")
    
    # Model parameters
    parser.add_argument("--model", type=str, default="mlp", 
                       choices=["mlp", "siren", "wire", "linear", "conv", "thera", "nir"])
    parser.add_argument("--network_depth", type=int, default=4)
    parser.add_argument("--network_hidden_dim", type=int, default=256)
    parser.add_argument("--projection_dim", type=int, default=256)
    parser.add_argument("--input_projection", type=str, default="fourier_10", 
                       choices=["linear", "fourier_10", "fourier_5", "fourier_20", "fourier_40", "fourier", "legendre", "none"])
    parser.add_argument("--use_gnll", action="store_true")
    
    # Training parameters
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--iters", type=int, default=2000)
    parser.add_argument("--learning_rate", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--device", type=str, default="5", help="CUDA device number")
    
    # Execution parameters
    parser.add_argument("--output_dir", type=str, default="all_samples_results", help="Base output directory")
    parser.add_argument("--timeout", type=int, default=3600, help="Timeout per sample in seconds (1 hour default)")
    parser.add_argument("--sample_filter", type=str, help="Only process samples containing this string")
    parser.add_argument("--max_samples", type=int, help="Maximum number of samples to process")
    parser.add_argument("--resume", action="store_true", help="Resume from where we left off (skip completed samples)")
    
    args = parser.parse_args()
    
    # Parse input projection
    if args.input_projection.startswith("fourier_"):
        args.fourier_scale = float(args.input_projection.split("_")[1])
        args.input_projection = "fourier"
    
    print(f"🔍 Scanning data directory: {args.data_dir}")
    
    # Get all sample folders
    sample_folders = get_all_sample_folders(args.data_dir)
    if not sample_folders:
        print("No sample folders found!")
        return
    
    print(f"📁 Found {len(sample_folders)} sample folders")
    
    # Filter samples if requested
    if args.sample_filter:
        sample_folders = [f for f in sample_folders if args.sample_filter in f.name]
        print(f"🔍 Filtered to {len(sample_folders)} samples containing '{args.sample_filter}'")
    
    # Limit number of samples if requested
    if args.max_samples:
        sample_folders = sample_folders[:args.max_samples]
        print(f"📊 Limited to {len(sample_folders)} samples")
    
    # Create output directory
    output_base_dir = Path(args.output_dir)
    output_base_dir.mkdir(exist_ok=True)
    
    # Save run configuration
    config = {
        'timestamp': datetime.now().isoformat(),
        'args': vars(args),
        'sample_folders': [str(f) for f in sample_folders]
    }
    
    with open(output_base_dir / "run_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"📊 Processing {len(sample_folders)} samples")
    print(f"📁 Output directory: {output_base_dir}")
    print(f"⚙️  Configuration: df={args.df}, lr_shift={args.lr_shift}, aug={args.aug}")
    print(f"🤖 Model: {args.model}, depth={args.network_depth}, hidden={args.network_hidden_dim}")
    print(f"⏱️  Timeout per sample: {args.timeout}s")
    print(f"{'='*60}")
    
    # Track results
    total_samples = 0
    successful_samples = 0
    failed_samples = 0
    skipped_samples = 0
    
    start_time = time.time()
    
    for i, sample_folder in enumerate(sample_folders, 1):
        print(f"\n[{i}/{len(sample_folders)}] Processing sample: {sample_folder.name}")
        
        # Get scale folders for this sample
        scale_folders = get_scale_folders(sample_folder, args.df, args.lr_shift, args.aug)
        
        if not scale_folders:
            print(f"  ⚠️  No matching scale folders found for {sample_folder.name}")
            print(f"     Expected: scale_{args.df}_shift_{args.lr_shift:.1f}px_aug_{args.aug}")
            continue
        
        print(f"  📂 Found {len(scale_folders)} matching scale folder(s)")
        
        for scale_folder in scale_folders:
            total_samples += 1
            
            # Check if already processed
            sample_name = sample_folder.name
            scale_name = scale_folder.name
            output_dir = output_base_dir / sample_name / scale_name
            
            if args.resume and (output_dir / "super_resolution_comparison.png").exists():
                print(f"  ⏭️  Skipping {sample_name}/{scale_name} - already processed")
                skipped_samples += 1
                continue
            
            # Run optimization
            success = run_optimization_for_sample(sample_folder, scale_folder, output_base_dir, args)
            
            if success:
                successful_samples += 1
            else:
                failed_samples += 1
    
    end_time = time.time()
    total_duration = end_time - start_time
    
    # Collect and analyze aggregate results
    print(f"\n📊 Collecting aggregate results from all samples...")
    aggregate_stats = collect_aggregate_results(output_base_dir)
    
    # Create aggregate plots
    if aggregate_stats['successful_samples'] > 0:
        print(f"📈 Creating aggregate analysis plots...")
        create_aggregate_plots(aggregate_stats, output_base_dir)
    
    # Save comprehensive final summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'execution_summary': {
            'total_samples': total_samples,
            'successful_samples': successful_samples,
            'failed_samples': failed_samples,
            'skipped_samples': skipped_samples,
            'total_duration_seconds': total_duration,
            'average_duration_per_sample': total_duration / total_samples if total_samples > 0 else 0,
            'success_rate': successful_samples / total_samples if total_samples > 0 else 0
        },
        'aggregate_results': aggregate_stats
    }
    
    with open(output_base_dir / "final_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print final summary
    print(f"\n{'='*60}")
    print("🎉 BATCH PROCESSING COMPLETE!")
    print(f"{'='*60}")
    print(f"📊 Execution Summary:")
    print(f"  ✅ Successful: {successful_samples}")
    print(f"  ❌ Failed: {failed_samples}")
    print(f"  ⏭️  Skipped: {skipped_samples}")
    print(f"  📈 Total: {total_samples}")
    print(f"  📊 Success Rate: {summary['execution_summary']['success_rate']:.1%}")
    print(f"  ⏱️  Total Duration: {total_duration/3600:.1f} hours")
    print(f"  ⏱️  Average per Sample: {summary['execution_summary']['average_duration_per_sample']/60:.1f} minutes")
    
    if aggregate_stats['successful_samples'] > 0:
        print(f"\n📈 Aggregate Performance Results:")
        print(f"  📊 Model PSNR: {aggregate_stats['model_psnr']['mean']:.2f} ± {aggregate_stats['model_psnr']['std']:.2f} dB")
        print(f"  🎯 PSNR Improvement: {aggregate_stats['psnr_improvement']['mean']:.2f} ± {aggregate_stats['psnr_improvement']['std']:.2f} dB")
        print(f"  📉 Best Sample: {aggregate_stats['individual_results'][0]['sample_name'] if aggregate_stats['individual_results'] else 'N/A'}")
        print(f"  📉 Worst Sample: {aggregate_stats['individual_results'][-1]['sample_name'] if aggregate_stats['individual_results'] else 'N/A'}")
        print(f"  📊 PSNR Range: {aggregate_stats['model_psnr']['min']:.2f} - {aggregate_stats['model_psnr']['max']:.2f} dB")
    
    print(f"  📁 Results saved to: {output_base_dir}")
    print(f"  📄 Summary: {output_base_dir / 'final_summary.json'}")
    print(f"  📊 Aggregate Analysis: {output_base_dir / 'aggregate_results_analysis.png'}")

if __name__ == "__main__":
    main() 