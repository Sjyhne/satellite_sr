import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import random
import wandb  # Add at the top with other imports
import pandas as pd  # Add to imports
from torchmetrics.functional import structural_similarity_index_measure as ssim
from torch.utils.data import DataLoader
from data import get_dataset
from utils import bilinear_resize_torch, align_kornia_brute_force
from losses import BasicLosses
from viz_utils import (
    plot_training_curves, visualize_translations,
    visualize_shift_evolution, visualize_shift_evolution_animation_frames
)
from evals import PSNR, SSIM, LPIPS as LPIPS_Eval  # Import evaluation metrics from evals.py
from evals import get_gaussian_kernel, match_colors

from models.utils import get_decoder
from input_projections.utils import get_input_projection
from models.inr import INR
from torchmetrics.functional import structural_similarity_index_measure


import einops
import json
import argparse
import pathlib
import lpips
import seaborn as sns
import copy
import traceback

def crop_borders(img, margin=16):
    """
    Crop border pixels from an image to avoid edge artifacts in evaluation.
    
    Args:
        img: Input image tensor [C, H, W] or [B, C, H, W]
        margin: Number of pixels to crop from each edge
        
    Returns:
        Cropped image tensor with same batch and channel dimensions
    """
    if len(img.shape) == 3:  # [C, H, W]
        return img[:, margin:-margin, margin:-margin]
    elif len(img.shape) == 4:  # [B, C, H, W]
        return img[:, :, margin:-margin, margin:-margin]
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")

def calculate_metrics(pred, target, loss_fn_alex=None, crop_margin=16):
    """
    Calculate all evaluation metrics for the given prediction and target.
    
    Args:
        pred: Predicted image tensor [B, C, H, W]
        target: Target image tensor [B, C, H, W]
        loss_fn_alex: Optional pre-initialized LPIPS model
        crop_margin: Number of pixels to crop from border before calculating metrics (0 for no cropping)
        
    Returns:
        Dictionary of metrics
    """
    # Ensure inputs are in correct format
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
    if target.dim() == 3:
        target = target.unsqueeze(0)
        
    # Ensure tensors are on the right device
    device = pred.device
    
    # Crop borders to avoid edge artifacts in evaluation
    if crop_margin > 0:
        pred = crop_borders(pred, crop_margin)
        target = crop_borders(target, crop_margin)

    psnr_metric = PSNR(boundary_ignore=crop_margin, max_value=1.0)
    
    mse = F.mse_loss(pred, target)
    psnr_value = psnr_metric(pred, target)
    ssim_value = structural_similarity_index_measure(pred, target, data_range=1.0)
    
    # For LPIPS, handle RGGB format (4 channels) by converting to RGB (3 channels)
    # LPIPS requires RGB inputs
    pred_lpips = pred
    target_lpips = target
    
    # Convert RGGB to RGB if necessary
    if pred.shape[1] == 4:
        # Extract R, G1, G2, B channels
        R_pred = pred[:, 0:1]
        G1_pred = pred[:, 1:2]
        G2_pred = pred[:, 2:3]
        B_pred = pred[:, 3:4]
        
        # Average G1 and G2 to create RGB
        G_pred = (G1_pred + G2_pred) / 2
        pred_lpips = torch.cat([R_pred, G_pred, B_pred], dim=1)
    
    if target.shape[1] == 4:
        # Extract R, G1, G2, B channels
        R_target = target[:, 0:1]
        G1_target = target[:, 1:2]
        G2_target = target[:, 2:3]
        B_target = target[:, 3:4]
        
        # Average G1 and G2 to create RGB
        G_target = (G1_target + G2_target) / 2
        target_lpips = torch.cat([R_target, G_target, B_target], dim=1)
    
    # For LPIPS, either use the provided model or create a new one from evals
    if loss_fn_alex is not None and isinstance(loss_fn_alex, lpips.LPIPS):
        # Use the existing model but wrap with our interface
        lpips_value = loss_fn_alex(pred_lpips, target_lpips).mean()
    else:
        # Create a new LPIPS from evals
        lpips_metric = LPIPS_Eval(type='alex').to(device)
        lpips_value = lpips_metric(pred_lpips, target_lpips)
    
    return {
        'mse': mse.item(),
        'psnr': psnr_value.item(),
        'lpips': lpips_value.item(),
        'ssim': ssim_value.item()
    }

def decoder_trainable(sample_id: torch.Tensor, keep_out_indices: torch.Tensor) -> bool:
    """
    Decide whether the current sample should update the decoder.
    Replace this with your own rule or a lookup into a global mask.
    sample_id is a scalar tensor living on the same device as the input.
    """
    # If the sample_id is in the keep_out_indices, return False, otherwise return True
    return int(sample_id.item()) not in keep_out_indices


def train_one_iteration(model, optimizer, shift_optimizer, train_sample, device, downsample_factor,
                        iteration=0, use_gt=False):
    """
    Forward / backward **per sample**.
    * shift module is always trained
    * decoder is trained only when decoder_trainable(sample_id) is True
    """

    model.train()

    ####── loss functions ────────────────────────────────────────────────####
    recon_criterion = BasicLosses.mse_loss
    monitor_criterion = BasicLosses.mse_loss  # Always use MSE for monitoring
    trans_criterion = BasicLosses.mae_loss
    if model.use_gnll:
        recon_criterion = nn.GaussianNLLLoss()          # reduction='mean' inside

    ####── move batch tensors to device ─────────────────────────────────####
    x          = train_sample['input'      ].to(device)      # [B, …]
    y          = train_sample['lr_target'  ].to(device)      # [B, …]
    sample_ids = train_sample['sample_id'  ].to(device)      # [B]
    lr_mean    = train_sample['mean'       ].to(device)
    lr_std     = train_sample['std'        ].to(device)

    if 'shifts' in train_sample and 'dx_percent' in train_sample['shifts']:
        gt_dx = train_sample['shifts']['dx_percent'].to(device)
        gt_dy = train_sample['shifts']['dy_percent'].to(device)
    else:
        gt_dx = torch.zeros_like(sample_ids, dtype=x.dtype, device=device)
        gt_dy = torch.zeros_like(sample_ids, dtype=x.dtype, device=device)

    ####── containers to log epoch stats ────────────────────────────────####
    recon_loss_total = 0.0
    monitor_recon_loss_total = 0.0  # For monitoring with MSE
    trans_loss_total = 0.0
    keep_count = 0

    # Keep-out statistics
    recon_loss_keepout_total = 0.0
    monitor_recon_loss_keepout_total = 0.0  # For monitoring with MSE
    trans_loss_keepout_total = 0.0
    keepout_count = 0

    B = x.shape[0]            # current batch size (might be 1)
    for i in range(B):
        # ── slice tensors for this single sample ────────────────────────
        xi         = x[i:i+1]                      # keep dims ⇒ batch-1 tensor
        yi         = y[i:i+1]
        sid        = sample_ids[i:i+1]
        gtdx_i     = gt_dx[i:i+1]
        gtdy_i     = gt_dy[i:i+1]

        if yi.shape[1] == 3:
            yi = yi.permute(0, 2, 3, 1)
        optim_decode = decoder_trainable(sid, model.keep_out_indices)

        # ── zero grads (per-sample) ─────────────────────────────────────
        optimizer.zero_grad()
        if shift_optimizer is not None:
            shift_optimizer.zero_grad()

        # ── forward ─────────────────────────────────────────────────────
        if model.use_gnll:
            out, pred_shifts, pred_var = model(xi, sid, scale_factor=1/downsample_factor,
                                               lr_frames=yi)
            recon_loss = recon_criterion(out, yi, pred_var)
            
            # Calculate MSE loss for monitoring (not used in optimization)
            with torch.no_grad():
                monitor_recon_loss = monitor_criterion(out, yi)
        else:
            out, pred_shifts = model(xi, sid, scale_factor=1/downsample_factor,
                                      lr_frames=yi)

            recon_loss = recon_criterion(out, yi)
            monitor_recon_loss = recon_loss  # Same when not using GNLL

        pred_dx, pred_dy = pred_shifts
        trans_loss = (trans_criterion(pred_dx, gtdx_i)
                      + trans_criterion(pred_dy, gtdy_i))

        # ── backward ────────────────────────────────────────────────────
        recon_loss.backward()

        # ── optionally wipe decoder grads ───────────────────────────────
        if not optim_decode:
            for p in model.decoder.parameters():
                p.grad = None                      # skip decoder update
        
        # ── step optimizers ─────────────────────────────────────────────
        optimizer.step()
        if shift_optimizer is not None:
            shift_optimizer.step()

        # ── accumulate logs ─────────────────────────────────────────────
        if sample_ids != 0 and optim_decode:
            recon_loss_total += recon_loss.item()
            monitor_recon_loss_total += monitor_recon_loss.item()
            trans_loss_total += trans_loss.item()
            keep_count += 1
        # ── keep-out logs ───────────────────────────────────────────────
        if not optim_decode:
            recon_loss_keepout_total += recon_loss.item()
            monitor_recon_loss_keepout_total += monitor_recon_loss.item()
            trans_loss_keepout_total += trans_loss.item()
            keepout_count += 1

    # average over the mini-batch
    if keep_count > 0:
        recon_loss_avg = recon_loss_total / keep_count
        monitor_recon_loss_avg = monitor_recon_loss_total / keep_count
        trans_loss_avg = trans_loss_total / keep_count
        total_loss_avg = recon_loss_avg + trans_loss_avg
    else:
        recon_loss_avg = 0.0
        monitor_recon_loss_avg = 0.0
        trans_loss_avg = 0.0
        total_loss_avg = 0.0

    # compute keep-out averages if any keep-out samples were present
    if keepout_count > 0:
        recon_loss_keepout_avg = recon_loss_keepout_total / keepout_count
        monitor_recon_loss_keepout_avg = monitor_recon_loss_keepout_total / keepout_count
        trans_loss_keepout_avg = trans_loss_keepout_total / keepout_count
        total_loss_keepout_avg = recon_loss_keepout_avg + trans_loss_keepout_avg
    else:
        recon_loss_keepout_avg = 0.0
        monitor_recon_loss_keepout_avg = 0.0
        trans_loss_keepout_avg = 0.0
        total_loss_keepout_avg = 0.0

    return {
        "recon_loss":  recon_loss_avg,
        "monitor_recon_loss": monitor_recon_loss_avg,  # MSE loss for monitoring
        "trans_loss":  trans_loss_avg,
        "total_loss":  total_loss_avg,
        "recon_loss_keepout":  recon_loss_keepout_avg,
        "monitor_recon_loss_keepout": monitor_recon_loss_keepout_avg,  # MSE loss for keep-out monitoring
        "trans_loss_keepout":  trans_loss_keepout_avg,
        "total_loss_keepout":  total_loss_keepout_avg,
        "pred_dx":     pred_dx.detach().cpu().numpy(),   # last sample's preds
        "pred_dy":     pred_dy.detach().cpu().numpy(),
        "gt_dx":       gt_dx.detach().cpu().numpy(),
        "gt_dy":       gt_dy.detach().cpu().numpy(),
    }

def test_one_epoch(model, test_loader, device):
    model.eval()

    # Get HR features from the test loader
    hr_coords = test_loader.get_hr_coordinates().unsqueeze(0).to(device)
    hr_image = test_loader.get_original_hr()
    
    if hr_image is not None:
        hr_image = hr_image.unsqueeze(0).to(device)

    sample_id = torch.tensor([0]).to(device)

    with torch.no_grad():
        if model.use_gnll:
            output, _, _ = model(hr_coords, sample_id, scale_factor=1, training=False, lr_frames=None)
        else:
            output, _ = model(hr_coords, sample_id, scale_factor=1, training=False)

    output = output * test_loader.get_lr_std(0).cuda() + test_loader.get_lr_mean(0).cuda()

    if hr_image is None:
        hr_image = torch.zeros_like(output)
    
    if hr_image.shape[1] == 3:
        hr_image = hr_image.permute(0, 2, 3, 1)

    loss = F.mse_loss(output, hr_image)
    
    return loss.item(), output.detach(), hr_image.detach()

def validate_dataset(model, train_dataloader, device, downsample_factor):
    """
    Evaluate the model specifically on keep-out samples to measure generalization.
    This function does not affect training and only computes the loss.
    
    Args:
        model: The model to evaluate
        train_dataloader: DataLoader for the training data
        device: Device to run the evaluation on
    
    Returns:
        Dictionary containing average reconstruction loss and translation loss for keep-out samples
    """
    model.eval()
    
    # Use the same loss functions as in training
    recon_criterion = BasicLosses.mae_loss
    monitor_criterion = BasicLosses.mse_loss  # Always use MSE for monitoring
    trans_criterion = BasicLosses.mse_loss
    if model.use_gnll:
        recon_criterion = nn.GaussianNLLLoss()
    
    # Containers for keep-out losses
    recon_loss_keepout_total = 0.0
    monitor_recon_loss_keepout_total = 0.0  # For MSE monitoring
    trans_loss_keepout_total = 0.0
    keepout_count = 0
    
    with torch.no_grad():
        for train_sample in train_dataloader:
            # Get sample data and move to device
            x = train_sample['input'].to(device)
            y = train_sample['lr_target'].to(device)
            sample_ids = train_sample['sample_id'].to(device)
            
            if 'shifts' in train_sample and 'dx_percent' in train_sample['shifts']:
                gt_dx = train_sample['shifts']['dx_percent'].to(device)
                gt_dy = train_sample['shifts']['dy_percent'].to(device)
            else:
                gt_dx = torch.zeros_like(sample_ids, dtype=x.dtype, device=device)
                gt_dy = torch.zeros_like(sample_ids, dtype=x.dtype, device=device)
            
            # Process each sample in the batch
            B = x.shape[0]
            for i in range(B):
                # Slice tensors for this single sample
                xi = x[i:i+1]
                yi = y[i:i+1]
                sid = sample_ids[i:i+1]
                gtdx_i = gt_dx[i:i+1]
                gtdy_i = gt_dy[i:i+1]
                
                # Check if this is a keep-out sample
                if not decoder_trainable(sid, model.keep_out_indices):
                    # Forward pass
                    if model.use_gnll:
                        out, pred_shifts, pred_var = model(xi, sid, scale_factor=1/downsample_factor, lr_frames=yi)
                        recon_loss = recon_criterion(out, yi, pred_var)
                        # Calculate MSE loss for monitoring
                        monitor_recon_loss = monitor_criterion(out, yi)
                    else:
                        out, pred_shifts = model(xi, sid, scale_factor=1/downsample_factor, lr_frames=yi)
                        recon_loss = recon_criterion(out, yi)
                        monitor_recon_loss = monitor_criterion(out, yi)
                    
                    pred_dx, pred_dy = pred_shifts
                    trans_loss = (trans_criterion(pred_dx, gtdx_i) + trans_criterion(pred_dy, gtdy_i))
                    
                    # Accumulate losses
                    recon_loss_keepout_total += recon_loss.item()
                    monitor_recon_loss_keepout_total += monitor_recon_loss.item()
                    trans_loss_keepout_total += trans_loss.item()
                    keepout_count += 1
    
    # Compute averages if any keep-out samples were present
    if keepout_count > 0:
        validation_results = {
            'recon_loss_keepout_val': recon_loss_keepout_total / keepout_count,
            'monitor_recon_loss_keepout_val': monitor_recon_loss_keepout_total / keepout_count,
            'trans_loss_keepout_val': trans_loss_keepout_total / keepout_count,
            'total_loss_keepout_val': (recon_loss_keepout_total + trans_loss_keepout_total) / keepout_count,
            'keepout_count': keepout_count
        }
    else:
        validation_results = {
            'recon_loss_keepout_val': 0.0,
            'monitor_recon_loss_keepout_val': 0.0,
            'trans_loss_keepout_val': 0.0,
            'total_loss_keepout_val': 0.0,
            'keepout_count': 0
        }
    
    return validation_results

def aggregate_results(base_dir):
    """
    Aggregate results across all samples for each model type and configuration.
    Results are separated by dataset for more meaningful comparisons.
    
    Args:
        base_dir: Path to the base results directory
    
    Returns:
        Dictionary of DataFrames with aggregated results per dataset
    """
    # Find all dataset directories
    dataset_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
    
    # Dictionary to hold results for each dataset
    dataset_results = {}
    
    # Process each dataset separately
    for dataset_dir in dataset_dirs:
        dataset_name = dataset_dir.name
        print(f"Processing dataset: {dataset_name}")
        
        results = []
        
        # Find all experiment directories
        experiment_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
        
        for experiment_dir in experiment_dirs:
            # Find all sample directories
            sample_dirs = [d for d in experiment_dir.iterdir() if d.is_dir()]
            
            for sample_dir in sample_dirs:
                # Find all model directories
                model_dirs = [d for d in sample_dir.iterdir() if d.is_dir()]
                
                for model_dir in model_dirs:
                    # Check if metrics.json exists
                    metrics_file = model_dir / 'metrics.json'
                    if metrics_file.exists():
                        with open(metrics_file, 'r') as f:
                            metrics = json.load(f)
                        
                        # Add sample and model info
                        metrics['sample_id'] = sample_dir.name
                        metrics['dataset'] = dataset_name
                        metrics['experiment'] = experiment_dir.name
                        
                        # Extract experiment details from experiment name
                        # Format is typically: df{df}_shift{shift}_samples{num_samples}_aug{aug}
                        experiment_parts = experiment_dir.name.split('_')
                        for part in experiment_parts:
                            if part.startswith('samples'):
                                try:
                                    metrics['num_samples'] = int(part[7:])  # Extract number after 'samples'
                                except ValueError:
                                    pass
                        
                        # Parse model directory name to extract all parameters
                        model_parts = model_dir.name.split('_')
                        
                        # Extract base model type (first part)
                        metrics['model_type'] = model_parts[0]
                        
                        # Extract projection type and parameters
                        if len(model_parts) >= 2:
                            projection_parts = []
                            i = 1
                            # Collect all parts until we hit iterations or learning rate
                            while i < len(model_parts) and not (model_parts[i].isdigit() or model_parts[i].startswith('lr')):
                                projection_parts.append(model_parts[i])
                                i += 1
                            projection_type = '_'.join(projection_parts)
                            
                            # Normalize projection types
                            # For Fourier projections, standardize to a consistent name
                            if projection_type == 'fourier' or projection_type.startswith('fourier_'):
                                # Extract fourier scale from config if possible
                                fourier_scale = None
                                config_file = model_dir / 'experiment_config.json'
                                if config_file.exists():
                                    try:
                                        with open(config_file, 'r') as f:
                                            config = json.load(f)
                                            if 'fourier_scale' in config:
                                                fourier_scale = config['fourier_scale']
                                            if 'num_samples' in config and 'num_samples' not in metrics:
                                                metrics['num_samples'] = config['num_samples']
                                    except Exception:
                                        pass
                                        
                                # Get scale from projection name if present
                                if not fourier_scale and '_' in projection_type:
                                    try:
                                        scale_str = projection_type.split('_')[1]
                                        fourier_scale = float(scale_str)
                                    except (ValueError, IndexError):
                                        pass
                                
                                # Default to 10.0 if no scale found (this is the common default)
                                if not fourier_scale:
                                    fourier_scale = 10.0
                                    
                                # Use a standardized name format
                                projection_type = f"fourier_{fourier_scale}"
                                
                                # Also store the scale separately
                                metrics['fourier_scale'] = fourier_scale
                            
                            metrics['projection_type'] = projection_type
                        else:
                            metrics['projection_type'] = 'unknown'
                        
                        # Extract iterations (numeric value before lr)
                        for i, part in enumerate(model_parts):
                            if part.isdigit():
                                try:
                                    metrics['iterations'] = int(part)
                                except ValueError:
                                    pass
                                break
                        
                        # Extract learning rate
                        for part in model_parts:
                            if part.startswith('lr'):
                                try:
                                    metrics['learning_rate'] = float(part[2:])
                                except ValueError:
                                    metrics['learning_rate'] = None
                                break
                        
                        # Add flags
                        metrics['use_gnll'] = '_gnll' in model_dir.name
                        metrics['use_dual_optimizer'] = '_dual' in model_dir.name
                        
                        # Check for additional configuration in experiment_config.json
                        config_file = model_dir / 'experiment_config.json'
                        if config_file.exists():
                            try:
                                with open(config_file, 'r') as f:
                                    config = json.load(f)
                                    # Add network configuration details
                                    if 'network_depth' in config:
                                        metrics['network_depth'] = config['network_depth']
                                    if 'network_hidden_dim' in config:
                                        metrics['network_hidden_dim'] = config['network_hidden_dim']
                                    if 'projection_dim' in config:
                                        metrics['projection_dim'] = config['projection_dim']
                                    if 'fourier_scale' in config and 'fourier_scale' not in metrics:
                                        metrics['fourier_scale'] = config['fourier_scale']
                                    if 'legendre_max_degree' in config:
                                        metrics['legendre_max_degree'] = config['legendre_max_degree']
                                    if 'rotation' in config:
                                        metrics['rotation'] = config['rotation']
                                    if 'weight_decay' in config:
                                        metrics['weight_decay'] = config['weight_decay']
                                    if 'iterations' in config and 'iterations' not in metrics:
                                        metrics['iterations'] = config['iterations']
                                    if 'num_samples' in config and 'num_samples' not in metrics:
                                        metrics['num_samples'] = config['num_samples']
                            except Exception as e:
                                print(f"Error reading config file {config_file}: {e}")
                        
                        # Create detailed model identifier matching directory structure
                        model_identifier = f"{metrics['model_type']}_{metrics['projection_type']}"
                        if 'iterations' in metrics and metrics['iterations']:
                            model_identifier += f"_{metrics['iterations']}"
                        if 'learning_rate' in metrics and metrics['learning_rate']:
                            model_identifier += f"_lr{metrics['learning_rate']}"
                        if metrics['use_gnll']:
                            model_identifier += "_gnll"
                        if metrics['use_dual_optimizer']:
                            model_identifier += "_dual"
                            
                        # Add network parameters to identifier if available
                        model_details = model_identifier
                        if 'network_depth' in metrics and 'network_hidden_dim' in metrics:
                            model_details += f"_d{metrics['network_depth']}_h{metrics['network_hidden_dim']}"
                        
                        metrics['model_identifier'] = model_identifier
                        metrics['model_details'] = model_details
                        
                        results.append(metrics)
        
        # Convert to DataFrame
        if results:
            df = pd.DataFrame(results)
            dataset_results[dataset_name] = df
            
            # Create dataset-specific output directory
            dataset_output_dir = base_dir / dataset_name / 'aggregated'
            dataset_output_dir.mkdir(exist_ok=True, parents=True)
            
            # Save full results for this dataset
            df.to_csv(dataset_output_dir / 'aggregated_results.csv', index=False)
            
            # Create summary tables for this dataset
            _create_summary_tables(df, dataset_output_dir, dataset_name)
        else:
            print(f"No results found for dataset: {dataset_name}")
    
    # Also create a combined results file (for backward compatibility)
    if dataset_results:
        all_results = pd.concat(dataset_results.values(), ignore_index=True)
        all_results.to_csv(base_dir / 'all_datasets_aggregated_results.csv', index=False)
    
    return dataset_results

def _create_summary_tables(df, output_dir, dataset_name):
    """Helper function to create summary tables for a single dataset"""
    # Define columns for groupby
    groupby_columns = ['model_type', 'projection_type', 'learning_rate', 'use_gnll', 'use_dual_optimizer']
    
    # Add downsampling factor to groupby columns to separate different scales
    if 'downsampling_factor' in df.columns:
        groupby_columns.insert(0, 'downsampling_factor')  # Insert at the beginning for prominence
    
    # Add network parameters to groupby if they exist
    if 'network_depth' in df.columns:
        groupby_columns.append('network_depth')
    if 'network_hidden_dim' in df.columns:
        groupby_columns.append('network_hidden_dim')
    if 'iterations' in df.columns:
        groupby_columns.append('iterations')
    if 'num_samples' in df.columns:
        groupby_columns.append('num_samples')
    
    # Define metrics to aggregate
    agg_dict = {
        'final_psnr': ['mean', 'std', 'min', 'max', 'count'],
        'psnr_improvement': ['mean', 'std'],
        'final_baseline_psnr': ['mean', 'std']  # Add baseline PSNR
    }
    
    # Add LPIPS metrics if available
    if 'final_lpips' in df.columns:
        agg_dict['final_lpips'] = ['mean', 'std']
        agg_dict['final_baseline_lpips'] = ['mean', 'std']  # Add baseline LPIPS
        if 'lpips_improvement' in df.columns:
            agg_dict['lpips_improvement'] = ['mean', 'std']
    
    # Add SSIM metrics if available
    if 'final_ssim' in df.columns:
        agg_dict['final_ssim'] = ['mean', 'std']
        agg_dict['final_baseline_ssim'] = ['mean', 'std']  # Add baseline SSIM
        if 'ssim_improvement' in df.columns:
            agg_dict['ssim_improvement'] = ['mean', 'std']
        
    # Create detailed summary table
    summary_table = df.groupby(groupby_columns).agg(agg_dict).reset_index()

    # Format the table for readability
    formatted_table_columns = {}
    
    # Add scale (downsampling factor) as the first column if available
    if 'downsampling_factor' in summary_table.columns:
        formatted_table_columns['Scale'] = summary_table['downsampling_factor']
    
    # Add other columns
    formatted_table_columns.update({
        'Model Type': summary_table['model_type'],
        'Projection': summary_table['projection_type'],
        'Learning Rate': summary_table['learning_rate'],
        'GNLL': summary_table['use_gnll'],
        'Dual Optimizer': summary_table['use_dual_optimizer'],
    })
    
    # Add optional columns if they exist
    if 'network_depth' in summary_table.columns:
        formatted_table_columns['Network Depth'] = summary_table['network_depth']
    if 'network_hidden_dim' in summary_table.columns:
        formatted_table_columns['Hidden Dim'] = summary_table['network_hidden_dim']
    if 'iterations' in summary_table.columns:
        formatted_table_columns['Iterations'] = summary_table['iterations']
    if 'num_samples' in summary_table.columns:
        formatted_table_columns['LR Samples'] = summary_table['num_samples']
        
    # Add metric columns
    formatted_table_columns.update({
        'Sample Count': summary_table['final_psnr']['count'],
        'PSNR': [f"{m:.2f} ± {s:.2f}" for m, s in zip(
            summary_table['final_psnr']['mean'], 
            summary_table['final_psnr']['std']
        )],
        'Baseline PSNR': [f"{m:.2f} ± {s:.2f}" for m, s in zip(
            summary_table['final_baseline_psnr']['mean'], 
            summary_table['final_baseline_psnr']['std']
        )],
        'PSNR Improvement': [f"{m:.2f} ± {s:.2f}" for m, s in zip(
            summary_table['psnr_improvement']['mean'], 
            summary_table['psnr_improvement']['std']
        )]
    })
    
    # Add LPIPS metrics if available
    if 'final_lpips' in summary_table.columns:
        formatted_table_columns['LPIPS'] = [f"{m:.4f} ± {s:.4f}" for m, s in zip(
            summary_table['final_lpips']['mean'], 
            summary_table['final_lpips']['std']
        )]
        formatted_table_columns['Baseline LPIPS'] = [f"{m:.4f} ± {s:.4f}" for m, s in zip(
            summary_table['final_baseline_lpips']['mean'], 
            summary_table['final_baseline_lpips']['std']
        )]
        if 'lpips_improvement' in summary_table.columns:
            formatted_table_columns['LPIPS Improvement'] = [f"{m:.4f} ± {s:.4f}" for m, s in zip(
                summary_table['lpips_improvement']['mean'], 
                summary_table['lpips_improvement']['std']
            )]
    
    # Add SSIM metrics if available
    if 'final_ssim' in summary_table.columns:
        formatted_table_columns['SSIM'] = [f"{m:.4f} ± {s:.4f}" for m, s in zip(
            summary_table['final_ssim']['mean'], 
            summary_table['final_ssim']['std']
        )]
        formatted_table_columns['Baseline SSIM'] = [f"{m:.4f} ± {s:.4f}" for m, s in zip(
            summary_table['final_baseline_ssim']['mean'], 
            summary_table['final_baseline_ssim']['std']
        )]
        if 'ssim_improvement' in summary_table.columns:
            formatted_table_columns['SSIM Improvement'] = [f"{m:.4f} ± {s:.4f}" for m, s in zip(
                summary_table['ssim_improvement']['mean'], 
                summary_table['ssim_improvement']['std']
            )]
    
    formatted_table = pd.DataFrame(formatted_table_columns)

    # Sort by scale first, then by PSNR improvement
    if 'Scale' in formatted_table.columns:
        formatted_table = formatted_table.sort_values(['Scale', 'PSNR Improvement'], ascending=[True, False])
    else:
        formatted_table = formatted_table.sort_values('PSNR Improvement', ascending=False)
        
    formatted_table.to_csv(output_dir / 'model_comparison_detailed.csv', index=False)
    
    # Create a simplified summary table for backward compatibility
    simple_columns = {}
    
    # Add scale as the first column if available
    if 'downsampling_factor' in summary_table.columns:
        simple_columns['Scale'] = summary_table['downsampling_factor']
        
    simple_columns.update({
        'Model': summary_table['model_type'],
        'Projection': summary_table['projection_type'],
        'Learning Rate': summary_table['learning_rate'],
    })
    
    # Add iterations and num_samples to the simplified table as well
    if 'iterations' in summary_table.columns:
        simple_columns['Iterations'] = summary_table['iterations']
    if 'num_samples' in summary_table.columns:
        simple_columns['LR Samples'] = summary_table['num_samples']
        
    # Add metrics to simplified table
    simple_columns.update({
        'Sample Count': summary_table['final_psnr']['count'],
        'PSNR': [f"{m:.2f} ± {s:.2f}" for m, s in zip(
            summary_table['final_psnr']['mean'], 
            summary_table['final_psnr']['std']
        )],
        'Baseline PSNR': [f"{m:.2f} ± {s:.2f}" for m, s in zip(
            summary_table['final_baseline_psnr']['mean'], 
            summary_table['final_baseline_psnr']['std']
        )],
        'PSNR Improvement': [f"{m:.2f} ± {s:.2f}" for m, s in zip(
            summary_table['psnr_improvement']['mean'], 
            summary_table['psnr_improvement']['std']
        )]
    })
    
    # Add LPIPS metrics to simplified table if available
    if 'final_lpips' in summary_table.columns:
        simple_columns['LPIPS'] = [f"{m:.4f} ± {s:.4f}" for m, s in zip(
            summary_table['final_lpips']['mean'], 
            summary_table['final_lpips']['std']
        )]
        simple_columns['Baseline LPIPS'] = [f"{m:.4f} ± {s:.4f}" for m, s in zip(
            summary_table['final_baseline_lpips']['mean'], 
            summary_table['final_baseline_lpips']['std']
        )]
        if 'lpips_improvement' in summary_table.columns:
            simple_columns['LPIPS Improvement'] = [f"{m:.4f} ± {s:.4f}" for m, s in zip(
                summary_table['lpips_improvement']['mean'], 
                summary_table['lpips_improvement']['std']
            )]
    
    # Add SSIM metrics to simplified table if available
    if 'final_ssim' in summary_table.columns:
        simple_columns['SSIM'] = [f"{m:.4f} ± {s:.4f}" for m, s in zip(
            summary_table['final_ssim']['mean'], 
            summary_table['final_ssim']['std']
        )]
        simple_columns['Baseline SSIM'] = [f"{m:.4f} ± {s:.4f}" for m, s in zip(
            summary_table['final_baseline_ssim']['mean'], 
            summary_table['final_baseline_ssim']['std']
        )]
        if 'ssim_improvement' in summary_table.columns:
            simple_columns['SSIM Improvement'] = [f"{m:.4f} ± {s:.4f}" for m, s in zip(
                summary_table['ssim_improvement']['mean'], 
                summary_table['ssim_improvement']['std']
            )]
    
    simple_table = pd.DataFrame(simple_columns)
    
    # Sort by scale first, then by PSNR improvement
    if 'Scale' in simple_table.columns:
        simple_table = simple_table.sort_values(['Scale', 'PSNR Improvement'], ascending=[True, False])
    else:
        simple_table = simple_table.sort_values('PSNR Improvement', ascending=False)
        
    simple_table.to_csv(output_dir / 'model_comparison_formatted.csv', index=False)

    # Create additional summaries by learning rate
    for lr in formatted_table['Learning Rate'].unique():
        lr_table = formatted_table[formatted_table['Learning Rate'] == lr]
        lr_table.to_csv(output_dir / f'model_comparison_lr_{lr}_detailed.csv', index=False)
        
        # Also create a simplified version
        simple_lr_table = simple_table[simple_table['Learning Rate'] == lr]
        simple_lr_table.to_csv(output_dir / f'model_comparison_lr_{lr}.csv', index=False)
    
    # Also create summaries by scale if available
    if 'Scale' in formatted_table.columns:
        for scale in formatted_table['Scale'].unique():
            scale_table = formatted_table[formatted_table['Scale'] == scale]
            scale_table.to_csv(output_dir / f'model_comparison_scale_{int(scale)}_detailed.csv', index=False)
            
            # Also create a simplified version
            simple_scale_table = simple_table[simple_table['Scale'] == scale]
            simple_scale_table.to_csv(output_dir / f'model_comparison_scale_{int(scale)}.csv', index=False)
    
    # Also group by model type and projection type for a higher level overview
    # Include iterations and num_samples in the higher level overview as well
    model_groupby_columns = ['model_type', 'projection_type']
    
    # Add scale to the model groupby columns if available
    if 'downsampling_factor' in df.columns:
        model_groupby_columns.insert(0, 'downsampling_factor')  # Insert at the beginning
        
    if 'iterations' in df.columns:
        model_groupby_columns.append('iterations')
    if 'num_samples' in df.columns:
        model_groupby_columns.append('num_samples')
    
    # Define metrics for model type summary
    model_agg_dict = {
        'final_psnr': ['mean', 'std', 'min', 'max', 'count'],
        'psnr_improvement': ['mean', 'std'],
        'final_baseline_psnr': ['mean', 'std']  # Add baseline PSNR
    }
    
    # Add LPIPS metrics for model type summary if available
    if 'final_lpips' in df.columns:
        model_agg_dict['final_lpips'] = ['mean', 'std']
        model_agg_dict['final_baseline_lpips'] = ['mean', 'std']  # Add baseline LPIPS
        if 'lpips_improvement' in df.columns:
            model_agg_dict['lpips_improvement'] = ['mean', 'std']
    
    # Add SSIM metrics for model type summary if available
    if 'final_ssim' in df.columns:
        model_agg_dict['final_ssim'] = ['mean', 'std']
        model_agg_dict['final_baseline_ssim'] = ['mean', 'std']  # Add baseline SSIM
        if 'ssim_improvement' in df.columns:
            model_agg_dict['ssim_improvement'] = ['mean', 'std']
        
    model_summary = df.groupby(model_groupby_columns).agg(model_agg_dict).reset_index()
    
    model_columns = {}
    
    # Add scale as the first column if available
    if 'downsampling_factor' in model_summary.columns:
        model_columns['Scale'] = model_summary['downsampling_factor']
        
    model_columns.update({
        'Model Type': model_summary['model_type'],
        'Projection': model_summary['projection_type'],
    })
    
    # Add iterations and num_samples
    if 'iterations' in model_summary.columns:
        model_columns['Iterations'] = model_summary['iterations']
    if 'num_samples' in model_summary.columns:
        model_columns['LR Samples'] = model_summary['num_samples']
        
    # Add metrics
    model_columns.update({
        'Sample Count': model_summary['final_psnr']['count'],
        'PSNR': [f"{m:.2f} ± {s:.2f}" for m, s in zip(
            model_summary['final_psnr']['mean'], 
            model_summary['final_psnr']['std']
        )],
        'Baseline PSNR': [f"{m:.2f} ± {s:.2f}" for m, s in zip(
            model_summary['final_baseline_psnr']['mean'], 
            model_summary['final_baseline_psnr']['std']
        )],
        'PSNR Improvement': [f"{m:.2f} ± {s:.2f}" for m, s in zip(
            model_summary['psnr_improvement']['mean'], 
            model_summary['psnr_improvement']['std']
        )]
    })
    
    # Add LPIPS metrics to model type summary if available
    if 'final_lpips' in model_summary.columns:
        model_columns['LPIPS'] = [f"{m:.4f} ± {s:.4f}" for m, s in zip(
            model_summary['final_lpips']['mean'], 
            model_summary['final_lpips']['std']
        )]
        model_columns['Baseline LPIPS'] = [f"{m:.4f} ± {s:.4f}" for m, s in zip(
            model_summary['final_baseline_lpips']['mean'], 
            model_summary['final_baseline_lpips']['std']
        )]
        if 'lpips_improvement' in model_summary.columns:
            model_columns['LPIPS Improvement'] = [f"{m:.4f} ± {s:.4f}" for m, s in zip(
                model_summary['lpips_improvement']['mean'], 
                model_summary['lpips_improvement']['std']
            )]
    
    # Add SSIM metrics to model type summary if available
    if 'final_ssim' in model_summary.columns:
        model_columns['SSIM'] = [f"{m:.4f} ± {s:.4f}" for m, s in zip(
            model_summary['final_ssim']['mean'], 
            model_summary['final_ssim']['std']
        )]
        model_columns['Baseline SSIM'] = [f"{m:.4f} ± {s:.4f}" for m, s in zip(
            model_summary['final_baseline_ssim']['mean'], 
            model_summary['final_baseline_ssim']['std']
        )]
        if 'ssim_improvement' in model_summary.columns:
            model_columns['SSIM Improvement'] = [f"{m:.4f} ± {s:.4f}" for m, s in zip(
                model_summary['ssim_improvement']['mean'], 
                model_summary['ssim_improvement']['std']
            )]
    
    model_table = pd.DataFrame(model_columns)
    
    # Sort by scale first, then by PSNR improvement
    if 'Scale' in model_table.columns:
        model_table = model_table.sort_values(['Scale', 'PSNR Improvement'], ascending=[True, False])
    else:
        model_table = model_table.sort_values('PSNR Improvement', ascending=False)
        
    model_table.to_csv(output_dir / 'model_type_comparison.csv', index=False)
    
    # Also create summaries by scale for the model type comparison if available
    if 'Scale' in model_table.columns:
        for scale in model_table['Scale'].unique():
            scale_model_table = model_table[model_table['Scale'] == scale]
            scale_model_table.to_csv(output_dir / f'model_type_comparison_scale_{int(scale)}.csv', index=False)
    
    print(f"Created summary tables for dataset: {dataset_name}")
    return True

def setup_device(args):
    """
    Set up and return the appropriate device for training.
    """
    if torch.cuda.is_available():
        torch.cuda.set_device(f"cuda:{args.d}")  
    device = torch.device(f"cuda:{args.d}" if torch.cuda.is_available() else "cpu")
    return device

def setup_seed(seed):
    """
    Set all random seeds for reproducibility.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # For completely reproducible results
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_result_directories(args, sample_id):
    """
    Create the directory structure for experiment results.
    Returns the path to the results directory.
    """
    # Base results directory
    base_results_dir = Path("results")
    base_results_dir.mkdir(exist_ok=True)

    # First level: dataset name
    dataset_dir = base_results_dir / args.dataset
    dataset_dir.mkdir(exist_ok=True)
    
    # Second level: key experiment parameters
    experiment_name = f"df{args.df}_shift{args.lr_shift:.1f}_samples{args.num_samples}"
    if args.aug != "none":
        experiment_name += f"_aug{args.aug}"
    experiment_dir = dataset_dir / experiment_name
    experiment_dir.mkdir(exist_ok=True)
    
    # Third level: sample ID
    sample_dir = experiment_dir / sample_id
    sample_dir.mkdir(exist_ok=True)
    
    # Normalize input projection for consistent directory naming
    dir_input_projection = args.input_projection
    
    # For Fourier projections, always use the fourier_scale format
    if dir_input_projection == "fourier" or dir_input_projection.startswith("fourier_"):
        # Extract scale from name if present
        fourier_scale = args.fourier_scale
        if "_" in dir_input_projection:
            try:
                scale_str = dir_input_projection.split("_")[1]
                fourier_scale = float(scale_str)
            except (ValueError, IndexError):
                pass
        
        # Use standardized format
        dir_input_projection = f"fourier_{fourier_scale}"
    
    # For Legendre projections, include the max degree
    elif dir_input_projection == "legendre" and hasattr(args, 'legendre_max_degree'):
        dir_input_projection = f"{dir_input_projection}_{args.legendre_max_degree}"

    # Create directory using consistent naming, including all key parameters
    model_dir_name = f"{args.model}_{dir_input_projection}_{args.iters}_lr{args.learning_rate}"
    
    # Add network architecture details
    model_dir_name += f"_d{args.network_depth}_h{args.network_hidden_dim}"
    
    # Add optional flags
    if args.use_gnll:
        model_dir_name += "_gnll"
    if args.use_dual_optimizer:
        model_dir_name += "_dual"
    if args.rotation:
        model_dir_name += "_rot"
    
    results_dir = sample_dir / model_dir_name
    results_dir.mkdir(exist_ok=True)

    return results_dir

def save_config(args, results_dir, sample_id):
    """
    Save experiment configuration to a JSON file.
    """
    config = {
        # Dataset parameters
        "dataset": args.dataset,
        "sample_id": sample_id,
        "downsampling_factor": args.df,
        "lr_shift": args.lr_shift,
        "num_samples": args.num_samples,
        "augmentation": args.aug,
        "keep_rggb": args.keep_rggb,
        "crop_margin": args.crop_margin,  # Add the crop margin parameter
        
        # Model parameters
        "model": args.model,
        "network_depth": args.network_depth,
        "network_hidden_dim": args.network_hidden_dim,
        "projection_dim": args.projection_dim,
        "rotation": args.rotation,
        
        # Training parameters
        "iterations": args.iters,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "batch_size": args.bs,
        "seed": args.seed,
        "use_gt": args.use_gt
    }
    
    with open(results_dir / "experiment_config.json", "w") as f:
        json.dump(config, f, indent=2)

def initialize_model(args, device, num_samples):
    """
    Initialize the model, optimizers, and schedulers.
    """
    input_projection = get_input_projection(
        args.input_projection, 2, args.projection_dim, device, 
        args.fourier_scale, args.legendre_max_degree, args.activation
    )

    if args.input_projection == "legendre":
        original_dim = args.projection_dim
        args.projection_dim = input_projection.get_output_dim()
        print(f"Legendre projection: changing projection_dim from {original_dim} to {args.projection_dim}")

    # Adjust output dimension based on grayscale or RGGB flag
    if args.grayscale:
        output_dim = 1
    elif args.keep_rggb:
        output_dim = 4  # RGGB format (4 channels)
    else:
        output_dim = args.output_dim  # Default is 3 (RGB)
    
    decoder = get_decoder(args.model, args.network_depth, args.projection_dim, args.network_hidden_dim, output_dim=output_dim)
    model = INR(input_projection, decoder, num_samples, use_gnll=args.use_gnll, disable_shifts=args.disable_shifts, disable_frame_decoder=args.disable_frame_decoder).to(device)

    # Fix parameter passing to optimizer
    if args.use_gnll:
        recon_params = list(model.decoder.parameters()) + list(model.color_transforms.parameters()) + list(model.variance_predictor.parameters())
    else:
        recon_params = list(model.decoder.parameters()) + list(model.color_transforms.parameters())
    
    if args.use_dual_optimizer:
        optimizer = optim.AdamW(recon_params, lr=args.learning_rate, weight_decay=args.weight_decay)
        shift_optimizer = optim.AdamW(model.shift_vectors.parameters(), lr=args.learning_rate, weight_decay=0.0)
    else:
        optimizer = optim.AdamW(recon_params + list(model.shift_vectors.parameters()), lr=args.learning_rate, weight_decay=args.weight_decay)
        shift_optimizer = None
        
    # Use full number of iterations as T_max for one long cycle
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.iters, eta_min=1e-6)
    shift_scheduler = None
    if args.use_dual_optimizer:
        shift_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            shift_optimizer, T_0=args.iters, T_mult=1, eta_min=1e-7
        )

    return model, optimizer, shift_optimizer, scheduler, shift_scheduler

def initialize_history():
    """
    Initialize the history dictionary for tracking metrics.
    """
    return {
        'iterations': [],
        'recon_loss': [],  # This will hold monitor_recon_loss (MSE) for consistency
        'trans_loss': [],
        'recon_loss_keepout': [],  # This will hold monitor_recon_loss_keepout (MSE) for consistency
        'trans_loss_keepout': [],
        'total_loss_keepout': [],  # Total of monitor metrics
        'test_loss': [],
        'psnr': [],
        'lpips': [],
        'ssim': [],
        'baseline_psnr': [],
        'baseline_lpips': [],
        'baseline_ssim': [],
        'translation_data': [],
        'learning_rate': [],
        # Validation metrics for keep-out samples (using MSE for consistency)
        'recon_loss_keepout_val': [],
        'trans_loss_keepout_val': [],
        'total_loss_keepout_val': [],
        # Original training loss metrics when using GNLL (for reference)
        'original_recon_loss': [],
        'original_recon_loss_keepout': [],
        'original_recon_loss_keepout_val': [],
        # EMA smoothed metrics
        'ema_train_loss': [],
        'ema_val_loss': []
    }

def setup_wandb(args, sample_id, downsample_factor, lr_shift, num_samples, learning_rate):
    """
    Initialize Weights & Biases logging if enabled.
    """
    if not args.wandb:
        return

    dataset_name = f"lr_factor_{downsample_factor}x_shift_{lr_shift:.1f}px_samples_{num_samples}_aug_{args.aug}"
    wandb.init(
        project="satellite-super-res",
        name=dataset_name,
        group=args.model,  # Group runs by model type
        config={
            "model": args.model,
            "downsample_factor": downsample_factor,
            "lr_shift": lr_shift,
            "num_samples": num_samples,
            "iters": args.iters,
            "network_hidden_dim": args.network_hidden_dim,
            "network_depth": args.network_depth,
            "learning_rate": learning_rate,
            "use_gt": args.use_gt,
            "augmentation": args.aug
        }
    )

def load_dataset(args, sample_id, downsample_factor, lr_shift):
    """
    Load and prepare the dataset for training.
    """
    if args.dataset == "satburst_synth":
        args.root_satburst_synth = f"data/{sample_id}/scale_{downsample_factor}_shift_{lr_shift:.1f}px_aug_{args.aug}"
    elif args.dataset.startswith("handheld"):
        args.root_handheld = args.dataset
    
    # Set the sample ID for dataset loading
    args.sample_id = sample_id

    # Load the dataset
    train_data = get_dataset(args=args, name=args.dataset)
    train_dataloader = DataLoader(train_data, batch_size=args.bs, shuffle=True)
    
    return train_data, train_dataloader

def run_training_loop(model, optimizer, shift_optimizer, scheduler, shift_scheduler, 
                     train_data, train_dataloader, device, args, history, results_dir):
    """
    Execute the training loop with periodic evaluation and visualization.
    """
    # Initialize metrics from evals.py
    loss_fn_alex = lpips.LPIPS(net='alex').to(device)  # Keep using original LPIPS for compatibility
    psnr_metric = PSNR(max_value=1.0)
    ssim_metric = SSIM(use_for_loss=False)
    
    # Log crop margin being used
    print(f"Using crop margin of {args.crop_margin} pixels when calculating metrics")
    
    # Initialize containers for tracking metrics
    image_outputs = []
    image_iterations = []
    image_metrics = {'psnr': [], 'lpips': [], 'ssim': [], 'baseline_psnr': []}
    
    # Create progress bar
    progress_bar = tqdm(total=args.iters, desc="Training")
    
    # EMA (Exponential Moving Average) variables for loss smoothing
    ema_train_loss = None
    ema_val_loss = None
    ema_alpha = 0.8  # EMA coefficient: higher values give more weight to recent observations
    
    # Create a checkpoint directory for final model
    checkpoint_dir = results_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Main training loop
    i = 0
    aggregate_train_losses = {}
    
    while i < args.iters:
        for train_sample in train_dataloader:
            train_losses = train_one_iteration(
                model, optimizer, shift_optimizer, train_sample, 
                device, args.df, iteration=i+1, use_gt=args.use_gt
            )
            
            # Step schedulers
            if args.use_dual_optimizer:
                scheduler.step()
                shift_scheduler.step()
            else:
                scheduler.step()
                
            i += 1

            # Accumulate losses for this iteration
            for key, value in train_losses.items():
                if key not in aggregate_train_losses:
                    aggregate_train_losses[key] = []
                aggregate_train_losses[key].append(value)
            
            # Update progress bar
            progress_bar.update(1)
            progress_bar.set_postfix({
                'recon_loss': f"{train_losses['monitor_recon_loss']:.4f}",  # Show MSE for monitor
                'trans_loss': f"{train_losses['trans_loss']:.4f}"
            })
            
            # Periodic evaluation and logging
            if (i + 1) % 50 == 0:
                with torch.no_grad():
                    # Test the model
                    test_loss, test_output, hr_image = test_one_epoch(model, train_data, device)

                    # Run validation on keep-out samples
                    validation_results = validate_dataset(model, train_dataloader, device, args.df)

                    # Prepare output tensors for evaluation
                    test_output = einops.rearrange(test_output, 'b h w c -> b c h w')
                    hr_image = einops.rearrange(hr_image, 'b h w c -> b c h w')
                    lr_sample = train_data.get_lr_sample(0)

                    lr_sample = lr_sample.unsqueeze(0).to(device)
                    if lr_sample.shape[-1] == 3:
                        lr_sample = lr_sample.permute(0, 3, 1, 2)
                    baseline_pred = bilinear_resize_torch(lr_sample, (hr_image.shape[2], hr_image.shape[3]))

                    # Create versions for visualization and metrics
                    test_output_for_metrics = test_output
                    hr_image_for_metrics = hr_image
                    baseline_for_metrics = baseline_pred

                    if i == args.iters:
                        print("Aligning outputs for fair comparison")
                        gauss_kernel, ksz = get_gaussian_kernel(sd=1.5)
                        # Align outputs for fair comparison - consistent with final visualization
                        test_output_for_metrics = align_kornia_brute_force(test_output_for_metrics.squeeze(0), hr_image_for_metrics.squeeze(0)).unsqueeze(0)
                        test_output_aligned, _ = match_colors(test_output_for_metrics, hr_image_for_metrics, test_output_for_metrics, ksz, gauss_kernel)
                        
                        baseline_for_metrics = align_kornia_brute_force(baseline_for_metrics.squeeze(0), hr_image_for_metrics.squeeze(0)).unsqueeze(0)
                        baseline_pred_aligned, _ = match_colors(baseline_for_metrics, hr_image_for_metrics, baseline_for_metrics, ksz, gauss_kernel)
                    else:
                        test_output_aligned = test_output_for_metrics
                        baseline_pred_aligned = baseline_for_metrics

                    # Calculate metrics using the pre-initialized LPIPS model
                    baseline_metrics = calculate_metrics(baseline_pred_aligned, hr_image_for_metrics, loss_fn_alex=loss_fn_alex, crop_margin=args.crop_margin)
                    model_metrics = calculate_metrics(test_output_aligned, hr_image_for_metrics, loss_fn_alex=loss_fn_alex, crop_margin=args.crop_margin)
                    
                    # Store outputs and metrics for visualization
                    image_outputs.append(test_output.clone())
                    image_iterations.append(i + 1)
                    image_metrics['psnr'].append(model_metrics['psnr'])
                    image_metrics['lpips'].append(model_metrics['lpips'])
                    image_metrics['ssim'].append(model_metrics['ssim'])
                    image_metrics['baseline_psnr'].append(baseline_metrics['psnr'])
                    
                # Update history with metrics
                history['iterations'].append(i + 1)
                # Use monitor_recon_loss (MSE) for tracking even when using GNLL for training
                history['recon_loss'].append(aggregate_train_losses['monitor_recon_loss'][-1])
                history['trans_loss'].append(aggregate_train_losses['trans_loss'][-1])
                history['recon_loss_keepout'].append(aggregate_train_losses['monitor_recon_loss_keepout'][-1])
                history['trans_loss_keepout'].append(aggregate_train_losses['trans_loss_keepout'][-1])
                history['total_loss_keepout'].append(aggregate_train_losses['monitor_recon_loss_keepout'][-1] + 
                                                  aggregate_train_losses['trans_loss_keepout'][-1])
                history['test_loss'].append(test_loss)
                history['psnr'].append(model_metrics['psnr'])
                history['lpips'].append(model_metrics['lpips'])
                history['ssim'].append(model_metrics['ssim'])
                history['baseline_psnr'].append(baseline_metrics['psnr'])
                history['baseline_lpips'].append(baseline_metrics['lpips'])
                history['baseline_ssim'].append(baseline_metrics['ssim'])
                
                # Also keep track of original loss values when using GNLL
                if model.use_gnll:
                    history['original_recon_loss'].append(aggregate_train_losses['recon_loss'][-1])
                    history['original_recon_loss_keepout'].append(aggregate_train_losses['recon_loss_keepout'][-1])
                    history['original_recon_loss_keepout_val'].append(validation_results['recon_loss_keepout_val'])

                # Update history with validation metrics - use monitor metrics for consistency
                history['recon_loss_keepout_val'].append(validation_results['monitor_recon_loss_keepout_val'])
                history['trans_loss_keepout_val'].append(validation_results['trans_loss_keepout_val'])
                history['total_loss_keepout_val'].append(validation_results['monitor_recon_loss_keepout_val'] + 
                                                     validation_results['trans_loss_keepout_val'])

                # Store translation data
                translation_data = []
                for idx in range(len(train_data)):
                    gt_shift = train_data[idx]['shifts']
                    pred_shift = model.shift_vectors[idx]

                    # Convert target shifts to numpy arrays
                    if isinstance(gt_shift["dx_percent"], torch.Tensor):
                        gt_dx = gt_shift["dx_percent"].detach().cpu().numpy()
                        gt_dy = gt_shift["dy_percent"].detach().cpu().numpy()
                    else:
                        gt_dx = float(gt_shift["dx_percent"])
                        gt_dy = float(gt_shift["dy_percent"])

                    # Get predicted shifts
                    pred_dx = pred_shift[0].detach().cpu().numpy()
                    pred_dy = pred_shift[1].detach().cpu().numpy()

                    translation_data.append({
                        'target_dx': gt_dx,
                        'target_dy': gt_dy,
                        'pred_dx': pred_dx,
                        'pred_dy': pred_dy,
                    })

                # Store translation data in history
                history['translation_data'].append(translation_data)
                history['learning_rate'].append(scheduler.get_last_lr()[0])

                # Get current training and validation losses
                train_recon_loss = aggregate_train_losses['monitor_recon_loss'][-1]
                val_recon_loss = validation_results['monitor_recon_loss_keepout_val']
                
                # Apply exponential moving average to smooth the losses
                if ema_train_loss is None:
                    # Initialize EMAs with first values
                    ema_train_loss = train_recon_loss
                    ema_val_loss = val_recon_loss
                else:
                    # Update EMAs with new values
                    ema_train_loss = ema_alpha * train_recon_loss + (1 - ema_alpha) * ema_train_loss
                    ema_val_loss = ema_alpha * val_recon_loss + (1 - ema_alpha) * ema_val_loss
                
                # Store the smoothed losses in history for visualization
                if 'ema_train_loss' not in history:
                    history['ema_train_loss'] = []
                    history['ema_val_loss'] = []
                
                history['ema_train_loss'].append(ema_train_loss)
                history['ema_val_loss'].append(ema_val_loss)

                # Print metrics - use monitor_recon_loss for display
                print(f"Iter {i+1}: "
                    f"Train recon {'(MSE monitor)' if model.use_gnll else ''}: {aggregate_train_losses['monitor_recon_loss'][-1]:.6f}, "
                    f"trans: {aggregate_train_losses['trans_loss'][-1]:.6f}, "
                    f"total: {aggregate_train_losses['monitor_recon_loss'][-1] + aggregate_train_losses['trans_loss'][-1]:.6f}, "
                    f"Test: {test_loss:.6f}\n"
                    f"Keep-out recon {'(MSE monitor)' if model.use_gnll else ''}: {aggregate_train_losses['monitor_recon_loss_keepout'][-1]:.6f}, "
                    f"Keep-out trans: {aggregate_train_losses['trans_loss_keepout'][-1]:.6f}, "
                    f"Keep-out total: {aggregate_train_losses['monitor_recon_loss_keepout'][-1] + aggregate_train_losses['trans_loss_keepout'][-1]:.6f}\n"
                    f"Validation keep-out recon {'(MSE monitor)' if model.use_gnll else ''}: {validation_results['monitor_recon_loss_keepout_val']:.6f}, "
                    f"Validation keep-out trans: {validation_results['trans_loss_keepout_val']:.6f}, "
                    f"Validation keep-out total: {validation_results['monitor_recon_loss_keepout_val'] + validation_results['trans_loss_keepout_val']:.6f}\n"
                    f"Smoothed - Train: {ema_train_loss:.6f}, Val: {ema_val_loss:.6f}\n"
                    f"Metrics vs Baseline:\n"
                    f"PSNR: {model_metrics['psnr']:.2f}dB vs {baseline_metrics['psnr']:.2f}dB\n"
                    f"LPIPS: {model_metrics['lpips']:.4f} vs {baseline_metrics['lpips']:.4f} (lower is better)\n"
                    f"SSIM: {model_metrics['ssim']:.4f} vs {baseline_metrics['ssim']:.4f} (higher is better)")

                # Log to wandb if enabled
                if args.wandb:
                    wandb_log_dict = {
                        "iteration": i + 1,
                        "train/recon_loss": aggregate_train_losses['recon_loss'][-1],  # Original loss used for training
                        "train/monitor_recon_loss": aggregate_train_losses['monitor_recon_loss'][-1],  # MSE for monitoring
                        "train/trans_loss": aggregate_train_losses['trans_loss'][-1],
                        "train/total_loss": aggregate_train_losses['recon_loss'][-1] + aggregate_train_losses['trans_loss'][-1],
                        "train/monitor_total_loss": aggregate_train_losses['monitor_recon_loss'][-1] + aggregate_train_losses['trans_loss'][-1],
                        "train/recon_loss_keepout": aggregate_train_losses['recon_loss_keepout'][-1],
                        "train/monitor_recon_loss_keepout": aggregate_train_losses['monitor_recon_loss_keepout'][-1],
                        "train/trans_loss_keepout": aggregate_train_losses['trans_loss_keepout'][-1],
                        "train/total_loss_keepout": aggregate_train_losses['recon_loss_keepout'][-1] + aggregate_train_losses['trans_loss_keepout'][-1],
                        "train/monitor_total_loss_keepout": aggregate_train_losses['monitor_recon_loss_keepout'][-1] + aggregate_train_losses['trans_loss_keepout'][-1],
                        "validation/recon_loss_keepout": validation_results['recon_loss_keepout_val'],
                        "validation/monitor_recon_loss_keepout": validation_results['monitor_recon_loss_keepout_val'],
                        "validation/trans_loss_keepout": validation_results['trans_loss_keepout_val'],
                        "validation/total_loss_keepout": validation_results['recon_loss_keepout_val'] + validation_results['trans_loss_keepout_val'],
                        "validation/monitor_total_loss_keepout": validation_results['monitor_recon_loss_keepout_val'] + validation_results['trans_loss_keepout_val'],
                        "test/loss": test_loss,
                        "test/psnr": model_metrics['psnr'],
                        "test/lpips": model_metrics['lpips'],
                        "test/ssim": model_metrics['ssim'],
                        "test/baseline_psnr": baseline_metrics['psnr'],
                        "test/baseline_lpips": baseline_metrics['lpips'],
                        "test/baseline_ssim": baseline_metrics['ssim'],
                        "learning_rate": scheduler.get_last_lr()[0],
                        "metrics/psnr_improvement": model_metrics['psnr'] - baseline_metrics['psnr'],
                        "metrics/lpips_improvement": baseline_metrics['lpips'] - model_metrics['lpips'],
                        "metrics/ssim_improvement": model_metrics['ssim'] - baseline_metrics['ssim'],
                    }
                    
                    # Log smoothed losses
                    if ema_train_loss is not None:
                        wandb_log_dict["train/ema_loss"] = ema_train_loss
                        wandb_log_dict["validation/ema_loss"] = ema_val_loss
                    
                    wandb.log(wandb_log_dict)

                # Update progress bar description with validation/train ratio
                progress_bar.set_description(
                    f"Train: {ema_train_loss:.4f} | "
                    f"Val: {ema_val_loss:.4f} | "
                    f"PSNR: {model_metrics['psnr']:.2f}dB"
                )
                
                # Reset accumulated losses
                aggregate_train_losses = {}

            # Break if reached total iterations
            if i >= args.iters:
                break
    
    # Close progress bar
    progress_bar.close()
    
    # Save the final model
    checkpoint_path = checkpoint_dir / f'final_model_iter_{i}.pt'
    torch.save({
        'iteration': i,
        'model': copy.deepcopy(model.state_dict()),
        'metrics': model_metrics,
        'optimizer': copy.deepcopy(optimizer.state_dict()) if optimizer else None,
        'shift_optimizer': copy.deepcopy(shift_optimizer.state_dict()) if shift_optimizer else None,
        'ema_train_loss': ema_train_loss,
        'ema_val_loss': ema_val_loss
    }, checkpoint_path)
    
    # Create simple loss plot to show training progress
    if len(history['iterations']) > 1:
        plt.figure(figsize=(10, 6))
        
        # Create a figure with 2 subplots
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8))
        
        # Plot: Smoothed losses
        ax1.plot(history['iterations'], history['ema_train_loss'], 'b-', label='EMA Train Loss')
        ax1.plot(history['iterations'], history['ema_val_loss'], 'r-', label='EMA Validation Loss')
        ax1.set_title('Smoothed Training and Validation Losses')
        ax1.set_ylabel('Loss (log scale)')
        ax1.set_xlabel('Iteration')
        ax1.set_yscale('log')
        ax1.legend()
        ax1.grid(True)
        
        plt.tight_layout()
        plt.savefig(results_dir / 'training_losses.png')
        plt.close()
    
    # Return the metrics and the tracking info (no best model or overfitting detection)
    return history, train_losses, None, None, None, None, None

def visualize_results(model, train_data, history, results_dir, device, args):
    """
    Create and save visualizations of the training results.
    """
    test_loss = 0.0  # Default test_loss value
    
    try:
        # Generate standard visualizations
        plot_training_curves(history, save_path=results_dir / 'final_training_curves.png', using_gnll=model.use_gnll)
        
        # Create visualization for last translation data
        last_trans_data = history['translation_data'][-1]
        
        # Convert translation data for visualization
        pred_dx_list = []
        pred_dy_list = []
        target_dx_list = []
        target_dy_list = []

        for sample_data in last_trans_data:
            pred_dx_list.append(float(sample_data['pred_dx']))
            pred_dy_list.append(float(sample_data['pred_dy']))
            target_dx_list.append(float(sample_data['target_dx']))
            target_dy_list.append(float(sample_data['target_dy']))

        # Convert to numpy arrays
        pred_dx_array = np.array(pred_dx_list)
        pred_dy_array = np.array(pred_dy_list)
        target_dx_array = np.array(target_dx_list)
        target_dy_array = np.array(target_dy_list)

        # Visualize translations for final model
        visualize_translations(
            pred_dx_array,
            pred_dy_array,
            target_dx_array,
            target_dy_array,
            save_path=results_dir / 'final_translation_vis.png'
        )
        
        # Visualize shift evolution for full history
        visualize_shift_evolution(
            history,
            save_path=results_dir / 'shift_evolution.png'
        )
        
        # Generate animation if requested
        if args.use_full_visualization:
            print("Generating full visualizations (animations)...")
            visualize_shift_evolution_animation_frames(
                history,
                save_path=results_dir / 'shift_evolution_animation.gif'
            )

        # Initialize metrics from evals.py
        loss_fn_alex = lpips.LPIPS(net='alex').to(device)  # Keep using original LPIPS for compatibility
        psnr_metric = PSNR(max_value=1.0)
        ssim_metric = SSIM(use_for_loss=False)
        
        # Run inference with the current model
        with torch.no_grad():
            test_loss, test_output, hr_image = test_one_epoch(model, train_data, device)
            
            # Convert from [B, H, W, C] to [B, C, H, W] format
            test_output = einops.rearrange(test_output, 'b h w c -> b c h w')
            hr_image = einops.rearrange(hr_image, 'b h w c -> b c h w')
            
            # Create RGB versions for visualization and metrics
            if args.keep_rggb and test_output.shape[1] == 4:
                # Extract RGGB channels
                R_out = test_output[:, 0:1]
                G1_out = test_output[:, 1:2]
                G2_out = test_output[:, 2:3]
                B_out = test_output[:, 3:4]
                
                # Save individual RGGB channels for analysis
                rggb_dir = results_dir / 'rggb_channels'
                rggb_dir.mkdir(exist_ok=True)
                
                # Convert to numpy and save each channel
                R_out_np = R_out.cpu().permute(0, 2, 3, 1).numpy()[0, ..., 0]
                G1_out_np = G1_out.cpu().permute(0, 2, 3, 1).numpy()[0, ..., 0]
                G2_out_np = G2_out.cpu().permute(0, 2, 3, 1).numpy()[0, ..., 0]
                B_out_np = B_out.cpu().permute(0, 2, 3, 1).numpy()[0, ..., 0]
                
                plt.figure(figsize=(20, 5))
                
                plt.subplot(1, 4, 1)
                plt.imshow(R_out_np, cmap='gray')
                plt.title('R Channel')
                plt.axis('off')
                
                plt.subplot(1, 4, 2)
                plt.imshow(G1_out_np, cmap='gray')
                plt.title('G1 Channel')
                plt.axis('off')
                
                plt.subplot(1, 4, 3)
                plt.imshow(G2_out_np, cmap='gray')
                plt.title('G2 Channel')
                plt.axis('off')
                
                plt.subplot(1, 4, 4)
                plt.imshow(B_out_np, cmap='gray')
                plt.title('B Channel')
                plt.axis('off')
                
                plt.tight_layout()
                plt.savefig(rggb_dir / 'rggb_channels.png')
                plt.close()
                
                # Also save individual channel differences if GT has RGGB
                if hr_image.shape[1] == 4:
                    R_gt = hr_image[:, 0:1]
                    G1_gt = hr_image[:, 1:2]
                    G2_gt = hr_image[:, 2:3]
                    B_gt = hr_image[:, 3:4]
                    
                    # Calculate differences
                    R_diff = abs(R_out - R_gt)
                    G1_diff = abs(G1_out - G1_gt)
                    G2_diff = abs(G2_out - G2_gt)
                    B_diff = abs(B_out - B_gt)
                    
                    # Convert to numpy
                    R_diff_np = R_diff.cpu().permute(0, 2, 3, 1).numpy()[0, ..., 0]
                    G1_diff_np = G1_diff.cpu().permute(0, 2, 3, 1).numpy()[0, ..., 0]
                    G2_diff_np = G2_diff.cpu().permute(0, 2, 3, 1).numpy()[0, ..., 0]
                    B_diff_np = B_diff.cpu().permute(0, 2, 3, 1).numpy()[0, ..., 0]
                    
                    plt.figure(figsize=(20, 5))
                    
                    plt.subplot(1, 4, 1)
                    plt.imshow(R_diff_np, cmap='hot')
                    plt.title('R Channel Difference')
                    plt.axis('off')
                    
                    plt.subplot(1, 4, 2)
                    plt.imshow(G1_diff_np, cmap='hot')
                    plt.title('G1 Channel Difference')
                    plt.axis('off')
                    
                    plt.subplot(1, 4, 3)
                    plt.imshow(G2_diff_np, cmap='hot')
                    plt.title('G2 Channel Difference')
                    plt.axis('off')
                    
                    plt.subplot(1, 4, 4)
                    plt.imshow(B_diff_np, cmap='hot')
                    plt.title('B Channel Difference')
                    plt.axis('off')
                    
                    plt.tight_layout()
                    plt.savefig(rggb_dir / 'rggb_differences.png')
                    plt.close()
                
                # Average G1 and G2 for visualization
                G_out = (G1_out + G2_out) / 2
                
                # Create RGB tensor
                test_output_rgb = torch.cat([R_out, G_out, B_out], dim=1)
                
                # Use this RGB version for visualization and metrics
                test_output_for_vis = test_output_rgb
            else:
                test_output_for_vis = test_output
            
            # Handle GT if it's in RGGB format
            if args.keep_rggb and hr_image.shape[1] == 4:
                R_hr = hr_image[:, 0:1]
                G1_hr = hr_image[:, 1:2]
                G2_hr = hr_image[:, 2:3]
                B_hr = hr_image[:, 3:4]
                
                # Average G1 and G2 for visualization
                G_hr = (G1_hr + G2_hr) / 2
                
                # Create RGB tensor
                hr_image_rgb = torch.cat([R_hr, G_hr, B_hr], dim=1)
                
                # Use this RGB version for visualization and metrics
                hr_image_for_vis = hr_image_rgb
            else:
                hr_image_for_vis = hr_image
            
            # Get baseline prediction for comparison
            lr_sample = train_data.get_lr_sample(0).unsqueeze(0).to(device)
            
            # Convert lr_sample to correct format if needed
            if lr_sample.ndim == 3:  # [H, W, C]
                lr_sample = lr_sample.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
            elif lr_sample.ndim == 4 and lr_sample.shape[1] != 3 and lr_sample.shape[1] != 4:  # Not [B, C, H, W]
                lr_sample = lr_sample.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
            
            # Handle RGGB format in lr_sample
            if args.keep_rggb and lr_sample.shape[1] == 4:
                # Extract and combine G channels for visualization
                R_lr = lr_sample[:, 0:1]
                G1_lr = lr_sample[:, 1:2]
                G2_lr = lr_sample[:, 2:3]
                B_lr = lr_sample[:, 3:4]
                
                # Average G1 and G2 for visualization
                G_lr = (G1_lr + G2_lr) / 2
                
                # Create RGB tensor
                lr_sample_rgb = torch.cat([R_lr, G_lr, B_lr], dim=1)
                baseline_pred = bilinear_resize_torch(lr_sample_rgb, (hr_image_for_vis.shape[2], hr_image_for_vis.shape[3]))
            else:
                baseline_pred = bilinear_resize_torch(lr_sample, (hr_image_for_vis.shape[2], hr_image_for_vis.shape[3]))
            
            # Align outputs for fair comparison
            gauss_kernel, ksz = get_gaussian_kernel(sd=1.5)
            test_output_aligned = align_kornia_brute_force(test_output_for_vis.squeeze(0), hr_image_for_vis.squeeze(0)).unsqueeze(0)
            test_output_aligned, _ = match_colors(lr_sample, hr_image_for_vis, test_output_aligned, ksz, gauss_kernel)
            baseline_pred_aligned = align_kornia_brute_force(baseline_pred.squeeze(0), hr_image_for_vis.squeeze(0)).unsqueeze(0)
            baseline_pred_aligned, _ = match_colors(lr_sample, hr_image_for_vis, baseline_pred_aligned, ksz, gauss_kernel)
            
            # Calculate metrics
            model_metrics = calculate_metrics(test_output_aligned, hr_image_for_vis, loss_fn_alex=loss_fn_alex, crop_margin=args.crop_margin)
            baseline_metrics = calculate_metrics(baseline_pred_aligned, hr_image_for_vis, loss_fn_alex=loss_fn_alex, crop_margin=args.crop_margin)
            
            # Convert outputs to numpy for visualization
            hr_image_np = hr_image_for_vis.cpu().permute(0, 2, 3, 1).numpy()
            test_output_np = test_output_aligned.cpu().permute(0, 2, 3, 1).numpy()
            baseline_pred_np = baseline_pred_aligned.cpu().permute(0, 2, 3, 1).numpy()
            
            # Print a summary of the metrics for the final model
            print(f"\nFinal model metrics:")
            print(f"PSNR: {model_metrics['psnr']:.2f} dB (Baseline: {baseline_metrics['psnr']:.2f} dB)")
            print(f"LPIPS: {model_metrics['lpips']:.4f} (Baseline: {baseline_metrics['lpips']:.4f})")
            print(f"SSIM: {model_metrics['ssim']:.4f} (Baseline: {baseline_metrics['ssim']:.4f})")
        
        # Prepare LR target image for visualization
        lr_target_img = train_data.get_lr_sample(0)
        if torch.is_tensor(lr_target_img):
            # Handle tensor format conversion
            if lr_target_img.dim() < 3:  # If grayscale with no channel dimension
                lr_target_img = lr_target_img.unsqueeze(-1)  # Add channel dimension
            elif lr_target_img.shape[0] == 3 or lr_target_img.shape[0] == 4:  # If [C, H, W]
                lr_target_img = lr_target_img.permute(1, 2, 0)  # Convert to [H, W, C]
            
            # Handle RGGB format for visualization
            if args.keep_rggb and lr_target_img.shape[2] == 4:
                R = lr_target_img[..., 0]
                G = (lr_target_img[..., 1] + lr_target_img[..., 2]) / 2  # Average G1 and G2
                B = lr_target_img[..., 3]
                lr_target_img = torch.stack([R, G, B], dim=-1)
            
            # Convert to numpy for visualization
            lr_target_img = lr_target_img.cpu().numpy()

        # Create side-by-side comparison
        plt.figure(figsize=(24, 6))
        
        plt.subplot(1, 4, 1)
        plt.imshow(hr_image_np[0])
        plt.title('HR GT')
        plt.axis('off')
        
        plt.subplot(1, 4, 2)
        plt.imshow(lr_target_img)
        plt.title('LR Reference (Sample 00)')
        plt.axis('off')
        
        plt.subplot(1, 4, 3)
        plt.imshow(baseline_pred_np[0])
        plt.title(f'Bilinear\nPSNR: {baseline_metrics["psnr"]:.2f} dB\nLPIPS: {baseline_metrics["lpips"]:.4f}\nSSIM: {baseline_metrics["ssim"]:.4f}')
        plt.axis('off')
        
        plt.subplot(1, 4, 4)
        plt.imshow(test_output_np[0])
        plt.title(f'Final Model\nPSNR: {model_metrics["psnr"]:.2f} dB\nLPIPS: {model_metrics["lpips"]:.4f}\nSSIM: {model_metrics["ssim"]:.4f}')
        plt.axis('off')
        
        plt.suptitle(f'Test loss: {test_loss:.6f} | Model PSNR: {model_metrics["psnr"]:.2f} dB\n'
                    f'Baseline PSNR: {baseline_metrics["psnr"]:.2f} dB | Dataset scale: {args.df}x\n'
                    f'Iterations: {args.iters} | learning_rate: {args.learning_rate}\n'
                    f'Network: {args.model} | Hidden dim: {args.network_hidden_dim} | Depth: {args.network_depth} | Projection: {args.input_projection}'
                    f'{" | RGGB mode" if args.keep_rggb else ""}')
        plt.tight_layout()
        
        plt.savefig(results_dir / 'comparison.png')
        plt.close()

        # Save individual images as-is without any modifications
        # For test_output (model prediction)
        test_output_raw = test_output_for_vis.cpu()
        if test_output_raw.shape[0] == 1:  # Remove batch dimension if present
            test_output_raw = test_output_raw[0]
        # Save as numpy array
        np.save(results_dir / 'output_prediction_final.npy', test_output_raw.numpy())
        # Also save as image for easy viewing
        test_output_img = test_output_raw.permute(1, 2, 0).numpy()
        plt.imsave(results_dir / 'output_prediction_final.png', np.clip(test_output_img, 0, 1))

        # For HR ground truth
        hr_image_raw = hr_image_for_vis.cpu()
        if hr_image_raw.shape[0] == 1:  # Remove batch dimension if present
            hr_image_raw = hr_image_raw[0]
        # Save as numpy array
        np.save(results_dir / 'hr_ground_truth.npy', hr_image_raw.numpy())
        # Also save as image for easy viewing
        hr_image_img = hr_image_raw.permute(1, 2, 0).numpy()
        plt.imsave(results_dir / 'hr_ground_truth.png', np.clip(hr_image_img, 0, 1))

        # For LR input
        if torch.is_tensor(lr_sample):
            lr_sample_raw = lr_sample.cpu()
            if lr_sample_raw.shape[0] == 1:  # Remove batch dimension if present
                lr_sample_raw = lr_sample_raw[0]
            # Save as numpy array
            np.save(results_dir / 'lr_input.npy', lr_sample_raw.numpy())
            # Also save as image for easy viewing
            if lr_sample_raw.shape[0] in [1, 3, 4]:  # If in [C, H, W] format
                lr_sample_img = lr_sample_raw.permute(1, 2, 0).numpy()
                if lr_sample_img.shape[2] == 1:  # If grayscale
                    lr_sample_img = lr_sample_img[:, :, 0]
                plt.imsave(results_dir / 'lr_input.png', np.clip(lr_sample_img, 0, 1))
            else:
                plt.imsave(results_dir / 'lr_input.png', np.clip(lr_sample_raw.numpy(), 0, 1))

        # For baseline prediction
        baseline_pred_raw = baseline_pred.cpu()
        if baseline_pred_raw.shape[0] == 1:  # Remove batch dimension if present
            baseline_pred_raw = baseline_pred_raw[0]
        # Save as numpy array
        np.save(results_dir / 'baseline_prediction.npy', baseline_pred_raw.numpy())
        # Also save as image for easy viewing
        baseline_pred_img = baseline_pred_raw.permute(1, 2, 0).numpy()
        plt.imsave(results_dir / 'baseline_prediction.png', np.clip(baseline_pred_img, 0, 1))
        
        # Create metrics dictionary with all the calculated metrics
        metrics = {
            'psnr': model_metrics['psnr'],
            'psnr_baseline': baseline_metrics['psnr'],
            'lpips': model_metrics['lpips'],
            'baseline_lpips': baseline_metrics['lpips'],
            'ssim': model_metrics['ssim'],
            'baseline_ssim': baseline_metrics['ssim'],
            'model_type_for_metrics': 'final_model'
        }
            
        return metrics
    
    except Exception as e:
        print(f"Error in visualize_results: {e}")
        traceback.print_exc()  # Print the full traceback for debugging
        # Return basic metrics so the program can continue
        return {
            'psnr': 0.0,
            'psnr_baseline': 0.0,
            'lpips': 0.0,
            'baseline_lpips': 0.0,
            'ssim': 0.0,
            'baseline_ssim': 0.0,
            'model_type_for_metrics': 'final_model'
        }

def log_wandb_results(results_dir, args):
    """
    Log final visualizations to Weights & Biases.
    """
    if not args.wandb:
        return
        
    # Log the final comparison image
    wandb_logs = {
        "final_comparison": wandb.Image(str(results_dir / 'comparison.png')),
        "final_translation_vis": wandb.Image(str(results_dir / 'final_translation_vis.png')),
        "final_training_curves": wandb.Image(str(results_dir / 'final_training_curves.png')),
        "shift_evolution": wandb.Image(str(results_dir / 'shift_evolution.png')),
    }
    
    # Add animation logs only if full visualization is enabled
    if args.use_full_visualization:
        animation_logs = {
            "shift_evolution_animation": wandb.Video(str(results_dir / 'shift_evolution_animation.gif')),
        }
        wandb_logs.update(animation_logs)
    
    # If we have mask visualization
    mask_vis_path = results_dir / 'final_mask_vis' / 'masked_images_iter_final.png'
    if mask_vis_path.exists():
        wandb_logs["final_mask_vis"] = wandb.Image(str(mask_vis_path))
    
    wandb.log(wandb_logs)
    wandb.finish()

def save_metrics(train_losses, final_metrics, history, sample_id, results_dir, args):
    """
    Save metrics to files in various formats.
    """
    # Create metrics dictionary
    final_metrics_dict = {
        'sample_id': sample_id,  # Make sure each metrics file knows which sample it's for
        'downsampling_factor': args.df,
        'lr_shift': args.lr_shift,
        'num_samples': args.num_samples,
        'model': args.model,
        'projection': args.input_projection,
        'iterations': args.iters,
        'crop_margin': args.crop_margin,  # Add the crop margin parameter
        
        # Add model architecture parameters
        'network_depth': args.network_depth,
        'network_hidden_dim': args.network_hidden_dim,
        'projection_dim': args.projection_dim,
        'use_gnll': args.use_gnll,
        'use_dual_optimizer': args.use_dual_optimizer,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        
        # Add specialized parameters for specific projection types
        'fourier_scale': getattr(args, 'fourier_scale', None),
        'legendre_max_degree': getattr(args, 'legendre_max_degree', None),
        'rotation': getattr(args, 'rotation', False),
        
        # Loss metrics - these are now the MSE metrics for consistent comparison
        'final_recon_loss_mse': train_losses['monitor_recon_loss'],
        'final_trans_loss': train_losses['trans_loss'],
        'final_total_loss_mse': train_losses['monitor_recon_loss'] + train_losses['trans_loss'],
        'final_recon_loss_keepout_mse': train_losses['monitor_recon_loss_keepout'],
        'final_trans_loss_keepout': train_losses['trans_loss_keepout'],
        'final_total_loss_keepout_mse': train_losses['monitor_recon_loss_keepout'] + train_losses['trans_loss_keepout'],
        'final_test_loss': history['test_loss'][-1] if history['test_loss'] else 0,
        
        # Performance metrics
        'final_psnr': final_metrics['psnr'],
        'final_baseline_psnr': final_metrics['psnr_baseline'],
        'psnr_improvement': final_metrics['psnr'] - final_metrics['psnr_baseline'],
    }
    
    # Add original GNLL loss metrics if GNLL was used
    if args.use_gnll and 'recon_loss' in train_losses:
        final_metrics_dict.update({
            'final_recon_loss_gnll': train_losses['recon_loss'],
            'final_recon_loss_keepout_gnll': train_losses['recon_loss_keepout'],
            'final_total_loss_gnll': train_losses['recon_loss'] + train_losses['trans_loss'],
            'final_total_loss_keepout_gnll': train_losses['recon_loss_keepout'] + train_losses['trans_loss_keepout'],
        })
        
        # If we have original validation metrics
        if 'original_recon_loss_keepout_val' in history and history['original_recon_loss_keepout_val'] and len(history['original_recon_loss_keepout_val']) > 0:
            final_metrics_dict['final_recon_loss_keepout_val_gnll'] = history['original_recon_loss_keepout_val'][-1]
            final_metrics_dict['final_total_loss_keepout_val_gnll'] = history['original_recon_loss_keepout_val'][-1] + history['trans_loss_keepout_val'][-1]
            
    # Add validation metrics
    if 'recon_loss_keepout_val' in history and history['recon_loss_keepout_val'] and len(history['recon_loss_keepout_val']) > 0:
        final_metrics_dict.update({
            'final_recon_loss_keepout_val_mse': history['recon_loss_keepout_val'][-1],
            'final_trans_loss_keepout_val': history['trans_loss_keepout_val'][-1],
            'final_total_loss_keepout_val_mse': history['total_loss_keepout_val'][-1],
        })
    
    # Add LPIPS metrics if available
    if 'lpips' in final_metrics:    
        final_metrics_dict.update({
            'final_lpips': final_metrics['lpips'],
            'final_baseline_lpips': final_metrics['baseline_lpips'],
            'lpips_improvement': final_metrics['baseline_lpips'] - final_metrics['lpips'],
        })
    
    # Add SSIM metrics if available
    if 'ssim' in final_metrics:
        final_metrics_dict.update({
            'final_ssim': final_metrics['ssim'],
            'final_baseline_ssim': final_metrics['baseline_ssim'],
            'ssim_improvement': final_metrics['ssim'] - final_metrics['baseline_ssim'],
        })
    
    # Add model type information
    final_metrics_dict['metrics_from_model'] = 'final_model'

    # Save metrics to JSON
    with open(results_dir / 'metrics.json', 'w') as f:
        json.dump(final_metrics_dict, f, indent=2)

    # Save metrics to CSV
    metrics_df = pd.DataFrame([final_metrics_dict])
    metrics_df.to_csv(results_dir / 'metrics.csv', index=False)
    
    return final_metrics_dict

def main():
    # Parse arguments and initialize
    args = parse_arguments()
    
    # Example usage for RGGB mode:
    # python main.py --dataset burst_synth --sample_id 0 --keep_rggb
    #
    # The keep_rggb flag enables the following features:
    # 1. Preserves separate G1 and G2 channels instead of merging them
    # 2. Creates 4-channel RGGB GT images from RGB ground truth
    # 3. Saves visualizations of individual RGGB channels in the results/rggb_channels directory
    # 4. Computes channel-wise differences between prediction and ground truth
    # 5. Shows which channels contribute most to reconstruction error
    
    # Check if we're only testing aggregation
    if args.test_aggregate_only:
        print("Running aggregation test only...")
        base_results_dir = Path("results")
        dataset_results = aggregate_results(base_results_dir)
        
        # Print summary of aggregated results
        if dataset_results:
            print("\nAggregated results by dataset:")
            for dataset_name, df in dataset_results.items():
                print(f"\n{'-'*40}")
                print(f"Dataset: {dataset_name}")
                print(f"Number of model runs: {len(df)}")
                
                # List available metrics
                metric_columns = [col for col in df.columns if col.startswith('final_') or col.endswith('_improvement')]
                print(f"Available metrics: {', '.join(metric_columns)}")
                
                # Show path to generated files
                dataset_dir = base_results_dir / dataset_name / 'aggregated'
                print(f"\nResults saved to: {dataset_dir}")
                print("Generated files:")
                print(f"- {dataset_dir}/aggregated_results.csv: Full raw results")
                print(f"- {dataset_dir}/model_comparison_detailed.csv: Detailed summary with all model parameters")
                print(f"- {dataset_dir}/model_comparison_formatted.csv: Basic summary with fewer columns")
                print(f"- {dataset_dir}/model_type_comparison.csv: High-level summary by model type")
                print(f"- {dataset_dir}/model_comparison_lr_*.csv: Separate summaries by learning rate")
                
                # Check if LPIPS and SSIM metrics are included
                has_lpips = 'final_lpips' in df.columns
                has_ssim = 'final_ssim' in df.columns
                
                metrics_included = ["PSNR and its improvement"]
                if has_lpips:
                    metrics_included.append("LPIPS and its improvement")
                if has_ssim:
                    metrics_included.append("SSIM and its improvement")
                
                print(f"\nMetrics included in summaries: {', '.join(metrics_included)}")
            
            # Mention combined file
            print(f"\n{'-'*40}")
            print(f"Combined results from all datasets saved to: {base_results_dir}/all_datasets_aggregated_results.csv")
            print("Note: For meaningful comparisons, prefer the dataset-specific result files.")
            
        else:
            print("No results found to aggregate.")
        return
    
    # Continue with normal execution if not just testing aggregation
    device = setup_device(args)
    setup_seed(args.seed)
    
    # Create base results directory
    Path('results').mkdir(exist_ok=True)
    
    # Get list of samples to process
    if args.dataset == "satburst_synth":
        samples = [x.stem for x in pathlib.Path("data").glob("*")]
    else:
        samples = [x.stem for x in pathlib.Path("SyntheticBurstVal/bursts").glob("*")]
    
    # Process selected samples
    for sample_id in samples[:]:  # Process the first 20 samples
        print(f"Processing sample: {sample_id}")
        
        # Setup directories
        results_dir = create_result_directories(args, sample_id)
        print(f"Saving results to: {results_dir}")
        
        # Handle fourier scale parameter
        if args.input_projection.startswith("fourier_"):
            args.fourier_scale = float(args.input_projection.split("_")[1])
            args.input_projection = "fourier"
        
        # Save configuration
        save_config(args, results_dir, sample_id)
        
        # Load dataset
        train_data, train_dataloader = load_dataset(
            args, sample_id, args.df, args.lr_shift
        )
        
        # Pass scale factor to dataset if it's a SyntheticBurstVal dataset
        if args.dataset == 'burst_synth' and hasattr(train_data, 'scale_factor'):
            # Update scale factor if it's not already set
            if train_data.scale_factor is None and args.df is not None:
                print(f"Setting dataset scale factor to {args.df}")
                train_data.scale_factor = args.df
                # Reload the GT image with the new scale factor
                train_data.gt_image = train_data._read_gt_image()
                # Update HR coordinates
                h, w = train_data.gt_image.shape[1:] if train_data.gt_image.dim() == 3 else train_data.gt_image.shape
                coords_h = np.linspace(0, 1, h, endpoint=False)
                coords_w = np.linspace(0, 1, w, endpoint=False)
                coords = np.stack(np.meshgrid(coords_h, coords_w), -1)
                train_data.hr_coords = torch.FloatTensor(coords).to(device)
        
        # Initialize model
        model, optimizer, shift_optimizer, scheduler, shift_scheduler = initialize_model(
            args, device, args.num_samples
        )
        
        # Initialize history
        history = initialize_history()
        
        # Setup wandb logging
        setup_wandb(args, sample_id, args.df, args.lr_shift, args.num_samples, args.learning_rate)
        
        # Run training loop
        history, train_losses, _, _, _, _, _ = run_training_loop(
            model, optimizer, shift_optimizer, scheduler, shift_scheduler,
            train_data, train_dataloader, device, args, history, results_dir
        )
        
        # Generate visualizations
        final_metrics = visualize_results(
            model, train_data, history, results_dir, device, args
        )
        
        # Log results to wandb
        log_wandb_results(results_dir, args)
        
        # Save metrics to disk
        save_metrics(train_losses, final_metrics, history, sample_id, results_dir, args)
    
    
    # Aggregate results for the current dataset
    base_results_dir = Path("results")
    dataset_results = aggregate_results(base_results_dir)
    
    # Print summary of aggregated results
    if dataset_results:
        dataset_name = args.dataset
        if dataset_name in dataset_results:
            df = dataset_results[dataset_name]
            dataset_dir = base_results_dir / dataset_name / 'aggregated'
            
            print(f"\n{'-'*40}")
            print(f"Aggregated results for dataset: {dataset_name}")
            print(f"Number of model runs: {len(df)}")
            print(f"\nResults saved to: {dataset_dir}")
            print("Generated files:")
            print(f"- {dataset_dir}/aggregated_results.csv: Full raw results")
            print(f"- {dataset_dir}/model_comparison_detailed.csv: Detailed summary with all model parameters")
            print(f"- {dataset_dir}/model_comparison_formatted.csv: Basic summary with fewer columns")
            print(f"- {dataset_dir}/model_type_comparison.csv: High-level summary by model type")
            print(f"- {dataset_dir}/model_comparison_lr_*.csv: Separate summaries by learning rate")
            
            # Check if LPIPS and SSIM metrics are included
            has_lpips = 'final_lpips' in df.columns
            has_ssim = 'final_ssim' in df.columns
            
            metrics_included = ["PSNR and its improvement"]
            if has_lpips:
                metrics_included.append("LPIPS and its improvement")
            if has_ssim:
                metrics_included.append("SSIM and its improvement")
            
            print(f"\nMetrics included in summaries: {', '.join(metrics_included)}")
        else:
            print(f"No aggregated results available for dataset: {args.dataset}")
    else:
        print("No results found to aggregate.")

def parse_arguments():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Satellite Super-Resolution Training")
    
    # Dataset parameters
    dataset_group = parser.add_argument_group('Dataset Parameters')
    dataset_group.add_argument("--dataset", type=str, default="satburst_synth",
                        help="Dataset implemented in data.get_dataset()")
    dataset_group.add_argument("--root_burst_synth", default="./SyntheticBurstVal", help="Set root of burst_synth")
    dataset_group.add_argument("--root_satburst_synth", default="~/data/satburst_synth", help="Set root of satburst_synth")
    dataset_group.add_argument("--root_worldstrat", default="~/data/worldstrat_kaggle", help="Set root of worldstrat dataset")
    dataset_group.add_argument("--area_name", type=str, default="UNHCR-LBNs006446", help="str: a sample name of worldstrat dataset")
    dataset_group.add_argument("--worldstrat_hr_size", type=int, default=None, help="int: Default size is 1054")
    dataset_group.add_argument("--sample_id", default="Landcover-743192_rgb", help="str: a sample index of burst_synth")
    dataset_group.add_argument("--df", type=int, default=4, help="Downsampling factor")
    dataset_group.add_argument("--lr_shift", type=float, default=1.0, help="Low-resolution shift amount")
    dataset_group.add_argument("--num_samples", type=int, default=16, help="Number of samples to use")
    dataset_group.add_argument("--use_gt", type=bool, default=False, help="Whether to use ground truth shifts")
    dataset_group.add_argument("--keep_rggb", action="store_true", help="Keep original RGGB channels without merging G1 and G2")
    dataset_group.add_argument("--aug", type=str, default="none", 
                       choices=['none', 'light', 'medium', 'heavy'],
                       help="Augmentation level to use")
    dataset_group.add_argument("--rotation", type=bool, default=False, help="Whether to use rotation augmentation")
    dataset_group.add_argument("--grayscale", action="store_true", help="Process images as grayscale")
    dataset_group.add_argument("--crop_margin", type=int, default=16, 
                       help="Number of pixels to crop from image borders when calculating metrics to avoid edge artifacts")
    
    # Model parameters
    model_group = parser.add_argument_group('Model Parameters')
    model_group.add_argument("--model", type=str, default="mlp", choices=["mlp", "siren", "wire", "linear", "conv", "thera"],
                      help="Type of model to use")
    model_group.add_argument("--network_depth", type=int, default=4, help="Depth of the network")
    model_group.add_argument("--network_hidden_dim", type=int, default=256, help="Hidden dimension of the network")
    model_group.add_argument("--projection_dim", type=int, default=256, help="Dimension of the projection")
    model_group.add_argument("--output_dim", type=int, default=3, help="Output dimension of the network")
    model_group.add_argument("--sigmoid_output", type=bool, default=False, help="Use sigmoid output for the network")
    model_group.add_argument("--use_gnll", type=bool, default=False, help="Use Gaussian NLL loss")
    model_group.add_argument("--disable_shifts", type=bool, default=False, help="Disable shifts")
    model_group.add_argument("--disable_frame_decoder", type=bool, default=False, help="Disable frame decoder")
    
    # Input projection parameters
    projection_group = parser.add_argument_group('Input Projection Parameters')
    projection_group.add_argument("--input_projection", type=str, default="fourier_10", help="Input projection to use")
    projection_group.add_argument("--fourier_scale", type=float, default=10.0, 
                           help="Fourier scale for the input projection")
    projection_group.add_argument("--legendre_max_degree", type=int, default=150, 
                           help="Maximum degree of Legendre polynomial for the input projection")
    projection_group.add_argument("--activation", type=nn.Module, default=F.relu) 
                                  
    # Training parameters
    training_group = parser.add_argument_group('Training Parameters')
    training_group.add_argument("--seed", type=int, default=10, help="Random seed for reproducibility")
    training_group.add_argument("--iters", type=int, default=2000, help="Number of training iterations")
    training_group.add_argument("--bs", type=int, default=1, help="Batch size")
    training_group.add_argument("--learning_rate", type=float, default=2e-3, help="Learning rate for the optimizer")
    training_group.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for AdamW")
    
    # Utility parameters
    utility_group = parser.add_argument_group('Utility Parameters')
    utility_group.add_argument("--d", type=str, default="0", help="CUDA device number")
    utility_group.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    utility_group.add_argument("--use_full_visualization", action="store_true", 
                              help="Enable full visualization features including image evolution animation")
    utility_group.add_argument("--use_dual_optimizer", action="store_true", 
                              help="Use dual optimizer for shift and reconstruction")
    utility_group.add_argument("--test-aggregate-only", action="store_true",
                              help="Only run the aggregate_results function for testing")
    
    return parser.parse_args()

if __name__ == "__main__":
    main()