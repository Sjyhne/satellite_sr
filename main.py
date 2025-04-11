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
from torch.optim.lr_scheduler import CosineAnnealingLR  # Change import
from torchmetrics.functional import structural_similarity_index_measure as ssim
from torch.utils.data import DataLoader
from data import get_dataset
from utils import apply_shift_torch, bilinear_resize_torch, align_output_to_target, get_valid_mask
from coordinate_based_mlp import FourierNetwork
from losses import BasicLosses, AdvancedLosses, CharbonnierLoss, RelativeLosses, GradientDifferenceLoss, PerceptualLoss
from viz_utils import (
    plot_training_curves, visualize_translations, create_model_comparison_grid,
    visualize_model_comparisons, visualize_per_sample_metrics, visualize_psnr_improvement_heatmap,
    visualize_metrics_correlation, visualize_improvement_across_samples,
    visualize_parallel_coordinates, visualize_metric_rankings, visualize_performance_distribution
)

from models.utils import get_decoder
from input_projections.utils import get_input_projection
from models.inr import INR


import einops
import json
import argparse
import pathlib
import os
import lpips
import seaborn as sns

def train_one_iteration(model, optimizer, train_sample, device, iteration=0, use_gt=False):
    model.train()

    # Initialize loss functions
    recon_criterion = BasicLosses.mse_loss
    trans_criterion = BasicLosses.mae_loss

    if model.use_gnll:
        recon_criterion = nn.GaussianNLLLoss()


    input = train_sample['input'].to(device)
    lr_target = train_sample['lr_target'].to(device)
    sample_id = train_sample['sample_id'].to(device)
    lr_mean = train_sample['mean'].to(device)
    lr_std = train_sample['std'].to(device)


    if 'shifts' in train_sample and 'dx_percent' in train_sample['shifts']:
        gt_dx = train_sample['shifts']['dx_percent'].to(device)
        gt_dy = train_sample['shifts']['dy_percent'].to(device)
    else:
        gt_dx = torch.zeros(lr_target.shape[0], device=device)
        gt_dy = torch.zeros(lr_target.shape[0], device=device)

    optimizer.zero_grad()

    if model.use_gnll:
        output, pred_shifts, pred_variance = model(input, sample_id, scale_factor=0.25)
        # pred_variance = bilinear_resize_torch(pred_variance.permute(0, 3, 1, 2), (lr_target.shape[1], lr_target.shape[2])).permute(0, 2, 3, 1)
    else:
        output, pred_shifts = model(input, sample_id, scale_factor=0.25)

    # Downsample to match target resolution
    # output = bilinear_resize_torch(output.permute(0, 3, 1, 2), (lr_target.shape[1], lr_target.shape[2])).permute(0, 2, 3, 1)

    if model.use_gnll:
        recon_loss = recon_criterion(output, lr_target, pred_variance)
    else:
        recon_loss = recon_criterion(output, lr_target)

    pred_dx, pred_dy = pred_shifts
    trans_loss = trans_criterion(pred_dx, gt_dx) + trans_criterion(pred_dy, gt_dy)

    # Only backpropagate the reconstruction loss
    recon_loss.backward()
    optimizer.step()
    
    return {
        'recon_loss': recon_loss.item(),
        'trans_loss': trans_loss.item(),
        'total_loss': recon_loss.item() + trans_loss.item(),
        'pred_dx': pred_dx.detach().cpu().numpy(),
        'pred_dy': pred_dy.detach().cpu().numpy(),
        'gt_dx': gt_dx.detach().cpu().numpy(),
        'gt_dy': gt_dy.detach().cpu().numpy()
    }

def test_one_epoch(model, test_loader, device):
    model.eval()

    # Get HR features from the test loader
    hr_coords = test_loader.get_hr_coordinates().unsqueeze(0).to(device)
    hr_image = test_loader.get_original_hr().unsqueeze(0).to(device)
    sample_id = torch.tensor([0]).to(device)
    
    if model.use_gnll:
        output, _, _ = model(hr_coords, sample_id, scale_factor=1, training=False)
    else:
        output, _ = model(hr_coords, sample_id, scale_factor=1, training=False)

    # output is centered around 0, so we need to unstandardize it
    # We then use the mean and std of the sample 0 which we use as "fixed"
    output = output * test_loader.get_lr_std(0).cuda() + test_loader.get_lr_mean(0).cuda()

    loss = F.mse_loss(output, hr_image)
    
    return loss.item(), output.detach(), hr_image.detach()



def calculate_metrics(pred, target):
    """Calculate multiple image quality metrics.
    
    Args:
        pred: Predicted image tensor [B, C, H, W]
        target: Target image tensor [B, C, H, W]
        
    Returns:
        Dictionary of metrics
    """
    # Ensure inputs are in correct format
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
    if target.dim() == 3:
        target = target.unsqueeze(0)
    
    # Move to same device as inputs
    loss_fn_alex = lpips.LPIPS(net='alex').to(pred.device)
    
    # Calculate metrics
    mse = F.mse_loss(pred, target)
    psnr = -10 * torch.log10(mse)
    lpips_value = loss_fn_alex(pred, target).mean()
    ssim_value = ssim(pred, target)
    
    return {
        'mse': mse.item(),
        'psnr': psnr.item(),
        'lpips': lpips_value.item(),
        'ssim': ssim_value.item()
    }

def aggregate_results(base_dir):
    """
    Aggregate results across all samples for each model type and configuration.
    
    Args:
        base_dir: Path to the base results directory
    
    Returns:
        DataFrame with aggregated results
    """
    results = []
    
    # Find all dataset directories
    dataset_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
    
    for dataset_dir in dataset_dirs:
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
                        
                        # Add sample and model info if not already in metrics
                        if 'sample_id' not in metrics:
                            metrics['sample_id'] = sample_dir.name
                        metrics['dataset'] = dataset_dir.name
                        metrics['experiment'] = experiment_dir.name
                        
                        # Parse model and projection from directory name if not already in metrics
                        if 'model_type' not in metrics:
                            model_parts = model_dir.name.split('_')
                            metrics['model_type'] = model_parts[0]
                            
                            # Handle projection type
                            if len(model_parts) >= 2:
                                # For fourier, keep the scale as part of the projection name
                                if "fourier" in model_parts[1]:
                                    if len(model_parts) >= 3 and model_parts[2].replace('.', '', 1).isdigit():
                                        metrics['projection_type'] = f"{model_parts[1]}_{model_parts[2]}"
                                    else:
                                        metrics['projection_type'] = model_parts[1]
                                else:
                                    metrics['projection_type'] = model_parts[1]
                            else:
                                metrics['projection_type'] = 'unknown'
                        
                        results.append(metrics)
    
    # Convert to DataFrame
    if results:
        df = pd.DataFrame(results)
        return df
    else:
        return pd.DataFrame()

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description="Satellite Super-Resolution Training")
    
    # Dataset parameters
    dataset_group = parser.add_argument_group('Dataset Parameters')
    dataset_group.add_argument("--dataset", type=str, default="satburst_synth",
                        choices=["satburst_synth", "worldstrat", "burst_synth"],
                        help="Dataset implemented in data.get_dataset()")
    dataset_group.add_argument("--root_burst_synth", default="./SyntheticBurstVal", help="Set root of burst_synth")
    dataset_group.add_argument("--root_satburst_synth", default="~/data/satburst_synth", help="Set root of worldstrat dataset")
    dataset_group.add_argument("--root_worldstrat", default="~/data/worldstrat_kaggle", help="Set root of worldstrat dataset")
    dataset_group.add_argument("--area_name", type=str, default="UNHCR-LBNs006446", help="str: a sample name of worldstrat dataset")
    dataset_group.add_argument("--worldstrat_hr_size", type=int, default=None, help="int: Default size is 1054")
    dataset_group.add_argument("--sample_id", default="Landcover-743192_rgb", help="str: a sample index of burst_synth")
    dataset_group.add_argument("--df", type=int, default=4, help="Downsampling factor")
    dataset_group.add_argument("--lr_shift", type=float, default=1.0, help="Low-resolution shift amount")
    dataset_group.add_argument("--num_samples", type=int, default=16, help="Number of samples to use")
    dataset_group.add_argument("--use_gt", type=bool, default=False, help="Whether to use ground truth shifts")
    dataset_group.add_argument("--aug", type=str, default="none", 
                       choices=['none', 'light', 'medium', 'heavy'],
                       help="Augmentation level to use")
    dataset_group.add_argument("--rotation", type=bool, default=False, help="Whether to use rotation augmentation")
    
    # Model parameters
    model_group = parser.add_argument_group('Model Parameters')
    model_group.add_argument("--model", type=str, default="mlp", choices=["mlp", "siren", "wire", "linear", "conv", "thera"],
                      help="Type of model to use")
    model_group.add_argument("--network_depth", type=int, default=10, help="Depth of the network")
    model_group.add_argument("--network_hidden_dim", type=int, default=256, help="Hidden dimension of the network")
    model_group.add_argument("--projection_dim", type=int, default=256, help="Dimension of the projection")
    model_group.add_argument("--output_dim", type=int, default=3, help="Output dimension of the network")
    model_group.add_argument("--sigmoid_output", type=bool, default=False, help="Use sigmoid output for the network")
    model_group.add_argument("--use_gnll", type=bool, default=False, help="Use Gaussian NLL loss")
    
    # Input projection parameters
    projection_group = parser.add_argument_group('Input Projection Parameters')
    projection_group.add_argument("--input_projection", type=str, default="fourier_10", 
                           choices=["linear", "fourier_10", "fourier_5", "fourier_20", "fourier_40", "fourier", "legendre", "none"], 
                           help="Input projection to use")
    projection_group.add_argument("--fourier_scale", type=float, default=10.0, 
                           help="Fourier scale for the input projection")
    projection_group.add_argument("--legendre_max_degree", type=int, default=150, 
                           help="Maximum degree of Legendre polynomial for the input projection")
    projection_group.add_argument("--activation", type=nn.Module, default=F.relu) 
                                  
    # Training parameters
    training_group = parser.add_argument_group('Training Parameters')
    training_group.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    training_group.add_argument("--iters", type=int, default=1000, help="Number of training iterations")
    training_group.add_argument("--bs", type=int, default=1, help="Batch size")
    training_group.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for the optimizer")
    training_group.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for AdamW")
    
    # Utility parameters
    utility_group = parser.add_argument_group('Utility Parameters')
    utility_group.add_argument("--d", type=str, default="2", help="CUDA device number")
    utility_group.add_argument("--wandb", action="store_true", help="Enable wandb logging")

    args = parser.parse_args()

    # set torch cuda device
    if torch.cuda.is_available():
        torch.cuda.set_device(f"cuda:{args.d}")  
    device = torch.device(f"cuda:{args.d}" if torch.cuda.is_available() else "cpu")
    
    # Create base results directory
    Path('results').mkdir(exist_ok=True)

    # Set all seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # For completely reproducible results
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Network specific parameters

    network_depth = args.network_depth
    network_hidden_dim = args.network_hidden_dim
    projection_dim = args.projection_dim
    learning_rate = args.learning_rate
    iters = args.iters


    # Dataset specific parameters
    downsample_factor = args.df
    lr_shift = args.lr_shift
    num_samples = args.num_samples

    if args.dataset == "satburst_synth":
        sample_id = args.sample_id
    elif args.dataset == "burst_synth":
        sample_id = f"sample_{args.sample_id}"
    elif args.dataset == "worldstrat":
        sample_id = args.area_name
    else:
        sample_id = "unknown_sample"

    

    # Create a more organized results directory structure
    # Base results directory
    base_results_dir = Path("results")
    base_results_dir.mkdir(exist_ok=True)

    samples = [x.stem for x in pathlib.Path("data").glob("*")]
    
    # First level: dataset name
    dataset_dir = base_results_dir / args.dataset
    
    # Third level: key experiment parameters
    experiment_name = f"df{downsample_factor}_shift{lr_shift:.1f}_samples{num_samples}"
    if args.aug != "none":
        experiment_name += f"_aug{args.aug}"
    
    # Combine all parts (removed timestamp)
    results_dir = dataset_dir / experiment_name
    results_dir.mkdir(parents=True, exist_ok=True)

    for sample_id in samples[4:5]:
        # Store the original input projection for directory naming
        dir_input_projection = args.input_projection
        
        # If using fourier without explicit scale in the name, add it for the directory name
        if not "_" in dir_input_projection and dir_input_projection.startswith("fourier"):
            dir_input_projection = dir_input_projection + "_" + str(args.fourier_scale)

        # Create directory using consistent naming
        results_dir = dataset_dir / experiment_name / sample_id / f"{args.model}_{dir_input_projection}_{args.iters}{'_gnll' if args.use_gnll else ''}"
        results_dir.mkdir(parents=True, exist_ok=True)

        # Update fourier_scale parameter if needed
        if args.input_projection.startswith("fourier_"):
            args.fourier_scale = float(args.input_projection.split("_")[1])
            args.input_projection = "fourier"

        print(f"Saving results to: {results_dir}")

        # No need to process the input projection name here anymore
        # The shell script already sets the correct fourier_scale parameter

        args.sample_id = sample_id
    
        # Save detailed configuration to a JSON file
        config = {
            # Dataset parameters
            "dataset": args.dataset,
            "sample_id": sample_id,
            "downsampling_factor": downsample_factor,
            "lr_shift": lr_shift,
            "num_samples": num_samples,
            "augmentation": args.aug,
            
            # Model parameters
            "model": args.model,
            "network_depth": network_depth,
            "network_hidden_dim": network_hidden_dim,
            "projection_dim": projection_dim,
            "rotation": args.rotation,
            
            # Training parameters
            "iterations": iters,
            "learning_rate": learning_rate,
            "weight_decay": args.weight_decay,
            "batch_size": args.bs,
            "seed": args.seed,
            "use_gt": args.use_gt
        }
        
        with open(results_dir / "experiment_config.json", "w") as f:
            json.dump(config, f, indent=2)

        # Load the dataset
        if args.dataset == "satburst_synth":
            args.root_satburst_synth = f"data/{args.sample_id}/scale_{downsample_factor}_shift_{lr_shift:.1f}px_aug_{args.aug}"

        train_data = get_dataset(args=args, name=args.dataset)

        batch_size = args.bs

        # initialize the dataloader here
        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

        input_projection = get_input_projection(args.input_projection, 2, args.projection_dim, device, args.fourier_scale, args.legendre_max_degree, args.activation)

        if args.input_projection == "legendre":
            original_dim = args.projection_dim
            args.projection_dim = input_projection.get_output_dim()
            print(f"Legendre projection: changing projection_dim from {original_dim} to {args.projection_dim}")

        decoder = get_decoder(args.model, args.network_depth, args.projection_dim, args.network_hidden_dim)
        
        model = INR(input_projection, decoder, num_samples, use_gnll=args.use_gnll).to(device)

        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=args.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=iters, eta_min=1e-6)

        # Initialize history dictionary with new metrics
        history = {
            'iterations': [],
            'recon_loss': [],
            'trans_loss': [],
            'test_loss': [],
            'psnr': [],
            'lpips': [],
            'ssim': [],
            'baseline_psnr': [],
            'baseline_lpips': [],
            'baseline_ssim': [],
            'translation_data': [],
            'learning_rate': []  # Changed from recon_lr and trans_lr to single learning_rate
        }

        # Initialize wandb if enabled
        if args.wandb:
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
                    "iters": iters,
                    "network_hidden_dim": network_hidden_dim,
                    "network_depth": network_depth,
                    "learning_rate": learning_rate,
                    "use_gt": args.use_gt,
                    "augmentation": args.aug
                }
            )
        
        import lpips

        # Initialize LPIPS model once
        loss_fn_alex = lpips.LPIPS(net='alex').to(device)

        def calculate_metrics(pred, target):
            """Calculate multiple image quality metrics."""
            # Ensure inputs are in correct format
            if pred.dim() == 3:
                pred = pred.unsqueeze(0)
            if target.dim() == 3:
                target = target.unsqueeze(0)
            
            # Calculate metrics using the shared LPIPS model
            mse = F.mse_loss(pred, target)
            psnr = -10 * torch.log10(mse)
            lpips_value = loss_fn_alex(pred, target).mean()
            ssim_value = ssim(pred, target)
            
            return {
                'mse': mse.item(),
                'psnr': psnr.item(),
                'lpips': lpips_value.item(),
                'ssim': ssim_value.item()
            }

        i = 0
        aggregate_train_losses = {}
        
        # Create a tqdm progress bar for the entire training
        progress_bar = tqdm(total=iters, desc="Training")
        
        while i < iters:
            for train_sample in train_dataloader:
                train_losses = train_one_iteration(model, optimizer, train_sample, device, iteration=i+1, use_gt=args.use_gt)
                scheduler.step()
                i += 1

                for key, value in train_losses.items():
                    if key not in aggregate_train_losses:
                        aggregate_train_losses[key] = []
                    aggregate_train_losses[key].append(value)
                
                # Update the progress bar
                progress_bar.update(1)
                progress_bar.set_postfix({
                    'recon_loss': f"{train_losses['recon_loss']:.4f}",
                    'trans_loss': f"{train_losses['trans_loss']:.4f}"
                })
                
                
                
                if (i + 1) % 100 == 0:  # More frequent logging
                    with torch.no_grad():
                        test_loss, test_output, hr_image = test_one_epoch(model, train_data, device)

                        # Unstandardize the output
                        test_output = einops.rearrange(test_output, 'b h w c -> b c h w')
                        hr_image = einops.rearrange(hr_image, 'b h w c -> b c h w')
                        lr_sample = train_data.get_lr_sample(0).unsqueeze(0).permute(0, 3, 1, 2).to(device)
                        baseline_pred = bilinear_resize_torch(lr_sample, (hr_image.shape[2], hr_image.shape[3]))

                        # align predictions and targets spectrally (no spatial alignment while optimizing)
                        test_output = align_output_to_target(input=test_output, reference=hr_image, spatial=False)
                        baseline_pred = align_output_to_target(input=baseline_pred, reference=hr_image, spatial=False)

                        # Calculate metrics for model output
                        baseline_metrics = calculate_metrics(baseline_pred, hr_image)
                        model_metrics = calculate_metrics(test_output, hr_image)
                        


                    
                    # Store values in history
                    history['iterations'].append(i + 1)
                    history['recon_loss'].append(aggregate_train_losses['recon_loss'][-1])
                    history['trans_loss'].append(aggregate_train_losses['trans_loss'][-1])
                    history['test_loss'].append(test_loss)
                    history['psnr'].append(model_metrics['psnr'])
                    history['lpips'].append(model_metrics['lpips'])
                    history['ssim'].append(model_metrics['ssim'])
                    history['baseline_psnr'].append(baseline_metrics['psnr'])
                    history['baseline_lpips'].append(baseline_metrics['lpips'])
                    history['baseline_ssim'].append(baseline_metrics['ssim'])

                    translation_data = []
                    for idx in range(len(train_data)):
                        gt_shift = train_data[idx]['shifts']
                        pred_shift = model.shift_vectors[idx]

                        gt_dx = gt_shift["dx_percent"]
                        gt_dy = gt_shift["dy_percent"]

                        pred_dy = pred_shift[1]
                        pred_dx = pred_shift[0]

                        translation_data.append({
                            'target_dx': gt_dx,
                            'target_dy': gt_dy,
                            'pred_dx': pred_dx.detach().cpu().numpy(),
                            'pred_dy': pred_dy.detach().cpu().numpy(),
                        })

                    # Store all translations for this iteration
                    history['translation_data'].append(translation_data)

                    # Store learning rates in history
                    history['learning_rate'].append(scheduler.get_last_lr()[0])

                    # Print all metrics
                    print(f"Iter {i+1}: "
                        f"Train recon: {aggregate_train_losses['recon_loss'][-1]:.6f}, "
                        f"trans: {aggregate_train_losses['trans_loss'][-1]:.6f}, "
                        f"total: {aggregate_train_losses['total_loss'][-1]:.6f}, "
                        f"Test: {test_loss:.6f}\n"
                        f"Metrics vs Baseline:\n"
                        f"PSNR: {model_metrics['psnr']:.2f}dB vs {baseline_metrics['psnr']:.2f}dB\n"
                        f"LPIPS: {model_metrics['lpips']:.4f} vs {baseline_metrics['lpips']:.4f} (lower is better)\n"
                        f"SSIM: {model_metrics['ssim']:.4f} vs {baseline_metrics['ssim']:.4f} (higher is better)")

                    if args.wandb:
                        wandb.log({
                            "iteration": i + 1,
                            "train/recon_loss": aggregate_train_losses['recon_loss'][-1],
                            "train/trans_loss": aggregate_train_losses['trans_loss'][-1],
                            "train/total_loss": aggregate_train_losses['total_loss'][-1],
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
                        })

                    # Update progress bar description with current metrics
                    progress_bar.set_description(
                        f"Train: {aggregate_train_losses['recon_loss'][-1]:.4f} | "
                        f"PSNR: {model_metrics['psnr']:.2f}dB"
                    )
                    
                    aggregate_train_losses = {}

                # Break if we've reached the desired number of iterations
                if i >= iters:
                    break
        
        # Close the progress bar
        progress_bar.close()

        # Create all visualizations at the end
        # Training curves
        plot_training_curves(history, save_path=results_dir / 'final_training_curves.png')
        
        # Translation visualization (using last iteration's data)
        last_trans_data = history['translation_data'][-1]  # Get the last iteration's data

        # Extract all translation data
        pred_dx_list = []
        pred_dy_list = []
        target_dx_list = []
        target_dy_list = []

        for sample_data in last_trans_data:
            # Extract values and convert to float
            pred_dx_list.append(float(sample_data['pred_dx']))
            pred_dy_list.append(float(sample_data['pred_dy']))
            target_dx_list.append(float(sample_data['target_dx']))
            target_dy_list.append(float(sample_data['target_dy']))

        # Convert to numpy arrays
        pred_dx_array = np.array(pred_dx_list)
        pred_dy_array = np.array(pred_dy_list)
        target_dx_array = np.array(target_dx_list)
        target_dy_array = np.array(target_dy_list)

        # Visualize translations
        visualize_translations(
            pred_dx_array,
            pred_dy_array,
            target_dx_array,
            target_dy_array,
            save_path=results_dir / 'final_translation_vis.png'
        )


        # Create downsampled then upsampled version for comparison
        with torch.no_grad():

            # Final test and visualization
            test_loss, test_output, hr_image = test_one_epoch(model, train_data, device)
            lr_sample = train_data.get_lr_sample(0).unsqueeze(0).permute(0, 3, 1, 2).to(device)

            test_output = einops.rearrange(test_output, 'b h w c -> b c h w')
            hr_image = einops.rearrange(hr_image, 'b h w c -> b c h w')

            baseline_pred = bilinear_resize_torch(lr_sample, (hr_image.shape[2], hr_image.shape[3]))

            # align predictions and targets
            test_output = align_output_to_target(input=test_output, reference=hr_image, spatial=False)
            baseline_pred = align_output_to_target(input=baseline_pred, reference=hr_image, spatial=False)        

            # Calculate PSNR for model output
            valid_mask = get_valid_mask(test_output, hr_image)
            print("valid_mask:", valid_mask.shape)
            mse_model = BasicLosses.mse_loss(test_output, hr_image, mask=valid_mask)
            psnr_model = -10 * torch.log10(mse_model)
            
            # Calculate PSNR for baseline
            valid_mask = get_valid_mask(baseline_pred, hr_image)
            print("valid_mask:", valid_mask.shape)
            mse_baseline = BasicLosses.mse_loss(baseline_pred, hr_image, mask=valid_mask)
            psnr_baseline = -10 * torch.log10(mse_baseline)
        
        # Get LR target image (use sample_00 as it matches the HR ground truth)
        lr_target_img = train_data.get_lr_sample(0).permute(2, 0, 1)  # Get sample_00
        if torch.is_tensor(lr_target_img):
            # First, ensure the image is in the right format
            # If it's [C, H, W], convert to [H, W, C]
            if lr_target_img.shape[0] == 4 or lr_target_img.shape[0] == 3:
                # It's in [C, H, W] format, convert to [H, W, C]
                lr_target_img = lr_target_img.permute(1, 2, 0)
            
            # Now check if it's a RAW Bayer image with 4 channels
            if lr_target_img.shape[2] == 4:
                # Convert RGGB to RGB for visualization
                R = lr_target_img[..., 0]
                G = (lr_target_img[..., 1] + lr_target_img[..., 2]) / 2
                B = lr_target_img[..., 3]
                # Stack to create RGB image
                lr_target_img = torch.stack([R, G, B], dim=-1)
            
            # Convert to numpy for plotting
            lr_target_img = lr_target_img.numpy()
        
        # Convert tensors to numpy arrays for visualization
        hr_image = hr_image.cpu().permute(0, 2, 3, 1).numpy()
        baseline_pred = baseline_pred.cpu().permute(0, 2, 3, 1).numpy()
        test_output = test_output.cpu().permute(0, 2, 3, 1).numpy()

        # Create side by side visualization
        plt.figure(figsize=(24, 6))
        
        plt.subplot(1, 4, 1)
        plt.imshow(hr_image[0])
        plt.title('HR GT')
        plt.axis('off')
        
        plt.subplot(1, 4, 2)
        plt.imshow(lr_target_img)
        plt.title('LR Reference (Sample 00)')
        plt.axis('off')
        
        plt.subplot(1, 4, 3)
        plt.imshow(baseline_pred[0])
        plt.title(f'Bilinear\nPSNR: {psnr_baseline:.2f} dB')
        plt.axis('off')
        
        plt.subplot(1, 4, 4)
        plt.imshow(test_output[0])
        plt.title(f'Pred\nPSNR: {psnr_model:.2f} dB')
        plt.axis('off')
        
        plt.suptitle(f'Test loss: {test_loss:.6f} | Model PSNR: {psnr_model:.2f} dB\nBaseline PSNR: {psnr_baseline:.2f} dB | Dataset scale: {downsample_factor}x\nIterations: {iters} | learning_rate: {scheduler.get_last_lr()[0]}\nNetwork: {args.model} | Hidden dim: {network_hidden_dim} | Depth: {network_depth} | Projection: {args.input_projection}')
        plt.tight_layout()
        plt.savefig(results_dir / 'comparison.png')
        plt.close()

        # Save individual output images
        plt.figure(figsize=(8, 8))
        plt.imshow(test_output[0])
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(results_dir / 'output_prediction.png', bbox_inches='tight', pad_inches=0)
        plt.close()

        # Save HR ground truth
        plt.figure(figsize=(8, 8))
        plt.imshow(hr_image[0])
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(results_dir / 'hr_ground_truth.png', bbox_inches='tight', pad_inches=0)
        plt.close()

        # Save LR input
        plt.figure(figsize=(8, 8))
        plt.imshow(lr_target_img)
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(results_dir / 'lr_input.png', bbox_inches='tight', pad_inches=0)
        plt.close()

        # Save baseline (bilinear upsampling)
        plt.figure(figsize=(8, 8))
        plt.imshow(baseline_pred[0])
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(results_dir / 'baseline_prediction.png', bbox_inches='tight', pad_inches=0)
        plt.close()

        # At the end, log final images
        if args.wandb:
            # Log the final comparison image
            wandb.log({
                "final_comparison": wandb.Image(str(results_dir / 'comparison.png')),
                "final_translation_vis": wandb.Image(str(results_dir / 'final_translation_vis.png')),
                "final_training_curves": wandb.Image(str(results_dir / 'final_training_curves.png')),
            })
            
            # If we have mask visualization
            mask_vis_path = results_dir / 'final_mask_vis' / 'masked_images_iter_final.png'
            if mask_vis_path.exists():
                wandb.log({"final_mask_vis": wandb.Image(str(mask_vis_path))})
            
            wandb.finish()

        # Store final metrics in a dictionary
        final_metrics = {
            'sample_id': sample_id,  # Make sure each metrics file knows which sample it's for
            'downsampling_factor': downsample_factor,
            'lr_shift': lr_shift,
            'num_samples': num_samples,
            'model': args.model,
            'projection': dir_input_projection,  # Use the directory version to maintain consistency
            'iterations': iters,
            'final_recon_loss': train_losses['recon_loss'],
            'final_trans_loss': train_losses['trans_loss'],
            'final_total_loss': train_losses['total_loss'],
            'final_test_loss': test_loss,
            'final_psnr': model_metrics['psnr'],
            'final_lpips': model_metrics['lpips'],
            'final_ssim': model_metrics['ssim'],
            'final_baseline_psnr': baseline_metrics['psnr'],
            'final_baseline_lpips': baseline_metrics['lpips'],
            'final_baseline_ssim': baseline_metrics['ssim'],
            'psnr_improvement': model_metrics['psnr'] - baseline_metrics['psnr'],
            'lpips_improvement': baseline_metrics['lpips'] - model_metrics['lpips'],
            'ssim_improvement': model_metrics['ssim'] - baseline_metrics['ssim'],
        }

        # Save the metrics to a json file
        with open(results_dir / 'metrics.json', 'w') as f:
            json.dump(final_metrics, f)

        # Save metrics to CSV for this experiment
        metrics_df = pd.DataFrame([final_metrics])
        metrics_df.to_csv(results_dir / 'metrics.csv', index=False)

    # After all samples are processed, aggregate results
    base_results_dir = Path("results")
    aggregated_results = aggregate_results(base_results_dir)
    
    if not aggregated_results.empty:
        # Save aggregated results
        aggregated_results.to_csv(base_results_dir / 'aggregated_results.csv', index=False)
        
        # Create visualizations
        visualize_per_sample_metrics(aggregated_results, base_results_dir)
        visualize_model_comparisons(aggregated_results, base_results_dir)
        visualize_psnr_improvement_heatmap(aggregated_results, base_results_dir)
        visualize_metrics_correlation(aggregated_results, base_results_dir)
        visualize_improvement_across_samples(aggregated_results, base_results_dir)
        visualize_parallel_coordinates(aggregated_results, base_results_dir)
        visualize_metric_rankings(aggregated_results, base_results_dir)
        visualize_performance_distribution(aggregated_results, base_results_dir)
        
        # Create a more detailed summary table with per-model metrics
        summary_table = aggregated_results.groupby(['model_type', 'projection_type']).agg({
            'final_psnr': ['mean', 'std', 'min', 'max', 'count'],
            'final_lpips': ['mean', 'std'],
            'final_ssim': ['mean', 'std'],
            'psnr_improvement': ['mean', 'std'],
            'lpips_improvement': ['mean', 'std'],
            'ssim_improvement': ['mean', 'std']
        }).reset_index()

        # Format the table for better readability
        formatted_table = pd.DataFrame({
            'Model': summary_table['model_type'],
            'Projection': summary_table['projection_type'],
            'Samples': summary_table['final_psnr']['count'],
            'PSNR': [f"{m:.2f} ± {s:.2f}" for m, s in zip(summary_table['final_psnr']['mean'], summary_table['final_psnr']['std'])],
            'LPIPS': [f"{m:.4f} ± {s:.4f}" for m, s in zip(summary_table['final_lpips']['mean'], summary_table['final_lpips']['std'])],
            'SSIM': [f"{m:.4f} ± {s:.4f}" for m, s in zip(summary_table['final_ssim']['mean'], summary_table['final_ssim']['std'])],
            'PSNR Improvement': [f"{m:.2f} ± {s:.2f}" for m, s in zip(summary_table['psnr_improvement']['mean'], summary_table['psnr_improvement']['std'])]
        })

        # Sort by mean PSNR improvement
        formatted_table = formatted_table.sort_values('PSNR Improvement', ascending=False)

        # Save as CSV
        formatted_table.to_csv(base_results_dir / 'model_comparison_formatted.csv', index=False)

    # Create model comparison grid
    create_model_comparison_grid(base_results_dir, base_results_dir)

if __name__ == "__main__":
    main()