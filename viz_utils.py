import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import einops
import lpips
import json
import pandas as pd

def plot_training_curves(history, save_path='training_curves.png'):
    """Plot training curves including losses and all metrics."""
    plt.figure(figsize=(15, 20))
    
    # Plot losses
    plt.subplot(5, 1, 1)
    plt.plot(history['iterations'], history['recon_loss'], label='Reconstruction Loss')
    plt.plot(history['iterations'], history['trans_loss'], label='Transform Loss')
    plt.plot(history['iterations'], history['test_loss'], label='Test Loss')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title('Training Losses')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')

    # Plot PSNR
    plt.subplot(5, 1, 2)
    plt.plot(history['iterations'], history['psnr'], label='Model PSNR')
    plt.axhline(y=history['baseline_psnr'][-1], color='r', linestyle='--', 
                label=f'Baseline PSNR: {history["baseline_psnr"][-1]:.2f}dB')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title('PSNR')
    plt.xlabel('Iteration')
    plt.ylabel('PSNR (dB)')

    # Plot LPIPS
    plt.subplot(5, 1, 3)
    plt.plot(history['iterations'], history['lpips'], label='Model LPIPS')
    plt.axhline(y=history['baseline_lpips'][-1], color='r', linestyle='--', 
                label=f'Baseline LPIPS: {history["baseline_lpips"][-1]:.4f}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title('LPIPS (lower is better)')
    plt.xlabel('Iteration')
    plt.ylabel('LPIPS')

    # Plot SSIM
    plt.subplot(5, 1, 4)
    plt.plot(history['iterations'], history['ssim'], label='Model SSIM')
    plt.axhline(y=history['baseline_ssim'][-1], color='r', linestyle='--', 
                label=f'Baseline SSIM: {history["baseline_ssim"][-1]:.4f}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title('SSIM (higher is better)')
    plt.xlabel('Iteration')
    plt.ylabel('SSIM')

    # Plot learning rates
    plt.subplot(5, 1, 5)
    plt.plot(history['iterations'], history['learning_rate'], label='Learning Rate')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title('Learning Rate')
    plt.xlabel('Iteration')
    plt.ylabel('Learning Rate')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def visualize_translations(pred_dx, pred_dy, target_dx, target_dy, save_path='translation_vis.png'):
    """Create a simple visualization comparing predicted and target translations."""
    plt.figure(figsize=(10, 8))
    
    # Set up the plot with a light gray background grid
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    
    # Plot target points (blue circles)
    plt.scatter(target_dx, target_dy, c='blue', s=90, label='Target', alpha=0.7)
    
    # Plot predicted points (red x's)
    plt.scatter(pred_dx, pred_dy, c='red', s=90, label='Predicted', marker='x', alpha=0.7)
    
    # Connect corresponding points with gray lines
    for i in range(len(target_dx)):
        plt.plot([target_dx[i], pred_dx[i]], [target_dy[i], pred_dy[i]], 'gray', alpha=0.5)
        
        # Add sample number next to the target point with larger font and background
        plt.text(target_dx[i], target_dy[i], f' {i}', 
                fontsize=10,  # Increased font size
                fontweight='normal'  # Make the text bold
                )
    
    # Add labels and title
    plt.xlabel('X Translation')
    plt.ylabel('Y Translation')
    plt.title('Target vs. Predicted Translations')
    plt.legend(loc='upper right')
    
    # Make sure the aspect ratio is equal
    plt.axis('equal')
    
    # Add a bit of padding around the points
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(save_path, dpi=150)
    plt.close()

def create_comparison_grid(hr_file, lr_file, baseline_file, baseline_metrics, output_files, model_names, metrics, sample_name, save_path):
    """
    Create a grid of images comparing different model outputs for the same sample.
    
    Args:
        hr_file: Path to HR ground truth image
        lr_file: Path to LR input image
        baseline_file: Path to bilinear baseline image
        baseline_metrics: Dictionary of baseline metrics
        output_files: List of paths to model output images
        model_names: List of model names corresponding to the outputs
        metrics: List of metrics dictionaries for each model
        sample_name: Name of the sample being compared
        save_path: Path to save the output grid
    """
    n_models = len(output_files)
    
    if n_models == 0:
        return
    
    # Create a figure with a grid of subplots
    # We'll have n_models + 3 images (HR, LR, Baseline, and one for each model)
    n_cols = min(3, n_models + 3)  # Limit columns to prevent very wide images
    n_rows = (n_models + 3 + n_cols - 1) // n_cols  # Ceiling division
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    elif n_rows == 1 or n_cols == 1:
        axes = axes.reshape(-1)
    axes = axes.flatten()
    
    # Load and display HR ground truth
    hr_img = plt.imread(hr_file)
    axes[0].imshow(hr_img)
    axes[0].set_title('HR Ground Truth')
    axes[0].axis('off')
    
    # Load and display LR input
    lr_img = plt.imread(lr_file)
    axes[1].imshow(lr_img)
    axes[1].set_title('LR Input')
    axes[1].axis('off')
    
    # Load and display baseline prediction
    baseline_img = plt.imread(baseline_file)
    axes[2].imshow(baseline_img)
    title = "Bilinear Baseline\n"
    title += f"PSNR: {baseline_metrics.get('final_psnr', 0):.2f} dB\n"
    title += f"LPIPS: {baseline_metrics.get('final_lpips', 0):.4f}\n"
    title += f"SSIM: {baseline_metrics.get('final_ssim', 0):.4f}"
    axes[2].set_title(title)
    axes[2].axis('off')
    
    # Now add each model's output
    for i, (output_file, model_name, model_metrics) in enumerate(zip(output_files, model_names, metrics)):
        # Load the model output image
        model_output = plt.imread(output_file)
        
        # Display in the appropriate subplot
        axes[i+3].imshow(model_output)
        
        # Create title with model name and metrics
        title = f"{model_name}\n"
        title += f"PSNR: {model_metrics.get('final_psnr', 0):.2f} dB\n"
        title += f"LPIPS: {model_metrics.get('final_lpips', 0):.4f}\n"
        title += f"SSIM: {model_metrics.get('final_ssim', 0):.4f}"
        
        axes[i+3].set_title(title)
        axes[i+3].axis('off')
    
    # Hide any unused subplots
    for i in range(n_models + 3, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f"Model Comparison for Sample: {sample_name}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Make room for the suptitle
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

def create_model_comparison_grid(base_dir, save_dir):
    """
    Create a grid visualization showing outputs from all models for each sample.
    
    Args:
        base_dir: Path to the base results directory
        save_dir: Directory to save visualizations
    """
    # Find all dataset directories
    dataset_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
    
    for dataset_dir in dataset_dirs:
        # Skip directories that don't look like dataset directories
        if dataset_dir.name.startswith('.') or dataset_dir.name == 'aggregated_results.csv':
            continue
            
        # Find all experiment directories
        experiment_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
        
        for experiment_dir in experiment_dirs:
            # Find all sample directories
            sample_dirs = [d for d in experiment_dir.iterdir() if d.is_dir()]
            
            for sample_dir in sample_dirs:
                # Find all model directories for this sample
                model_dirs = [d for d in sample_dir.iterdir() if d.is_dir()]
                
                if not model_dirs:
                    continue
                
                # Find image files in each model directory
                model_outputs = []
                model_names = []
                metrics = []
                hr_ground_truth = None
                lr_input = None
                baseline_pred = None
                baseline_metrics = None
                
                for model_dir in model_dirs:
                    # Look for the necessary files
                    output_file = model_dir / 'output_prediction.png'
                    hr_file = model_dir / 'hr_ground_truth.png'
                    lr_file = model_dir / 'lr_input.png'
                    baseline_file = model_dir / 'baseline_prediction.png'
                    metrics_file = model_dir / 'metrics.json'
                    
                    if output_file.exists() and metrics_file.exists():
                        # Read metrics
                        with open(metrics_file, 'r') as f:
                            model_metrics = json.load(f)
                        
                        # Add to our lists
                        model_outputs.append(output_file)
                        model_names.append(model_dir.name)
                        metrics.append(model_metrics)
                        
                        # Store HR, LR, and baseline files (they're the same for all models)
                        if hr_ground_truth is None and hr_file.exists():
                            hr_ground_truth = hr_file
                        if lr_input is None and lr_file.exists():
                            lr_input = lr_file
                        if baseline_pred is None and baseline_file.exists():
                            baseline_pred = baseline_file
                            # Extract baseline metrics from the same metrics file
                            baseline_metrics = {
                                'final_psnr': model_metrics.get('final_baseline_psnr', 0),
                                'final_lpips': model_metrics.get('final_baseline_lpips', 0),
                                'final_ssim': model_metrics.get('final_baseline_ssim', 0)
                            }
                
                if not model_outputs or hr_ground_truth is None or lr_input is None or baseline_pred is None:
                    print(f"Missing required files for {sample_dir.name}. Skipping.")
                    continue
                
                # Now create a grid comparing all model outputs
                create_comparison_grid(
                    hr_ground_truth,
                    lr_input,
                    baseline_pred,
                    baseline_metrics,
                    model_outputs, 
                    model_names, 
                    metrics, 
                    sample_dir.name,
                    save_dir / f"{dataset_dir.name}_{experiment_dir.name}_{sample_dir.name}_comparison_grid.png"
                )
                print(f"Created comparison grid for sample: {sample_dir.name}")

def visualize_model_comparisons(aggregated_results, save_dir):
    """
    Create visualizations comparing mean metrics across all models versus baseline.
    
    Args:
        aggregated_results: DataFrame with all results
        save_dir: Directory to save visualizations
    """
    if aggregated_results.empty:
        return
    
    # Group by model and projection type to get aggregate statistics
    model_stats = aggregated_results.groupby(['model_type', 'projection_type']).agg({
        'final_psnr': ['mean', 'std'],
        'final_lpips': ['mean', 'std'],
        'final_ssim': ['mean', 'std'],
        'final_baseline_psnr': ['mean'],
        'final_baseline_lpips': ['mean'],
        'final_baseline_ssim': ['mean'],
        'psnr_improvement': ['mean', 'std']
    }).reset_index()
    
    # Create labels for x-axis
    labels = [f"{row['model_type'][0]}_{row['projection_type'][0]}" for _, row in model_stats.iterrows()]
    x = np.arange(len(labels))
    
    # Set width of bars
    width = 0.35
    
    # Create figure with 3 subplots (PSNR, LPIPS, SSIM)
    fig, axs = plt.subplots(3, 1, figsize=(14, 18))
    
    # Plot PSNR
    axs[0].bar(x - width/2, model_stats['final_psnr']['mean'], width, 
              yerr=model_stats['final_psnr']['std'], label='Model')
    axs[0].bar(x + width/2, model_stats['final_baseline_psnr']['mean'], width, 
              label='Baseline', color='lightgray')
    axs[0].set_ylabel('PSNR (dB)')
    axs[0].set_title('PSNR Comparison (higher is better)')
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(labels, rotation=45, ha='right')
    axs[0].legend()
    axs[0].grid(axis='y', alpha=0.3)
    
    # Add PSNR improvement as text above bars
    for i, (mean, std) in enumerate(zip(model_stats['psnr_improvement']['mean'], 
                                       model_stats['psnr_improvement']['std'])):
        axs[0].annotate(f"+{mean:.2f}dB", 
                      xy=(i - width/2, model_stats['final_psnr']['mean'][i] + 0.5), 
                      ha='center', va='bottom', 
                      fontweight='bold', color='green')
    
    # Plot LPIPS (lower is better)
    axs[1].bar(x - width/2, model_stats['final_lpips']['mean'], width, 
              yerr=model_stats['final_lpips']['std'], label='Model')
    axs[1].bar(x + width/2, model_stats['final_baseline_lpips']['mean'], width, 
              label='Baseline', color='lightgray')
    axs[1].set_ylabel('LPIPS')
    axs[1].set_title('LPIPS Comparison (lower is better)')
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(labels, rotation=45, ha='right')
    axs[1].legend()
    axs[1].grid(axis='y', alpha=0.3)
    axs[1].invert_yaxis()  # Invert for LPIPS since lower is better
    
    # Plot SSIM (higher is better)
    axs[2].bar(x - width/2, model_stats['final_ssim']['mean'], width, 
              yerr=model_stats['final_ssim']['std'], label='Model')
    axs[2].bar(x + width/2, model_stats['final_baseline_ssim']['mean'], width, 
              label='Baseline', color='lightgray')
    axs[2].set_ylabel('SSIM')
    axs[2].set_title('SSIM Comparison (higher is better)')
    axs[2].set_xticks(x)
    axs[2].set_xticklabels(labels, rotation=45, ha='right')
    axs[2].legend()
    axs[2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'model_metrics_comparison.png')
    plt.close(fig)

def visualize_per_sample_metrics(aggregated_results, save_dir):
    """
    Create visualizations showing metrics for each sample across models.
    
    Args:
        aggregated_results: DataFrame with all results
        save_dir: Directory to save visualizations
    """
    if aggregated_results.empty:
        return
    
    # Get all unique samples
    samples = aggregated_results['sample_id'].unique()
    
    # For each metric, create a bar chart comparing models for each sample
    metrics = ['final_psnr', 'final_lpips', 'final_ssim', 'psnr_improvement']
    metric_titles = {
        'final_psnr': 'PSNR (dB) by Model',
        'final_lpips': 'LPIPS by Model (lower is better)',
        'final_ssim': 'SSIM by Model (higher is better)',
        'psnr_improvement': 'PSNR Improvement over Baseline (dB)'
    }
    
    for sample in samples:
        # Filter for this sample
        sample_data = aggregated_results[aggregated_results['sample_id'] == sample]
        
        if sample_data.empty:
            continue
            
        # Create a figure with subplots for each metric
        fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4*len(metrics)))
        
        # Get model-projection combinations for this sample
        model_combos = sample_data[['model_type', 'projection_type']].drop_duplicates()
        labels = [f"{row['model_type']}_{row['projection_type']}" for _, row in model_combos.iterrows()]
        
        for i, metric in enumerate(metrics):
            # Get data for this metric
            metric_values = []
            
            for _, row in model_combos.iterrows():
                model = row['model_type']
                proj = row['projection_type']
                
                # Get the metric value for this model-projection combo
                value = sample_data[
                    (sample_data['model_type'] == model) & 
                    (sample_data['projection_type'] == proj)
                ][metric].values
                
                if len(value) > 0:
                    metric_values.append(value[0])
                else:
                    metric_values.append(0)
            
            # Plot bar chart
            axes[i].bar(range(len(labels)), metric_values)
            axes[i].set_xticks(range(len(labels)))
            axes[i].set_xticklabels(labels, rotation=45, ha='right')
            axes[i].set_title(f"{metric_titles[metric]} - Sample: {sample}")
            axes[i].grid(axis='y', alpha=0.3)
            
            # For LPIPS, lower is better, so invert y-axis
            if metric == 'final_lpips':
                axes[i].invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(save_dir / f'sample_{sample}_metrics.png')
        plt.close(fig)

def visualize_psnr_improvement_heatmap(aggregated_results, save_dir):
    """
    Create a heatmap showing PSNR improvement across model-types and samples.
    
    Args:
        aggregated_results: DataFrame with all results
        save_dir: Directory to save visualizations
    """
    if aggregated_results.empty:
        return
    
    # Prepare data in the right format for heatmap
    pivot_data = aggregated_results.pivot_table(
        index='sample_id', 
        columns=['model_type', 'projection_type'], 
        values='psnr_improvement'
    )
    
    # Create figure
    plt.figure(figsize=(14, max(8, len(pivot_data) * 0.4)))
    
    # Create heatmap
    sns.heatmap(pivot_data, annot=True, fmt=".2f", cmap="RdYlGn", 
                linewidths=.5, center=0, vmin=-1, vmax=pivot_data.max().max() + 1)
    
    plt.title('PSNR Improvement (dB) by Model and Sample', fontsize=14)
    plt.ylabel('Sample ID')
    plt.xlabel('Model Type and Projection')
    plt.tight_layout()
    plt.savefig(save_dir / 'psnr_improvement_heatmap.png')
    plt.close()

def visualize_metrics_correlation(aggregated_results, save_dir):
    """
    Create scatter plots showing correlations between different metrics.
    
    Args:
        aggregated_results: DataFrame with all results
        save_dir: Directory to save visualizations
    """
    if aggregated_results.empty:
        return
    
    # Define metrics to plot
    metrics = ['final_psnr', 'final_lpips', 'final_ssim']
    metric_labels = {'final_psnr': 'PSNR (dB)', 
                    'final_lpips': 'LPIPS (lower is better)', 
                    'final_ssim': 'SSIM (higher is better)'}
    
    # Create a grid of scatter plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Get unique model-projection combinations
    model_projections = aggregated_results[['model_type', 'projection_type']].drop_duplicates()
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_projections)))
    
    # Scatter plots for PSNR vs LPIPS, PSNR vs SSIM, LPIPS vs SSIM
    plot_pairs = [
        ('final_psnr', 'final_lpips', axes[0]),
        ('final_psnr', 'final_ssim', axes[1]),
        ('final_lpips', 'final_ssim', axes[2])
    ]
    
    for (x_metric, y_metric, ax) in plot_pairs:
        # Plot each model-projection combination with a different color
        for i, (_, row) in enumerate(model_projections.iterrows()):
            model = row['model_type']
            proj = row['projection_type']
            
            # Filter data for this model-projection
            model_data = aggregated_results[
                (aggregated_results['model_type'] == model) & 
                (aggregated_results['projection_type'] == proj)
            ]
            
            # Plot the data points
            ax.scatter(model_data[x_metric], model_data[y_metric], 
                      color=colors[i], label=f"{model}_{proj}", alpha=0.7)
        
        # Add labels and title
        ax.set_xlabel(metric_labels[x_metric])
        ax.set_ylabel(metric_labels[y_metric])
        ax.set_title(f"{metric_labels[x_metric]} vs {metric_labels[y_metric]}")
        ax.grid(True, alpha=0.3)
        
        # Invert y-axis for LPIPS (lower is better)
        if y_metric == 'final_lpips':
            ax.invert_yaxis()
        
        # Invert x-axis for LPIPS (lower is better)
        if x_metric == 'final_lpips':
            ax.invert_xaxis()
    
    # Add a single legend for all plots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0), 
              ncol=min(5, len(labels)), frameon=True)
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])  # Make room for the legend
    plt.savefig(save_dir / 'metrics_correlation.png')
    plt.close()

def visualize_improvement_across_samples(aggregated_results, save_dir):
    """
    Create a grouped bar chart showing improvement metrics for each model across all samples.
    
    Args:
        aggregated_results: DataFrame with all results
        save_dir: Directory to save visualizations
    """
    if aggregated_results.empty:
        return
    
    # Create model_projection column for easier grouping
    aggregated_results['model_projection'] = aggregated_results['model_type'] + '_' + aggregated_results['projection_type']
    
    # Get unique samples and models
    samples = sorted(aggregated_results['sample_id'].unique())
    models = sorted(aggregated_results['model_projection'].unique())
    
    # Set up the plot
    fig, ax = plt.figure(figsize=(max(12, len(samples) * 1.5), 10)), plt.gca()
    
    # Number of groups and width of bars
    n_models = len(models)
    width = 0.8 / n_models
    
    # Set up colors
    colors = plt.cm.viridis(np.linspace(0, 0.9, n_models))
    
    # Plot bars for each model
    for i, model in enumerate(models):
        # Get data for this model
        model_data = aggregated_results[aggregated_results['model_projection'] == model]
        
        # Prepare data for plotting
        x_positions = []
        improvements = []
        
        for sample in samples:
            sample_data = model_data[model_data['sample_id'] == sample]
            if not sample_data.empty:
                x_positions.append(samples.index(sample))
                improvements.append(sample_data['psnr_improvement'].values[0])
            else:
                x_positions.append(samples.index(sample))
                improvements.append(0)
        
        # Plot the bars
        pos = [x + width * (i - n_models/2 + 0.5) for x in range(len(samples))]
        bars = ax.bar(pos, improvements, width, label=model, color=colors[i], alpha=0.8)
        
        # Add text labels for significant improvements
        for j, improvement in enumerate(improvements):
            if abs(improvement) >= 1.0:  # Only label significant improvements
                ax.text(pos[j], improvement + (0.2 if improvement >= 0 else -0.4), 
                       f"{improvement:.1f}", ha='center', fontsize=8, fontweight='bold',
                       color='green' if improvement >= 0 else 'red')
    
    # Add a horizontal line at y=0
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Set labels and title
    ax.set_xlabel('Sample ID')
    ax.set_ylabel('PSNR Improvement over Baseline (dB)')
    ax.set_title('PSNR Improvement by Model and Sample', fontsize=14)
    
    # Set x-tick labels to sample IDs
    ax.set_xticks(range(len(samples)))
    ax.set_xticklabels(samples, rotation=90)
    
    # Add grid
    ax.grid(axis='y', alpha=0.3)
    
    # Add legend with smaller font outside the plot
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), 
             ncol=min(5, len(models)), fontsize=10)
    
    plt.tight_layout(rect=[0, 0.1, 1, 1])  # Make room for the legend
    plt.savefig(save_dir / 'improvement_across_samples.png')
    plt.close()

def visualize_parallel_coordinates(aggregated_results, save_dir):
    """
    Create a parallel coordinates plot showing model performance across multiple metrics.
    
    Args:
        aggregated_results: DataFrame with all results
        save_dir: Directory to save visualizations
    """
    if aggregated_results.empty:
        return
    
    # Create model_projection column for easier grouping
    aggregated_results['model_projection'] = aggregated_results['model_type'] + '_' + aggregated_results['projection_type']
    
    # Group by model_projection to get average metrics
    model_stats = aggregated_results.groupby('model_projection').agg({
        'final_psnr': 'mean',
        'final_lpips': 'mean',
        'final_ssim': 'mean',
        'psnr_improvement': 'mean'
    }).reset_index()
    
    # For LPIPS, lower is better, so invert it for consistent visualization
    # (multiply by -1 to maintain the same scale but reverse direction)
    model_stats['final_lpips_inverted'] = -model_stats['final_lpips']
    
    # Select metrics for parallel coordinates
    metrics = ['final_psnr', 'final_lpips_inverted', 'final_ssim', 'psnr_improvement']
    metric_labels = {
        'final_psnr': 'PSNR (dB) ↑',
        'final_lpips_inverted': 'LPIPS (inverted) ↑',
        'final_ssim': 'SSIM ↑',
        'psnr_improvement': 'PSNR Improvement (dB) ↑'
    }
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Create parallel coordinates plot
    pd.plotting.parallel_coordinates(
        model_stats, 'model_projection', 
        cols=metrics,
        colormap=plt.cm.viridis,
        alpha=0.8,
        linewidth=2.5
    )
    
    # Improve axis labels
    ax = plt.gca()
    ax.set_xticklabels([metric_labels[m] for m in metrics], rotation=15)
    
    # Add a legend with smaller font size
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.12), 
             ncol=min(4, len(labels)), fontsize=10)
    
    # Add title
    plt.title('Model Performance Across Multiple Metrics', fontsize=14)
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.98])  # Make room for the legend
    plt.savefig(save_dir / 'parallel_coordinates.png')
    plt.close()

def visualize_metric_rankings(aggregated_results, save_dir):
    """
    Create a ranking plot showing how models rank on each metric.
    
    Args:
        aggregated_results: DataFrame with all results
        save_dir: Directory to save visualizations
    """
    if aggregated_results.empty:
        return
    
    # Create model_projection column for easier grouping
    aggregated_results['model_projection'] = aggregated_results['model_type'] + '_' + aggregated_results['projection_type']
    
    # Group by model and get average metrics
    model_stats = aggregated_results.groupby('model_projection').agg({
        'final_psnr': 'mean',
        'final_lpips': 'mean',
        'final_ssim': 'mean',
        'psnr_improvement': 'mean'
    }).reset_index()
    
    # Define metrics and their labels
    metrics = ['final_psnr', 'final_lpips', 'final_ssim', 'psnr_improvement']
    metric_labels = {
        'final_psnr': 'PSNR (dB)',
        'final_lpips': 'LPIPS',
        'final_ssim': 'SSIM',
        'psnr_improvement': 'PSNR Improvement (dB)'
    }
    
    # For each metric, compute the ranking (1 is best)
    for metric in metrics:
        if metric == 'final_lpips':  # Lower is better
            model_stats[f'{metric}_rank'] = model_stats[metric].rank()
        else:  # Higher is better
            model_stats[f'{metric}_rank'] = model_stats[metric].rank(ascending=False)
    
    # Create a figure with a subplot for each metric
    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 6))
    
    # For each metric, plot the ranks
    for i, metric in enumerate(metrics):
        # Sort by rank for this metric
        sorted_data = model_stats.sort_values(f'{metric}_rank')
        
        # Determine bar colors based on actual metric values
        if metric == 'final_lpips':  # Lower is better
            norm = plt.Normalize(sorted_data[metric].min(), sorted_data[metric].max())
            colors = plt.cm.RdYlGn_r(norm(sorted_data[metric]))
        else:  # Higher is better
            norm = plt.Normalize(sorted_data[metric].min(), sorted_data[metric].max())
            colors = plt.cm.RdYlGn(norm(sorted_data[metric]))
        
        # Plot horizontal bars
        y_pos = range(len(sorted_data))
        bars = axes[i].barh(y_pos, sorted_data[f'{metric}_rank'], color=colors)
        
        # Add model names as y-tick labels
        axes[i].set_yticks(y_pos)
        axes[i].set_yticklabels(sorted_data['model_projection'])
        
        # Add value annotations to the bars
        for j, (_, row) in enumerate(sorted_data.iterrows()):
            if metric == 'final_lpips':
                value_text = f"{row[metric]:.3f}"
            elif metric in ['final_psnr', 'psnr_improvement']:
                value_text = f"{row[metric]:.2f} dB"
            else:
                value_text = f"{row[metric]:.3f}"
                
            axes[i].text(row[f'{metric}_rank'] + 0.1, j, value_text, 
                        va='center', fontsize=9)
        
        # Set title and labels
        axes[i].set_title(metric_labels[metric])
        axes[i].set_xlabel('Rank (1 is best)')
        if i == 0:
            axes[i].set_ylabel('Model')
        
        # Set x-axis limits
        axes[i].set_xlim(0.5, len(sorted_data) + 0.5)
        
        # Invert x-axis so rank 1 is on left
        axes[i].invert_xaxis()
        
        # Add grid
        axes[i].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'metric_rankings.png')
    plt.close()

def visualize_performance_distribution(aggregated_results, save_dir):
    """
    Create violin plots showing the distribution of performance metrics across models.
    
    Args:
        aggregated_results: DataFrame with all results
        save_dir: Directory to save visualizations
    """
    if aggregated_results.empty:
        return
    
    # Create model_projection column for easier grouping
    aggregated_results['model_projection'] = aggregated_results['model_type'] + '_' + aggregated_results['projection_type']
    
    # Define metrics and their labels
    metrics = ['final_psnr', 'final_lpips', 'final_ssim', 'psnr_improvement']
    metric_titles = {
        'final_psnr': 'PSNR (dB) Distribution',
        'final_lpips': 'LPIPS Distribution (lower is better)',
        'final_ssim': 'SSIM Distribution (higher is better)',
        'psnr_improvement': 'PSNR Improvement (dB) Distribution'
    }
    
    # Create a figure with subplots for each metric
    fig, axes = plt.subplots(len(metrics), 1, figsize=(14, 5 * len(metrics)))
    
    for i, metric in enumerate(metrics):
        # Create violin plot with updated parameters to fix deprecation warning
        sns.violinplot(data=aggregated_results, 
                      x='model_projection', 
                      y=metric,
                      hue='model_projection',  # Add hue parameter
                      legend=False,  # Disable legend since it's redundant
                      ax=axes[i])
        
        # Add individual data points with jitter
        sns.stripplot(x='model_projection', 
                     y=metric, 
                     data=aggregated_results, 
                     ax=axes[i], 
                     color='black', 
                     size=4, 
                     alpha=0.4, 
                     jitter=True)
        
        # Add horizontal line for baseline where appropriate
        if metric in ['final_psnr', 'final_lpips', 'final_ssim']:
            baseline_metric = f'final_baseline_{metric.split("_")[1]}'
            baseline_mean = aggregated_results[baseline_metric].mean()
            axes[i].axhline(y=baseline_mean, color='r', linestyle='--', 
                          label=f'Baseline Mean: {baseline_mean:.2f}')
            axes[i].legend()
        
        # Add zero line for improvement
        if metric == 'psnr_improvement':
            axes[i].axhline(y=0, color='r', linestyle='--')
        
        # Set title and labels
        axes[i].set_title(metric_titles[metric])
        axes[i].set_xlabel('Model and Projection Type')
        axes[i].set_ylabel(metric)
        axes[i].grid(axis='y', alpha=0.3)
        
        # Rotate x-tick labels
        plt.setp(axes[i].get_xticklabels(), rotation=45, ha='right')
        
        # Invert y-axis for LPIPS (lower is better)
        if metric == 'final_lpips':
            axes[i].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(save_dir / 'performance_distribution.png')
    plt.close() 