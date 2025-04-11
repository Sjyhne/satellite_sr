# Satellite SR
Satellite super resolution using LR satellite images by learning the transformations

## Main idea
The main idea is to overfit/optimize a Coordinate-based MLP to learn the sub-pixel transformations in the LR images from a HR image and try to upsample it. This will be a self-supervised approach.

## Data Generation Flow

1. **Start with Original HR Image**
   - Input: Original HR image (e.g., 256×256)
   - This serves as our ground truth reference

2. **Upsampling Phase**
   - Double the size of original image (512×512)
   - Use bilinear interpolation
   - This gives us room for sub-pixel translations

3. **Translation Phase**
   - Generate 16 different random translations
   - Each translation is between -6 and +6 pixels in both x and y directions
   - Apply these translations to the upsampled image
   - This simulates different sub-pixel views of the same scene

4. **HR-LR Pair Creation**
   For each translated upsampled image:
   - HR: Downsample to original size (256×256)
   - LR: Further downsample to quarter size (64×64)
   - Use AREA interpolation for realistic downsampling

5. **Data Organization**
   Each sample is saved with:
   - LR image (64×64)
   - Corresponding HR image (256×256)
   - Transformation parameters in JSON:
     ```json
     {
       "sample_00": {
         "dx_pixels": 3.2,
         "dy_pixels": -2.1,
         "dx_percent": 0.00625,
         "dy_percent": -0.00410,
         "magnitude_pixels": 3.84,
         "hr_shape": [256, 256, 3],
         "lr_shape": [64, 64, 3]
       }
     }
     ```

This process creates a dataset where:
- Each LR image represents a slightly different view of the same scene
- The sub-pixel translations are precisely known
- The relationship between LR and HR pairs is well-defined
- All transformations are tracked and logged for evaluation

The goal is to learn these sub-pixel transformations to improve super-resolution quality.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/satellite_sr.git
   cd satellite_sr
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Generate Training Data

To create synthetic data from a single image:

```bash
python create_data_from_single_image.py --input_image path/to/image.png --output_dir data/sample_1
```

## Running the Training

### Basic Training

```bash
python main.py --dataset satburst_synth --sample_id sample_1 --df 4 --lr_shift 1.0 --iters 1000 --d 0
```

### Training with Different Models and Projections

```bash
# Train with MLP model and Fourier projection
python main.py --model mlp --input_projection fourier_10 --iters 2000 --df 4 --lr_shift 1.0 --dataset satburst_synth --sample_id sample_1 --d 0

# Train with SIREN model
python main.py --model siren --input_projection fourier_10 --iters 2000 --df 4 --lr_shift 1.0 --dataset satburst_synth --sample_id sample_1 --d 0
```

### Training on WorldStrat Dataset

```bash
# Basic WorldStrat training
python main.py --d 0 --lr_shift 1 --df 4 --num_samples 8 --bs 1 --dataset worldstrat --root_worldstrat ~/data/worldstrat_kaggle/ --area_name="Landcover-1295513" --iters 10000

# Optimize affine transformation (translation plus rotation angle)
python main.py --d 0 --lr_shift 1 --df 4 --num_samples 8 --bs 1 --dataset worldstrat --root_worldstrat ~/data/worldstrat_kaggle/ --area_name="UNHCR-SYRs008164" --iters 10000 --worldstrat_hr_size 512 --rotation True
```

### Training with Different Augmentation Levels

```bash
# No augmentation
python main.py --aug none --df 4 --lr_shift 1.0 --iters 2000 --dataset satburst_synth --sample_id sample_1 --d 0

# Light augmentation
python main.py --aug light --df 4 --lr_shift 1.0 --iters 2000 --dataset satburst_synth --sample_id sample_1 --d 0

# Heavy augmentation
python main.py --aug heavy --df 4 --lr_shift 1.0 --iters 2000 --dataset satburst_synth --sample_id sample_1 --d 0
```

## Parameter Description

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--dataset` | Dataset type: "satburst_synth", "worldstrat", "burst_synth" | "satburst_synth" |
| `--sample_id` | Sample ID for the dataset | "Landcover-743192_rgb" |
| `--df` | Downsampling factor | 4 |
| `--lr_shift` | Low-resolution shift amount | 1.0 |
| `--num_samples` | Number of LR samples to use | 16 |
| `--model` | Model type: "mlp", "siren", "wire", "linear", "conv", "thera" | "mlp" |
| `--input_projection` | Input projection: "linear", "fourier_10", "fourier_5", "fourier_20", "fourier_40", "fourier", "legendre", "none" | "fourier_10" |
| `--iters` | Number of training iterations | 1000 |
| `--d` | CUDA device number | "0" |
| `--use_gt` | Whether to use ground truth shifts | False |
| `--aug` | Augmentation level: "none", "light", "medium", "heavy" | "none" |
| `--rotation` | Whether to optimize rotation angle | False |

## Visualization Utilities

This project includes a comprehensive set of visualization utilities in the `viz_utils.py` file:

1. **Training Curves**: Visualize loss and metrics during training
   ```python
   from viz_utils import plot_training_curves
   plot_training_curves(history, save_path='training_curves.png')
   ```

2. **Translation Visualization**: Visualize predicted vs. ground truth translations
   ```python
   from viz_utils import visualize_translations
   visualize_translations(pred_dx, pred_dy, target_dx, target_dy, save_path='translations.png')
   ```

3. **Model Comparison**: Compare outputs from different models
   ```python
   from viz_utils import create_model_comparison_grid
   create_model_comparison_grid(base_dir='results', save_dir='comparison')
   ```

4. **Performance Metrics**: Visualize various metrics across models and samples
   ```python
   from viz_utils import visualize_model_comparisons, visualize_psnr_improvement_heatmap
   visualize_model_comparisons(aggregated_results, save_dir='metrics')
   visualize_psnr_improvement_heatmap(aggregated_results, save_dir='metrics')
   ```

These visualizations are automatically generated during training and stored in the results directory.

## Results

After training, results are saved in a structured directory:
```
results/
└── {dataset}/
    └── df{downsample_factor}_shift{lr_shift}_samples{num_samples}/
        └── {sample_id}/
            └── {model}_{projection}_{iters}/
                ├── comparison.png
                ├── final_training_curves.png
                ├── final_translation_vis.png
                ├── metrics.json
                ├── output_prediction.png
                ├── hr_ground_truth.png
                ├── lr_input.png
                └── baseline_prediction.png
```

## Aggregating Results

To aggregate results across multiple runs and generate comparison visualizations:

```bash
python aggregate_results.py --results_dir results
```

This will create aggregate visualizations in the results directory:
- Model metrics comparison
- PSNR improvement heatmap
- Per-sample metrics
- Performance distribution
- Metric rankings
- And more!


