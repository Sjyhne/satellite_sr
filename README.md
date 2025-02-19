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




## RUN

> python create_data_from_single_image.py

> python satellite_rs_train.py --iters <num_iters> --d <cuda:device> --use_gt <True/False> --df <downsample_factor>