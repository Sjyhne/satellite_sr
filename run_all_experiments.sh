#!/bin/bash

# Downsampling factors to test
FACTORS=(1 2 4 8)

# LR pixel shifts to test
SHIFTS=(0.5 1.0 2.0 4.0)

# Sample counts to test
SAMPLES=(4 8 12 16)

# Augmentation levels to test
AUGMENTATIONS=("none" "light" "medium" "heavy")

# GPU device to use
DEVICE=0

# Number of iterations
ITERS=3000

# Model type (FourierNetwork or TransformFourierNetwork)
MODEL="FourierNetwork"

# Enable/disable wandb logging
USE_WANDB=true

# Loop through all combinations
for df in "${FACTORS[@]}"; do
    for shift in "${SHIFTS[@]}"; do
        for samples in "${SAMPLES[@]}"; do
            for aug in "${AUGMENTATIONS[@]}"; do
                echo "Running experiment with factor ${df}x, shift ${shift}px, ${samples} samples, and ${aug} augmentation"
                
                # Add wandb flag if enabled
                WANDB_FLAG=""
                if [ "$USE_WANDB" = true ]; then
                    WANDB_FLAG="--wandb"
                fi
                
                python satellite_rs_train.py \
                    --d $DEVICE \
                    --df $df \
                    --lr_shift $shift \
                    --samples $samples \
                    --model $MODEL \
                    --iters $ITERS \
                    --aug $aug \
                    $WANDB_FLAG
                
                # Optional: wait a few seconds between runs
                sleep 5
            done
        done
    done
done

echo "All experiments completed!" 