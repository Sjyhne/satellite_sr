#!/bin/bash

# Downsampling factors to test
FACTORS=(2 4 8) # (1 2 4 8)

# LR pixel shifts to test
SHIFTS=(0.5 1.0 2.0) # (0.5 1.0 2.0 4.0)

# Sample counts to test
SAMPLES=(2 8 16) # (4 8 12 16)

# Augmentation levels to test
AUGMENTATIONS=("light") #("none" "light" "medium" "heavy")

# Valid model and input projection combinations
MODEL_INPUT_PROJECTIONS=(
    "mlp fourier_5"
    "mlp fourier_10"
    "siren none"
    "wire none"
)

# GPU device to use
DEVICE=6

# Number of iterations
ITERS=3000

# Dataset to use
DATASET="satburst_synth"

# Network configuration
NETWORK_DEPTH=4
NETWORK_HIDDEN_DIM=256
PROJECTION_DIM=256

# Input projection parameters
# FOURIER_SCALE=10.0
LEGENDRE_MAX_DEGREE=$((PROJECTION_DIM / 2 - 1))

# Training parameters
LEARNING_RATE=5e-3
WEIGHT_DECAY=0.01
BATCH_SIZE=1

# Enable/disable wandb logging
USE_WANDB=false

# Loop through valid combinations only
for MODEL_PROJ in "${MODEL_INPUT_PROJECTIONS[@]}"; do
    MODEL=$(echo $MODEL_PROJ | cut -d' ' -f1)
    INPUT_PROJECTION=$(echo $MODEL_PROJ | cut -d' ' -f2)

    for df in "${FACTORS[@]}"; do
        for shift in "${SHIFTS[@]}"; do
            for samples in "${SAMPLES[@]}"; do
                for aug in "${AUGMENTATIONS[@]}"; do
                    echo "Running experiment with model ${MODEL}, projection ${INPUT_PROJECTION}, factor ${df}x, shift ${shift}px, ${samples} samples, and ${aug} augmentation"
                    
                    # Add wandb flag if enabled
                    WANDB_FLAG=""
                    if [ "$USE_WANDB" = true ]; then
                        WANDB_FLAG="--wandb"
                    fi
                    
                    # Add projection-specific parameters
                    PROJECTION_PARAMS=""
                    if [[ "$INPUT_PROJECTION" == fourier* ]]; then
                        # Extract scale from the projection name (after the underscore)
                        FOURIER_SCALE=$(echo $INPUT_PROJECTION | cut -d'_' -f2)
                        # Set the input projection to just "fourier"
                        INPUT_PROJECTION="fourier"
                        PROJECTION_PARAMS="--fourier_scale $FOURIER_SCALE"
                    elif [ "$INPUT_PROJECTION" = "legendre" ]; then
                        PROJECTION_PARAMS="--legendre_max_degree $LEGENDRE_MAX_DEGREE"
                    fi
                    
                    python main.py \
                        --d $DEVICE \
                        --df $df \
                        --lr_shift $shift \
                        --num_samples $samples \
                        --aug $aug \
                        --dataset $DATASET \
                        --model $MODEL \
                        --network_depth $NETWORK_DEPTH \
                        --network_hidden_dim $NETWORK_HIDDEN_DIM \
                        --projection_dim $PROJECTION_DIM \
                        --input_projection $INPUT_PROJECTION \
                        $PROJECTION_PARAMS \
                        --iters $ITERS \
                        --learning_rate $LEARNING_RATE \
                        --weight_decay $WEIGHT_DECAY \
                        --bs $BATCH_SIZE \
                        $WANDB_FLAG
                    
                    # Optional: wait a few seconds between runs
                    sleep 5
                done
            done
        done
    done
done

echo "All experiments completed!" 