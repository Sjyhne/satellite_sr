#!/bin/bash

# Downsampling factors to test
FACTORS=(1 2 4 8)

# LR pixel shifts to test
SHIFTS=(0.5 1.0 2.0 4.0)

# GPU device to use
DEVICE=0

# Number of iterations
ITERS=3000

# Model type (FourierNetwork or TransformFourierNetwork)
MODEL="FourierNetwork"

# Loop through all combinations
for df in "${FACTORS[@]}"; do
    for shift in "${SHIFTS[@]}"; do
        echo "Running experiment with factor ${df}x and shift ${shift}px"
        python satellite_rs_train.py \
            --d $DEVICE \
            --df $df \
            --lr_shift $shift \
            --model $MODEL \
            --iters $ITERS
        
        # Optional: wait a few seconds between runs to let GPU cool down
        sleep 5
    done
done

echo "All experiments completed!" 