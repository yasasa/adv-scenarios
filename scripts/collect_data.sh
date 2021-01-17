#!/bin/bash

for SEED in 0 1 2 3 4
do

    python scripts/train_simulator_policy.py --collect-data --output-path ../experiments/rgb-carla-nerf-driving-extra2 --experiment-config configs/custom_train_configs/NGP-ColourAttack-1-2-2car-3hydrant-earlier-2.txt --uniform --no-start --train-samples-per-epoch 1 --rollout-batch-size 2 --priority-sample --parameter-dir ../NGP-ColourAttack-1-2-2car-3hydrant-earlier-2/MultiFrameNGPColourAttack/seed-$SEED/parameters-log-0.npy --T 600
    
    python scripts/train_simulator_policy.py --collect-data --output-path ../experiments/rgb-carla-nerf-driving-extra2 --experiment-config configs/custom_train_configs/NGP-ColourAttack-1-0-2car-3hydrant.txt --uniform --no-start --train-samples-per-epoch 1 --rollout-batch-size 2 --priority-sample --parameter-dir ../NGP-ColourAttack-1-0-2car-3hydrant/MultiFrameNGPColourAttack/seed-$SEED/parameters-log-0.npy --T 600
    
    python scripts/train_simulator_policy.py --collect-data --output-path ../experiments/rgb-carla-nerf-driving-extra2 --experiment-config configs/custom_train_configs/NGP-ColourAttack-0-1-2car-3hydrant.txt --uniform --no-start --train-samples-per-epoch 1 --rollout-batch-size 2 --priority-sample --parameter-dir ../exp-params/right-2c-3h/seed-$SEED.npy --T 600
    
done

python scripts/train_simulator_policy.py --fit-policy --output-path ../experiments/fixed-policy --load-extra-databases --extra-db-dir ../experiments/rgb-carla-nerf-driving-extra2 --input-path ../experiments/rgb-carla-nerf-driving/ --batch-size 200 --dataset-num-workers=19

