#!/bin/bash
# Run evaluation for baseline and DiT models across all resolutions and noise levels.
# Usage: bash scripts/run_comparison.sh
#
# Output layout:
#   output_traj/baseline_20k/PUNet/P2P-Bridge_ema_steps_10_{res}_{noise}/
#   output_traj/dit_20k/PUNet/P2P-Bridge_ema_steps_10_{res}_{noise}/
#   output_traj/comparison.csv
#   output_traj/plots/

set -e

SCRIPT="evaluate_objects_traj.py"
DATA_PATH="./data/objects/examples/"
DATASET_ROOT="./data/objects/"
DATASET="PUNet"
GPU="cuda:0"
STEPS=10

BASELINE_CKPT="experiments/baseline_sde/baseline_sde/step_20000.pth"
DIT_CKPT="experiments/dit_sde/dit_sde/step_20000.pth"

echo "========== Evaluating Baseline (step 20000) =========="
python $SCRIPT \
    --model_path $BASELINE_CKPT \
    --output_root output_traj/baseline_20k \
    --data_path $DATA_PATH \
    --dataset_root $DATASET_ROOT \
    --dataset $DATASET \
    --gpu $GPU \
    --steps $STEPS \
    --use_ema \
    --seed 42

echo "========== Evaluating DiT (step 20000) =========="
python $SCRIPT \
    --model_path $DIT_CKPT \
    --output_root output_traj/dit_20k \
    --data_path $DATA_PATH \
    --dataset_root $DATASET_ROOT \
    --dataset $DATASET \
    --gpu $GPU \
    --steps $STEPS \
    --use_ema \
    --seed 42

echo "========== Merging results =========="
python scripts/merge_results.py \
    --output_root output_traj \
    --models baseline_20k dit_20k \
    --dataset $DATASET \
    --out_csv output_traj/comparison.csv

echo "========== Visualizing results =========="
python scripts/visualize_comparison.py \
    --csv output_traj/comparison.csv \
    --out_dir output_traj/plots

echo "Done. Results at output_traj/comparison.csv and output_traj/plots/"
