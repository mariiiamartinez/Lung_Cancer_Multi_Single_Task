#!/usr/bin/env bash
# run_experiments.sh
# ------------------
# Runs the multi-task segmentation + classification experiments sequentially.
#
# Usage:
#   ./run_experiments.sh

set -euo pipefail

echo "============================================"
echo "  Multi-task segmentation + classification"
echo "============================================"

# ---------------------------------------------------------------------------
# Experiment 1: Balanced, loss weights 0.5 / 0.5
# ---------------------------------------------------------------------------

echo ""
echo ">>> Balanced | loss weights segm 0.5 / cls 0.5"
python main.py \
    --experiment-name multitask_tipo_1_vs_tipo_2_3_balanced \
    --task multitask \
    --encoder resnet50 \
    --encoder-weights imagenet \
    --dataset-folder ../dataset/images \
    --masks-folder ../dataset/masks \
    --classification-csv ../dataset/labels.csv \
    --num-classes 2 \
    --weight-loss 0.5 0.5 \
    --image-size 512 512 \
    --num-epochs 200 \
    --learning-rate 1e-4 \
    --lr-patience 10 \
    --lr-factor 0.3 \
    --n-repetitions 6 \
    --batch-size 16 \
    --num-workers 8 \
    --preprocessing Preprocessing \
    --data-augmentation DataAugmentation \
    --aug-strength medium \
    --seed 0

# ---------------------------------------------------------------------------
# Experiment 2: No balancing (keeps all images), loss weights 0.5 / 0.5
# ---------------------------------------------------------------------------

echo ""
echo ">>> No balancing | loss weights segm 0.5 / cls 0.5"
python main.py \
    --experiment-name multitask_tipo_1_vs_tipo_2_3_unbalanced \
    --task multitask \
    --encoder resnet50 \
    --encoder-weights imagenet \
    --dataset-folder ../dataset/images \
    --masks-folder ../dataset/masks \
    --classification-csv ../dataset/labels.csv \
    --num-classes 2 \
    --weight-loss 0.5 0.5 \
    --image-size 512 512 \
    --num-epochs 200 \
    --learning-rate 1e-4 \
    --lr-patience 10 \
    --lr-factor 0.3 \
    --n-repetitions 6 \
    --batch-size 16 \
    --num-workers 8 \
    --preprocessing Preprocessing \
    --data-augmentation DataAugmentation \
    --aug-strength medium \
    --no-balanced \
    --seed 0

# ---------------------------------------------------------------------------
# Experiment 3: Balanced, loss weights segm 0.3 / cls 0.7
# ---------------------------------------------------------------------------

echo ""
echo ">>> Balanced | loss weights segm 0.3 / cls 0.7"
python main.py \
    --experiment-name multitask_tipo_1_vs_tipo_2_3_weighted_cls \
    --task multitask \
    --encoder resnet50 \
    --encoder-weights imagenet \
    --dataset-folder ../dataset/images \
    --masks-folder ../dataset/masks \
    --classification-csv ../dataset/labels.csv \
    --num-classes 2 \
    --weight-loss 0.3 0.7 \
    --image-size 512 512 \
    --num-epochs 200 \
    --learning-rate 1e-4 \
    --lr-patience 10 \
    --lr-factor 0.3 \
    --n-repetitions 6 \
    --batch-size 16 \
    --num-workers 8 \
    --preprocessing Preprocessing \
    --data-augmentation DataAugmentation \
    --aug-strength medium \
    --seed 0

echo ""
echo "============================================"
echo "  All experiments finished."
echo "============================================"
