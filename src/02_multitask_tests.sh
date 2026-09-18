#!/usr/bin/env bash
# 02_multitask_tests.sh
# ---------------------
# Runs multi-task experiments with ResNet34.
#
# Usage:
#   ./02_multitask_tests.sh

set -euo pipefail

echo "======================================================="
echo "  Multi-task tests — ResNet34 — Init surgery or chemo"
echo "======================================================="


# ---------------------------------------------------------------------------
# 0.3 Segmentation / 0.7 Classification - Balanced
# ---------------------------------------------------------------------------

echo ""
echo ">>> ResNet34 | 0.3 Segmentation / 0.7 Classification | Balanced"

python main.py \
    --experiment-name init_surgery_init_cht_multitask_resnet34_balanced_03_07 \
    --task multitask \
    --encoder resnet34 \
    --encoder-weights imagenet \
    --weight-loss 0.3 0.7 \
    --dataset-folder ../dataset/images \
    --masks-folder ../dataset/masks \
    --classification-csv ../dataset/labels.csv


# ---------------------------------------------------------------------------
# 0.2 Segmentation / 0.8 Classification - Balanced
# ---------------------------------------------------------------------------

echo ""
echo ">>> ResNet34 | 0.2 Segmentation / 0.8 Classification | Balanced"

python main.py \
    --experiment-name init_surgery_init_cht_multitask_resnet34_balanced_02_08 \
    --task multitask \
    --encoder resnet34 \
    --encoder-weights imagenet \
    --weight-loss 0.2 0.8 \
    --dataset-folder ../dataset/images \
    --masks-folder ../dataset/masks \
    --classification-csv ../dataset/labels.csv


# ---------------------------------------------------------------------------
# 0.5 Segmentation / 0.5 Classification - Unbalanced
# ---------------------------------------------------------------------------

echo ""
echo ">>> ResNet34 | 0.5 Segmentation / 0.5 Classification | Unbalanced"

python main.py \
    --experiment-name init_surgery_init_cht_multitask_resnet34_unbalanced_05_05 \
    --task multitask \
    --encoder resnet34 \
    --encoder-weights imagenet \
    --no-balanced \
    --dataset-folder ../dataset/images \
    --masks-folder ../dataset/masks \
    --classification-csv ../dataset/labels.csv


# ---------------------------------------------------------------------------
# 0.3 Segmentation / 0.7 Classification - Unbalanced
# ---------------------------------------------------------------------------

echo ""
echo ">>> ResNet34 | 0.3 Segmentation / 0.7 Classification | Unbalanced"

python main.py \
    --experiment-name init_surgery_init_cht_multitask_resnet34_unbalanced_03_07 \
    --task multitask \
    --encoder resnet34 \
    --encoder-weights imagenet \
    --weight-loss 0.3 0.7 \
    --no-balanced \
    --dataset-folder ../dataset/images \
    --masks-folder ../dataset/masks \
    --classification-csv ../dataset/labels.csv


# ---------------------------------------------------------------------------
# 0.2 Segmentation / 0.8 Classification - Unbalanced
# ---------------------------------------------------------------------------

echo ""
echo ">>> ResNet34 | 0.2 Segmentation / 0.8 Classification | Unbalanced"

python main.py \
    --experiment-name init_surgery_init_cht_multitask_resnet34_unbalanced_02_08 \
    --task multitask \
    --encoder resnet34 \
    --encoder-weights imagenet \
    --weight-loss 0.2 0.8 \
    --no-balanced \
    --dataset-folder ../dataset/images \
    --masks-folder ../dataset/masks \
    --classification-csv ../dataset/labels.csv


echo ""
echo "============================================"
echo "  All multi-task tests completed"
echo "============================================"