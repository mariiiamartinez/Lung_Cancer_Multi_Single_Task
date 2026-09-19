#!/usr/bin/env bash
# run_baseline.sh
# ---------------
# Runs the baseline experiments sequentially.
#
# Usage:
#   ./run_baseline.sh

set -euo pipefail

echo "================================================"
echo "  Baseline experiments — Init surgery or chemo"
echo "================================================"

# ---------------------------------------------------------------------------
# U-Net 32
# ---------------------------------------------------------------------------

echo ""
echo ">>> U-Net 32 | Segmentation"
python main.py \
    --experiment-name init_surgery_init_cht_baseline_unet32_segmentation \
    --task segmentation \
    --init-features 32 \
    --dataset-folder ../dataset/images \
    --masks-folder ../dataset/masks \
    --classification-csv ../dataset/labels.csv

echo ""
echo ">>> U-Net 32 | Classification"
python main.py \
    --experiment-name init_surgery_init_cht_baseline_unet32_classification \
    --task classification \
    --init-features 32 \
    --dataset-folder ../dataset/images \
    --classification-csv ../dataset/labels.csv

echo ""
echo ">>> U-Net 32 | Multi-task"
python main.py \
    --experiment-name init_surgery_init_cht_baseline_unet32_multitask \
    --task multitask \
    --init-features 32 \
    --dataset-folder ../dataset/images \
    --masks-folder ../dataset/masks \
    --classification-csv ../dataset/labels.csv


# ---------------------------------------------------------------------------
# U-Net 64
# ---------------------------------------------------------------------------

echo ""
echo ">>> U-Net 64 | Segmentation"
python main.py \
    --experiment-name init_surgery_init_cht_baseline_unet64_segmentation \
    --task segmentation \
    --init-features 64 \
    --dataset-folder ../dataset/images \
    --masks-folder ../dataset/masks \
    --classification-csv ../dataset/labels.csv

echo ""
echo ">>> U-Net 64 | Classification"
python main.py \
    --experiment-name init_surgery_init_cht_baseline_unet64_classification \
    --task classification \
    --init-features 64 \
    --dataset-folder ../dataset/images \
    --classification-csv ../dataset/labels.csv

echo ""
echo ">>> U-Net 64 | Multi-task"
python main.py \
    --experiment-name init_surgery_init_cht_baseline_unet64_multitask \
    --task multitask \
    --init-features 64 \
    --dataset-folder ../dataset/images \
    --masks-folder ../dataset/masks \
    --classification-csv ../dataset/labels.csv


# ---------------------------------------------------------------------------
# ResNet18
# ---------------------------------------------------------------------------

echo ""
echo ">>> ResNet18 | Segmentation"
python main.py \
    --experiment-name init_surgery_init_cht_baseline_resnet18_segmentation \
    --task segmentation \
    --encoder resnet18 \
    --encoder-weights imagenet \
    --dataset-folder ../dataset/images \
    --masks-folder ../dataset/masks \
    --classification-csv ../dataset/labels.csv

echo ""
echo ">>> ResNet18 | Classification"
python main.py \
    --experiment-name init_surgery_init_cht_baseline_resnet18_classification \
    --task classification \
    --encoder resnet18 \
    --encoder-weights imagenet \
    --dataset-folder ../dataset/images \
    --classification-csv ../dataset/labels.csv

echo ""
echo ">>> ResNet18 | Multi-task"
python main.py \
    --experiment-name init_surgery_init_cht_baseline_resnet18_multitask \
    --task multitask \
    --encoder resnet18 \
    --encoder-weights imagenet \
    --dataset-folder ../dataset/images \
    --masks-folder ../dataset/masks \
    --classification-csv ../dataset/labels.csv


# ---------------------------------------------------------------------------
# ResNet34
# ---------------------------------------------------------------------------

echo ""
echo ">>> ResNet34 | Segmentation"
python main.py \
    --experiment-name init_surgery_init_cht_baseline_resnet34_segmentation \
    --task segmentation \
    --encoder resnet34 \
    --encoder-weights imagenet \
    --dataset-folder ../dataset/images \
    --masks-folder ../dataset/masks \
    --classification-csv ../dataset/labels.csv

echo ""
echo ">>> ResNet34 | Classification"
python main.py \
    --experiment-name init_surgery_init_cht_baseline_resnet34_classification \
    --task classification \
    --encoder resnet34 \
    --encoder-weights imagenet \
    --dataset-folder ../dataset/images \
    --classification-csv ../dataset/labels.csv

echo ""
echo ">>> ResNet34 | Multi-task"
python main.py \
    --experiment-name init_surgery_init_cht_baseline_resnet34_multitask \
    --task multitask \
    --encoder resnet34 \
    --encoder-weights imagenet \
    --dataset-folder ../dataset/images \
    --masks-folder ../dataset/masks \
    --classification-csv ../dataset/labels.csv


# ---------------------------------------------------------------------------
# ResNet50
# ---------------------------------------------------------------------------

echo ""
echo ">>> ResNet50 | Segmentation"
python main.py \
    --experiment-name init_surgery_init_cht_baseline_resnet50_segmentation \
    --task segmentation \
    --encoder resnet50 \
    --encoder-weights imagenet \
    --dataset-folder ../dataset/images \
    --masks-folder ../dataset/masks \
    --classification-csv ../dataset/labels.csv

echo ""
echo ">>> ResNet50 | Classification"
python main.py \
    --experiment-name init_surgery_init_cht_baseline_resnet50_classification \
    --task classification \
    --encoder resnet50 \
    --encoder-weights imagenet \
    --dataset-folder ../dataset/images \
    --classification-csv ../dataset/labels.csv

echo ""
echo ">>> ResNet50 | Multi-task"
python main.py \
    --experiment-name init_surgery_init_cht_baseline_resnet50_multitask \
    --task multitask \
    --encoder resnet50 \
    --encoder-weights imagenet \
    --dataset-folder ../dataset/images \
    --masks-folder ../dataset/masks \
    --classification-csv ../dataset/labels.csv


echo ""
echo "============================================"
echo "  All baseline experiments finished."
echo "============================================"