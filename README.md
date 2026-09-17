# Lung Cancer — Deep Learning Framework for Classification, Segmentation and Multi-Task Learning

## Overview

Deep learning framework for chest radiograph analysis, supporting **image classification, lung segmentation, and multi-task learning**. The framework includes configurable data preparation, preprocessing, data augmentation, model architectures, training strategies, evaluation, and Grad-CAM visualisation.

The framework supports both models trained from scratch and models using pretrained ResNet encoders. Experiments can be repeated with different seeds, using patient-level dataset splitting, optional class balancing, configurable losses, and automatic evaluation and result aggregation.


## Dataset preparation

The raw dataset is expected to contain chest radiographs, their lung segmentation masks, and a patient-level clinical register:

```text
raw_dataset/
├── cancer/
│   ├── 01500603_PA_2V.jpg
│   └── ...
├── masks_cancer/
│   ├── 01500603_PA_2V.png
│   └── ...
└── class_labels.xlsx
```

Radiograph filenames encode the patient ID. For example:

```text
01500603_PA_2V.jpg → patient 1500603
```

The Excel register must contain a `PATIENT ID` column and one or more clinical grouping columns that can be used as classification targets.

Run `group_dataset.py` by specifying the desired grouping column:

```bash
python group_dataset.py "TYPE STAGE"
```

Other examples:

```bash
python group_dataset.py "CHT — NO_CHT"
python group_dataset.py "UNIMODAL — MULTIMODAL"
python group_dataset.py "INIT_CHT — INIT_SURGERY"
```

The script matches each radiograph with its patient label and segmentation mask and creates:

```text
dataset/
├── images/
├── masks/
└── labels.csv
```

`labels.csv` contains:

```text
image_name,class_label
```

Images without a corresponding patient label or segmentation mask are excluded.


## Project structure

```text
Lung_Cancer_Multi_Single_Task/
│
├── raw_dataset/
│   ├── cancer/
│   ├── healthy/
│   ├── masks_cancer/
│   ├── masks_healthy/
│   └── class_labels.xlsx
│
├── dataset/
│   ├── images/
│   ├── masks/
│   └── labels.csv
│
├── group_dataset.py
│
├── src/
│   ├── main.py
│   ├── util_dataset.py
│   ├── util_models.py
│   ├── util_training.py
│   ├── util_metrics.py
│   ├── util_experiment.py
│   └── run_experiments.sh
│
└── Results/
```
* `util_dataset.py`: dataset loading, preprocessing, augmentation, balancing, and patient-level splitting.
* `util_models.py`: model architectures.
* `util_training.py`: training, validation, testing, losses, early stopping, and Grad-CAM.
* `util_metrics.py`: classification and segmentation metrics.
* `util_experiment.py`: repetitions, checkpoint selection, aggregation, and result management.
* `main.py`: command-line interface and experiment entry point.

## Models and tasks

Two model architectures are available.

### Custom U-Net

A 4-level U-Net trained from scratch. The initial number of feature channels can be configured with:

```text
--init-features 32 or 64
```

### SMP U-Net

U-Net with a ResNet encoder from Segmentation Models PyTorch, optionally using ImageNet pretrained weights. Supported encoders:

```text
--encoder resnet18 or resnet34 or resnet50
--encoder-weights ImageNet
```

The encoder can optionally be frozen with:
```text
--freeze-backbone
```

### Task modes

The model components used in each task mode are:

**Multi-task**

```text
Encoder → Decoder
             ├── Segmentation head
             └── Classification head
```

**Segmentation**

```text
Encoder → Decoder → Segmentation head
```

**Classification**

```text
Encoder → Bottleneck → Classification
```

In multi-task mode, both tasks share the encoder-decoder representation. In classification-only mode, classification is performed directly from the bottleneck.


## Running an experiment
Move to the `src/` directory before running an experiment.

Experiments can be run individually using `main.py` or in batches using `run_experiments.sh`.
```bash
    ./run_experiments.sh
```

Usage for multi-task learning:

```bash
python main.py \
    --experiment-name example \
    --task multitask \
    --dataset-folder ../dataset/images \
    --masks-folder ../dataset/masks \
    --classification-csv ../dataset/labels.csv \
    --num-classes 2
```

For classification:

```bash
python main.py \
    --experiment-name classification_example \
    --task classification \
    --dataset-folder ../dataset/images \
    --classification-csv ../dataset/labels.csv \
    --num-classes 2
```

For segmentation:

```bash
python main.py \
    --experiment-name segmentation_example \
    --task segmentation \
    --dataset-folder ../dataset/images \
    --masks-folder ../dataset/masks
```

The default configuration uses:

* 5 repetitions
* seed `1011`
* 70% train / 15% validation / 15% test
* image size `512 × 512`
* batch size `16`
* num workers `8`
* 200 epochs
* AdamW
* learning rate `1e-4`
* `ReduceLROnPlateau` scheduler
* lr patience `10`
* lr factor `0.3`
* data augmentation
* preprocessing
* class balancing

Splits are performed at **patient level** when classification labels are available, preventing images from the same patient from being distributed across different partitions.

## Main parameters

### Dataset

| Parameter                      | Description                             | Default                     |
| ------------------------------ | --------------------------------------- | --------------------------- |
| `--dataset-folder`             | Directory containing the images         | Required                    |
| `--masks-folder`               | Directory containing segmentation masks | Required for segmentation   |
| `--classification-csv`         | Classification labels CSV               | Required for classification |
| `--train-percentage`           | Training proportion                     | `0.70`                      |
| `--val-percentage`             | Validation proportion                   | `0.15`                      |
| `--test-percentage`            | Test proportion                         | `0.15`                      |
| `--balanced` / `--no-balanced` | Enable/disable class balancing          | Enabled                     |
| `--image-size H W`             | Input image size, divisible by 32       | `512 512`                   |

### Splitting

Holdout train/val/test split:
- 2 classes: balanced subsampling (majority reduced to minority count).
- 3 classes: majority subsampled to the average of the two minority counts.
- Patient-level `GroupShuffleSplit` using `seed + iteration` to avoid data leakage.
- Class balancing can be disabled with `--no-balanced` (all images are kept).


### Preprocessing and augmentation

| Argument                | Description            | Default            | Options                                  |
| ----------------------- | ---------------------- | ------------------ | ---------------------------------------- |
| `--preprocessing`       | Input normalization    | `Preprocessing`    | `Preprocessing`, `NoPreprocessing`       |
| `--augmentation-method` | Data augmentation      | `DataAugmentation` | `DataAugmentation`, `NoDataAugmentation` |
| `--aug-strength`        | Augmentation intensity | `medium`           | `light`, `medium`, `strong`              |

Preprocessing uses ImageNet statistics with ImageNet-pretrained encoders; otherwise, statistics are computed from the training subset. Augmentation is applied only to the training split.


### Losses

| Argument            | Description                              | Default      | Options                       |
| ------------------- | ---------------------------------------- | ------------ | ----------------------------- |
| `--cls-loss-func`   | Classification loss                      | `smooth_bce` | `smooth_bce`, `ce`, `focal`   |
| `--segm-loss-func`  | Segmentation loss                        | `dice`       | `dice`, `bce`, `focal`        |
| `--weight-loss S C` | Segmentation/classification loss weights | `0.5 0.5`    | Custom weights, must sum to 1 |

The loss weights apply to multi-task learning.

### Training

Early stopping can be enabled with:

```bash
--early-stopping
```

The best checkpoint is selected according to **validation loss** and saved as `best.pkl`.

## Outputs

Each experiment creates a timestamped directory:

```text
Results/
└── <experiment_name>_<timestamp>/
    ├── Rep1/
    │   ├── Models/
    │   ├── Screening/      # classification tasks
    │   ├── Segmentation/   # segmentation tasks
    │   └── loss_log.csv
    ├── Rep2/
    │   ├── Models/
    │   ├── Screening/
    │   ├── Segmentation/
    │   └── loss_log.csv
    ├── ...
    ├── Summary/
    └── output.log
```

`Models/` and `loss_log.csv` are generated for every repetition. `Screening/` and `Segmentation/` are generated only for the corresponding tasks.

Classification experiments include:

* confusion matrix
* accuracy, precision, recall and F1
* ROC-AUC
* per-image predictions and probabilities
* Grad-CAM visualisations

Segmentation experiments include:

* Dice
* IoU / Jaccard
* precision
* recall
* pixel accuracy
* predicted masks
* prediction overlays on ground-truth masks and original images

Results from repeated experiments are aggregated in `Summary/`, including mean and standard deviation across repetitions.
