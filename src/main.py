"""Entry point for lung task experiments (classification, segmentation, multi-task).

Parses CLI arguments, validates them, seeds RNGs, creates the output directory
tree, and delegates to :func:`util_experiment.run_experiment`.
"""
import argparse
import os
import sys
from datetime import datetime
import pandas as pd
import torch

from util_experiment import run_experiment, set_seed


class _Tee:
    """Duplicate writes to several streams (console + log file)."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            stream.write(data)
        self.flush()

    def flush(self):
        for stream in self._streams:
            stream.flush()


def _build_parser() -> argparse.ArgumentParser:
    """Build and return the full CLI argument parser."""
    p = argparse.ArgumentParser(
        description="Lung task experiment (classification / segmentation / multi-task).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    exp = p.add_argument_group("Experiment")
    exp.add_argument("--experiment-name", type=str, required=True,
                     help="Unique name for this run; used as the Results/ subdirectory name.")
    exp.add_argument("--task", type=str, default="multitask",
                     choices=["classification", "segmentation", "multitask"],
                     help="Task to run: single-task classification, single-task "
                          "segmentation, or the joint multi-task model.")
    exp.add_argument("--n-repetitions", type=int, default=5, dest="nofrepetitions",
                     help="Number of independent train/eval repetitions.")
    exp.add_argument("--seed", type=int, default=1011,
                     help="Global seed for reproducibility.")

    dat = p.add_argument_group("Dataset")
    dat.add_argument("--dataset-folder", type=str, required=True, dest="input_dir_root",
                     help="Path to the directory where the input images are stored.")
    dat.add_argument("--masks-folder", type=str, default=None, dest="lung_segm_root_dir",
                     help="Path to the directory with the lung segmentation masks. ")
    dat.add_argument("--classification-csv", type=str, default=None,
                     dest="classification_csv_file_path",
                     help="Path to the CSV file with classification labels. ")
    dat.add_argument("--classification-image-column", type=str, default="image_name",
                     help="Name of the image-name column in the classification CSV.")
    dat.add_argument("--classification-label-column", type=str, default="class_label",
                     help="Name of the class-label column in the classification CSV.")
    dat.add_argument("--class-names", nargs="+", type=str, default=None,
                     help="Optional ordered class names. If omitted, they are inferred from the CSV labels.")
    dat.add_argument("--num-classes", type=int, default=2,
                     help="Number of output classes for the classification head.")
    dat.add_argument("--balanced", action=argparse.BooleanOptionalAction, default=True,
                     help="Undersample the majority class before the split (default: enabled). "
                          "Use --no-balanced to keep all images.")
    dat.add_argument("--train-per", type=float, default=0.70, dest="train_pct",
                     help="Fraction of data used for training.")
    dat.add_argument("--val-per", type=float, default=0.15, dest="val_pct",
                     help="Fraction of data used for validation.")
    dat.add_argument("--test-per", type=float, default=0.15, dest="test_pct",
                     help="Fraction of data used for testing.")
    dat.add_argument("--preprocessing", type=str, default="Preprocessing",
                     choices=["NoPreprocessing", "Preprocessing"],
                     help="Type of preprocessing applied on images.")
    dat.add_argument("--data-augmentation", type=str, default="DataAugmentation",
                     choices=["NoDataAugmentation", "DataAugmentation"],
                     help="Type of data augmentation applied during training.")
    dat.add_argument("--aug-strength", type=str, default="medium",
                     choices=["light", "medium", "strong"],
                     help="Training augmentation intensity (light / medium / strong).")
    dat.add_argument("--image-size", nargs=2, type=int, default=[512, 512],
                     help="Resize all images and masks to HEIGHT WIDTH (e.g. 512 512). "
                          "Both values must be divisible by 32 (UNet requirement).")

    mdl = p.add_argument_group("Model")
    mdl.add_argument("--encoder", type=str, default=None,
                     help="Encoder name for smp.Unet (e.g. resnet34). When set, uses pretrained encoder backbone.")
    mdl.add_argument("--init-features", type=int, default=32,
                     help="Base channel width of the from-scratch UNet. "
                          "32 -> encoder ends at 512, decoder at 32; "
                          "64 -> encoder ends at 1024, decoder at 64. "
                          "Ignored when --encoder is set.")
    mdl.add_argument("--encoder-weights", type=str, default=None,
                     help="Encoder weights source (e.g. imagenet). Ignored if --encoder is not set.")
    mdl.add_argument("--freeze-backbone", action=argparse.BooleanOptionalAction, default=False,
                     help="Freeze the encoder backbone so only the heads are trained. "
                          "Only effective when --encoder is set.")
    mdl.add_argument("--weight-loss", type=float, nargs=2, default=[0.5, 0.5],
                     help="Segmentation and classification loss weights (must sum to 1.0). "
                          "Only used with --task multitask.")
    mdl.add_argument("--cls-loss-func", type=str, default="smooth_bce",
                     choices=["smooth_bce", "ce", "focal"],
                     help="Loss function for the classification head.")
    mdl.add_argument("--segm-loss-func", type=str, default="dice",
                     choices=["dice", "bce", "focal"],
                     help="Loss function for the segmentation head.")
    mdl.add_argument("--label-smoothing", type=float, default=0.05,
                     help="Label-smoothing epsilon for the classification loss.")
    mdl.add_argument("--focal-gamma", type=float, default=2.0,
                     help="Gamma exponent for the focal loss (classification and segmentation).")
    mdl.add_argument("--layers", nargs="+", type=str, default=None,
                     help="Layer names for Grad-CAM target layers "
                          "(e.g. encoder.layer4 for SMP, bottleneck for UNet custom).")

    trn = p.add_argument_group("Training")
    trn.add_argument("--num-epochs", type=int, default=200, dest="nofepochs_or_patience",
                     help="Maximum number of training epochs (or patience when --early-stopping is set).")
    trn.add_argument("--early-stopping", action="store_true",
                     help="If set, training stops after num-epochs epochs without val loss improvement.")
    trn.add_argument("--learning-rate", type=float, default=1e-4,
                     help="Initial learning rate for AdamW.")
    trn.add_argument("--lr-patience", type=int, default=10,
                     help="Epochs without val loss improvement before LR is reduced.")
    trn.add_argument("--lr-factor", type=float, default=0.3,
                     help="Multiplicative factor for LR reduction.")
    trn.add_argument("--store-model-frequency", type=int, default=-1,
                     help="Save intermediate models every N epochs. -1 disables it.")
    trn.add_argument("--batch-size", type=int, default=16,
                     help="Number of images per training/evaluation step.")
    trn.add_argument("--num-workers", type=int, default=8,
                     help="Number of DataLoader worker processes for data loading.")

    return p


def _validate_args(args: argparse.Namespace) -> None:
    """Check consistency of parsed arguments; exit on error."""
    task = args.task

    needs_masks = task in ("segmentation", "multitask")
    needs_labels = task in ("classification", "multitask")

    if needs_masks and args.lung_segm_root_dir is None:
        print("ERROR: --masks-folder is required for --task %s!" % task)
        print("Exiting the program...")
        exit(1)

    if needs_labels and args.classification_csv_file_path is None:
        print("ERROR: --classification-csv is required for --task %s!" % task)
        print("Exiting the program...")
        exit(1)

    if needs_labels:
        if args.class_names is not None and len(args.class_names) != args.num_classes:
            print(f"ERROR: --class-names has {len(args.class_names)} entries but --num-classes is {args.num_classes}.")
            print("Exiting the program...")
            exit(1)
        try:
            df = pd.read_csv(args.classification_csv_file_path)
            label_col = args.classification_label_column
            if label_col not in df.columns:
                print(f"ERROR: column {label_col!r} was not found in {args.classification_csv_file_path}.")
                print("Exiting the program...")
                exit(1)
            n_labels = int(df[label_col].dropna().nunique())
            if n_labels != args.num_classes:
                print(f"ERROR: --num-classes is {args.num_classes} but {args.classification_csv_file_path} "
                      f"has {n_labels} distinct labels.")
                print("Exiting the program...")
                exit(1)
        except FileNotFoundError:
            print(f"ERROR: classification CSV not found: {args.classification_csv_file_path}")
            print("Exiting the program...")
            exit(1)

    if task == "multitask":
        w = args.weight_loss
        if abs(w[0] + w[1] - 1.0) > 1e-6:
            print("ERROR: --weight-loss values must sum to 1.0.")
            print("Exiting the program...")
            exit(1)

    if args.cls_loss_func == 'ce' and args.num_classes == 2:
        print("ERROR: --cls-loss-func ce requires more than 2 classes. "
              "Use smooth_bce for binary classification.")
        print("Exiting the program...")
        exit(1)

    size = args.image_size
    if any(s % 32 != 0 for s in size):
        print(f"ERROR: --image-size values must be divisible by 32. Got {size}.")
        print("Exiting the program...")
        exit(1)

    total = args.train_pct + args.val_pct + args.test_pct
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"--train-per + --val-per + --test-per must sum to 1.0, got {total:.3f}."
        )


def main() -> None:
    """Parse args, validate, and run the full experiment."""
    args = _build_parser().parse_args()
    _validate_args(args)

    if args.seed is not None:
        set_seed(args.seed)

    timestamp = datetime.now().strftime("_%d-%m-%Y_%H%M%S")
    args.results_dir_root = os.path.join(
        "..", "Results", args.experiment_name + timestamp,
    )
    os.makedirs(args.results_dir_root, exist_ok=True)

    args.summary_dir = os.path.join(args.results_dir_root, 'Summary')
    os.makedirs(args.summary_dir, exist_ok=True)

    task = args.task
    args.screening_dir = os.path.join(args.results_dir_root, 'Summary', 'Screening')
    args.segmentation_dir = os.path.join(args.results_dir_root, 'Summary', 'Segmentation')
    if task in ("classification", "multitask"):
        os.makedirs(args.screening_dir, exist_ok=True)
    if task in ("segmentation", "multitask"):
        os.makedirs(args.segmentation_dir, exist_ok=True)

    log_path = os.path.join(args.results_dir_root, 'output.log')
    _log_file = open(log_path, 'a', encoding='utf-8')
    sys.stdout = _Tee(sys.stdout, _log_file)
    sys.stderr = _Tee(sys.stderr, _log_file)

    sep = "-" * 52
    print(f"\n{sep}")
    print(f"  Experiment : {args.experiment_name}")
    print(f"  Task       : {task}")
    print(f"  Results    : {args.results_dir_root}")
    print(f"  Summary    : {args.results_dir_root}/Summary")
    print(f"  Log        : {log_path}")
    print(f"{sep}\n")

    run_experiment(args)

    torch.cuda.empty_cache()
    print("\n+++ All iterations complete.\n")


if __name__ == "__main__":
    main()