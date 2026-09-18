"""Experiment orchestration: one training + test run per repetition.

Public functions
----------------
set_seed
    Make all relevant RNG sources deterministic.
run_repetition
    Single train-then-test repetition.
run_experiment
    Loop over repetitions, collect results.
"""
import os
import random

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from util_dataset import (HoldoutSplitting, NoDataAugmentation, DataAugmentation,
                          NoPreprocessing, Preprocessing, compute_normalization_stats)
from util_training import ClassificationOnlyTrainer, SegmentationOnlyTrainer, MultitaskTrainer, TrainingManager


def set_seed(seed: int) -> None:
    """Make all relevant RNG sources deterministic."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def _get_encoder_normalization(args):
    """Retrieve the mean/std preprocessing values for the chosen SMP encoder.

    Returns
    -------
    tuple
        ``(mean, std)`` as lists of length 3, or ``(None, None)`` when no
        encoder or encoder weights are set (stats are then computed from
        the training data).
    """
    if args.encoder is None or args.encoder_weights is None:
        return None, None
    from segmentation_models_pytorch.encoders import get_preprocessing_params
    try:
        prep = get_preprocessing_params(args.encoder, args.encoder_weights)
        return prep['mean'], prep['std']
    except (KeyError, ValueError):
        return None, None


def _print_configuration(args, enc_mean=None, enc_std=None, norm_source=None):
    """Print a condensed experiment configuration block."""
    if args.encoder is None:
        model_label = 'UNet custom [init_features=%d]' % args.init_features
    else:
        model_label = 'SMP %s [weights=%s]' % (
            args.encoder, args.encoder_weights if args.encoder_weights else 'scratch')
    if getattr(args, 'freeze_backbone', False):
        model_label += ' [freeze_backbone=True]'

    cls_loss = args.cls_loss_func
    if cls_loss == 'smooth_bce':
        cls_loss += ' [label_smoothing=%.2f]' % args.label_smoothing
    elif cls_loss == 'focal':
        cls_loss += ' [gamma=%.2f]' % args.focal_gamma
    segm_loss = args.segm_loss_func
    if segm_loss == 'focal':
        segm_loss += ' [gamma=%.2f]' % args.focal_gamma
    loss_weights = ''
    if args.task == 'multitask':
        loss_weights = ' / [weights=%.1f, %.1f]' % (args.weight_loss[0], args.weight_loss[1])

    class_weights = args.class_weights
    if class_weights is None:
        class_weights_str = 'none'
    else:
        class_weights_str = '[' + ', '.join('%.2f' % float(w) for w in class_weights.tolist()) + ']'

    normalization_str = ''
    if norm_source is not None and enc_mean is not None and enc_std is not None:
        mean_str = ', '.join('%.4f' % float(m) for m in enc_mean)
        std_str = ', '.join('%.4f' % float(s) for s in enc_std)
        normalization_str = ' | Normalization mean=[%s], std=[%s] [%s]' % (mean_str, std_str, norm_source)

    print('\n+++ CONFIGURATION')
    print('[Experiment]  %s  [reps=%d, seed=%s]' % (args.experiment_name, args.nofrepetitions, args.seed))
    print('[Class weights] %s' % class_weights_str)
    print('[Model] %s  [num_classes=%d]' % (model_label, args.num_classes))
    print('[Augmentation] %s with albumentations [%s]' % (args.data_augmentation, args.aug_strength))
    print('[Preprocessing] %s [image_size=%dx%d]%s' % (
        args.preprocessing, args.image_size[0], args.image_size[1], normalization_str))
    print('[Losses] cls=%s / segm=%s%s' % (cls_loss, segm_loss, loss_weights))
    print('[Training] [epochs=%d, batch=%d, workers=%d]' % (
        args.nofepochs_or_patience, args.batch_size, args.num_workers))
    print('[Optimizer] AdamW [lr=%.1e, weight_decay=1e-4]' % args.learning_rate)
    print('[Scheduler] ReduceLROnPlateau [patience=%d, factor=%.1f]  early_stopping=[%s]' % (
        args.lr_patience, args.lr_factor, args.early_stopping))
    if args.task in ('classification', 'multitask'):
        print('[Grad-CAM] layers=[%s]' % (args.layers if args.layers else 'auto-resolve'))


def _get_trainer(task):
    """Return the trainer instance matching the requested task."""
    if task == 'classification':
        return ClassificationOnlyTrainer()
    if task == 'segmentation':
        return SegmentationOnlyTrainer()
    return MultitaskTrainer()


def run_repetition(args, current_repetition, results_dir):
    """Run one full train + test repetition.

    Parameters
    ----------
    args:
        Parsed CLI arguments.
    current_repetition:
        Zero-based repetition index.
    results_dir:
        Root output directory for this experiment.
    """
    print('')
    print('>> Starting repetition %d/%d...' % (current_repetition + 1, args.nofrepetitions))

    args.rep_dir = os.path.join(args.results_dir_root, 'Rep%d' % (current_repetition + 1))
    args.rep_screening_dir = os.path.join(args.rep_dir, 'Screening')
    args.rep_segmentation_dir = os.path.join(args.rep_dir, 'Segmentation')
    os.makedirs(args.rep_dir, exist_ok=True)
    if args.task in ('classification', 'multitask'):
        os.makedirs(args.rep_screening_dir, exist_ok=True)
    if args.task in ('segmentation', 'multitask'):
        os.makedirs(args.rep_segmentation_dir, exist_ok=True)

    methodology = _get_trainer(args.task)

    split_seed = args.seed if args.seed is not None else 0
    splitting_obj = HoldoutSplitting(args.train_pct, args.val_pct, args.test_pct)
    splitting_obj.load_dataset_with_random_shuffling(
        args, iteration=current_repetition, seed=split_seed,
    )

    class_weights = splitting_obj.get_class_weights()
    args.class_weights = torch.tensor(class_weights, dtype=torch.float32) if class_weights is not None else None

    training_manager_obj = TrainingManager()

    model = methodology.init_model(args)

    data_augmentation_class = globals().get(args.data_augmentation, NoDataAugmentation)
    data_augmentation_obj = data_augmentation_class(aug_strength=args.aug_strength)

    enc_mean, enc_std = _get_encoder_normalization(args)
    norm_source = None
    preprocessing_class = globals().get(args.preprocessing, Preprocessing)
    if preprocessing_class is Preprocessing:
        if enc_mean is None or enc_std is None:
            enc_mean, enc_std = compute_normalization_stats(
                args.input_dir_root, splitting_obj.get_training_subset(), args.image_size,
            )
            norm_source = 'computed from train set'
        else:
            norm_source = 'ImageNet'
        preprocessing_method_obj = preprocessing_class(mean=enc_mean, std=enc_std)
    else:
        preprocessing_method_obj = preprocessing_class()

    _print_configuration(args, enc_mean=enc_mean, enc_std=enc_std, norm_source=norm_source)

    methodology.train(args, current_repetition, model, args.input_dir_root,
                      training_manager_obj, data_augmentation_obj, splitting_obj,
                      preprocessing_method_obj)

    # Evaluate the best model of the repetition.
    best_model_path = os.path.join(args.rep_dir, 'Models', 'best.pkl')
    if os.path.exists(best_model_path):
        model = methodology.load_model(model, best_model_path)
        print('Loaded best model for testing: %s' % best_model_path)
    else:
        print('WARNING: best model not found (%s) — testing with the last model.' % best_model_path)

    print('')
    print('>> Testing on held-out test set (rep %d)...' % (current_repetition + 1))
    test_subset = splitting_obj.get_test_subset()
    methodology.test(args, current_repetition_number=current_repetition, model=model,
                     input_dir_root=args.input_dir_root, input_dataset=test_subset,
                     preprocessing_method_obj=preprocessing_method_obj)

    if args.task in ('classification', 'multitask'):
        print('')
        print('Generating Grad-CAMs on the test set (rep %d)...' % (current_repetition + 1))
        methodology.get_gradcam_map(args, current_repetition_number=current_repetition, model=model,
                                    input_dir_root=args.input_dir_root, splitting_obj=splitting_obj,
                                    preprocessing_method_obj=preprocessing_method_obj)


def _read_loss_logs(args, n_repetitions):
    """Read the per-repetition train/val loss logs as DataFrames."""
    logs = []
    for rep in range(n_repetitions):
        path = os.path.join(args.results_dir_root, 'Rep%d' % (rep + 1), 'loss_log.csv')
        if os.path.exists(path):
            logs.append(pd.read_csv(path))
    return logs


def _matrix_from_logs(logs, column):
    """``(repetitions, min_epochs)`` float matrix from a loss-log column.

    Values that are not numeric (e.g. ``'N/A'``) become NaN. Rows are
    truncated to the shortest run so every repetition contributes at every
    epoch index (early stopping may make runs differ in length).
    """
    if not logs:
        return None
    min_epochs = min(len(df) for df in logs)
    values = []
    for df in logs:
        col = pd.to_numeric(df[column].head(min_epochs), errors='coerce').to_numpy(dtype=float)
        values.append(col)
    return np.stack(values)


def _plot_mean_std_curve(train_matrix, val_matrix, title, ylabel, save_path, train_label='Train', val_label='Validation'):
    """Plot per-epoch mean ± std across repetitions."""
    if train_matrix is None:
        return
    epochs = np.arange(1, train_matrix.shape[1] + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)

    for matrix, color, label in ((train_matrix, 'r', train_label), (val_matrix, 'b', val_label)):
        if matrix is None:
            continue
        mean = np.nanmean(matrix, axis=0)
        std = np.nanstd(matrix, axis=0)
        ax.plot(epochs, mean, color, linewidth=2, label=label)
        ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.2)

    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)


def _save_aggregated_figures(args, n_repetitions):
    """Save mean ± std training curves for the active tasks."""
    logs = _read_loss_logs(args, n_repetitions)
    if not logs:
        return

    has_seg = args.task in ('segmentation', 'multitask')
    has_cls = args.task in ('classification', 'multitask')

    if has_cls:
        os.makedirs(args.screening_dir, exist_ok=True)
        screening_pairs = [
            ('train_loss', 'val_loss', 'Training vs Validation Loss (mean ± std)', 'Loss',
             'training_validation_loss_mean_std.png'),
            ('train_accuracy', 'val_accuracy',
             'Training vs Validation Accuracy (mean ± std)', 'Accuracy',
             'training_validation_accuracy_mean_std.png'),
        ]
        for train_col, val_col, title, ylabel, fname in screening_pairs:
            _plot_mean_std_curve(
                _matrix_from_logs(logs, train_col),
                _matrix_from_logs(logs, val_col),
                title, ylabel,
                os.path.join(args.screening_dir, fname),
            )

        os.makedirs(args.summary_dir, exist_ok=True)
        if args.task == 'multitask':
            _plot_mean_std_curve(
                _matrix_from_logs(logs, 'train_loss'),
                _matrix_from_logs(logs, 'val_loss'),
                'Joint Loss (segmentation + classification) mean ± std', 'Joint Loss',
                os.path.join(args.summary_dir, 'training_validation_joint_loss_mean_std.png'),
            )

    if has_seg:
        os.makedirs(args.segmentation_dir, exist_ok=True)
        pairs = [
            ('train_loss', 'val_loss', 'Training vs Validation Loss (mean ± std)', 'Loss',
             'training_validation_loss_mean_std.png'),
            ('train_seg_accuracy', 'val_seg_accuracy',
             'Training vs Validation Accuracy (mean ± std)', 'Pixel accuracy',
             'training_validation_accuracy_mean_std.png'),
            ('train_dice', 'val_dice',
             'Training vs Validation Dice (mean ± std)', 'Dice',
             'training_validation_dice_mean_std.png'),
            ('train_iou', 'val_iou',
             'Training vs Validation Jaccard (mean ± std)', 'Jaccard',
             'training_validation_jaccard_mean_std.png'),
            ('train_precision', 'val_precision',
             'Training vs Validation Precision (mean ± std)', 'Precision',
             'training_validation_precision_mean_std.png'),
            ('train_recall', 'val_recall',
             'Training vs Validation Recall (mean ± std)', 'Recall',
             'training_validation_recall_mean_std.png'),
        ]
        for train_col, val_col, title, ylabel, fname in pairs:
            _plot_mean_std_curve(
                _matrix_from_logs(logs, train_col),
                _matrix_from_logs(logs, val_col),
                title, ylabel,
                os.path.join(args.segmentation_dir, fname),
            )


def _save_test_metrics_summary(args, n_repetitions):
    """Aggregate test metrics across repetitions into ``Summary/<task>/test_metrics_mean_std.csv``."""
    has_seg = args.task in ('segmentation', 'multitask')
    has_cls = args.task in ('classification', 'multitask')

    base_metrics = ('precision', 'recall', 'f1', 'accuracy', 'roc_auc')
    averages = ('micro', 'macro', 'weighted')
    screening_metrics = ['%s_%s' % (metric, average) for average in averages for metric in base_metrics]
    per_rep = []
    for rep in range(n_repetitions):
        entry = {}
        if has_cls:
            metrics_path = os.path.join(args.results_dir_root, 'Rep%d' % (rep + 1), 'Screening', 'test_metrics.csv')
            if os.path.exists(metrics_path):
                frame = pd.read_csv(metrics_path)
                for _, row in frame.iterrows():
                    average = row['average']
                    for metric in base_metrics:
                        entry['%s_%s' % (metric, average)] = row.get(metric, np.nan)
        if has_seg:
            table_path = os.path.join(args.results_dir_root, 'Rep%d' % (rep + 1), 'Segmentation', 'test_table.csv')
            if os.path.exists(table_path):
                table = pd.read_csv(table_path)
                mean_row = table[table['image'] == 'Mean']
                if len(mean_row) > 0:
                    for metric in ('dice', 'iou', 'precision', 'recall', 'accuracy'):
                        entry['seg_%s' % metric] = float(mean_row[metric].iloc[0])
        per_rep.append(entry)

    if not per_rep:
        return

    metric_names = []
    if has_cls:
        metric_names += screening_metrics
    if has_seg:
        metric_names += ['seg_dice', 'seg_iou', 'seg_precision', 'seg_recall', 'seg_accuracy']

    import csv
    os.makedirs(args.summary_dir, exist_ok=True)
    with open(os.path.join(args.summary_dir, 'test_metrics_mean_std.csv'), 'w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        writer.writerow(['metric', 'mean', 'std'])
        for metric in metric_names:
            v = np.array([m.get(metric, np.nan) for m in per_rep], dtype=float)
            v = v[np.isfinite(v)]
            writer.writerow([
                metric,
                '%.6f' % v.mean() if v.size > 0 else 'nan',
                '%.6f' % v.std() if v.size > 0 else 'nan',
            ])


def _save_train_val_xlsx(args, n_repetitions):
    """Save per-repetition train/val metrics as Excel files (one sheet per metric).

    Only written when segmentation is active (mirrors ``src_segmentation``'s
    ``repTrainMetrics.xlsx`` / ``repValidationMetrics.xlsx``).
    """
    if args.task not in ('segmentation', 'multitask'):
        return
    logs = _read_loss_logs(args, n_repetitions)
    if not logs:
        return

    metric_columns = {
        'Loss': ('train_loss', 'val_loss'),
        'Jaccard': ('train_iou', 'val_iou'),
        'Dice': ('train_dice', 'val_dice'),
        'Accuracy': ('train_seg_accuracy', 'val_seg_accuracy'),
        'Precision': ('train_precision', 'val_precision'),
        'Recall': ('train_recall', 'val_recall'),
    }
    os.makedirs(args.segmentation_dir, exist_ok=True)

    for sheet_name, (train_col, val_col) in metric_columns.items():
        train_frames = {}
        val_frames = {}
        for rep in range(n_repetitions):
            path = os.path.join(args.results_dir_root, 'Rep%d' % (rep + 1), 'loss_log.csv')
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path)
            train_frames[str(rep)] = pd.to_numeric(df[train_col], errors='coerce').to_numpy(dtype=float)
            val_frames[str(rep)] = pd.to_numeric(df[val_col], errors='coerce').to_numpy(dtype=float)
        if not train_frames:
            continue

        train_df = pd.DataFrame({rep: pd.Series(v) for rep, v in train_frames.items()})
        val_df = pd.DataFrame({rep: pd.Series(v) for rep, v in val_frames.items()})
        train_df.index = np.arange(1, len(train_df) + 1)
        val_df.index = np.arange(1, len(val_df) + 1)
        train_df.index.name = 'epoch'
        val_df.index.name = 'epoch'

        train_path = os.path.join(args.segmentation_dir, 'train_metrics.xlsx')
        val_path = os.path.join(args.segmentation_dir, 'val_metrics.xlsx')
        mode = 'a' if os.path.exists(train_path) else 'w'
        with pd.ExcelWriter(train_path, mode=mode, engine='openpyxl' if mode == 'a' else None) as writer:
            train_df.to_excel(writer, sheet_name=sheet_name, index=True)
        mode = 'a' if os.path.exists(val_path) else 'w'
        with pd.ExcelWriter(val_path, mode=mode, engine='openpyxl' if mode == 'a' else None) as writer:
            val_df.to_excel(writer, sheet_name=sheet_name, index=True)


def run_experiment(args):
    """Iterate over all repetitions, training and testing each."""
    if args.seed is not None:
        set_seed(args.seed)

    os.makedirs(args.results_dir_root, exist_ok=True)

    for current_repetition in range(args.nofrepetitions):
        run_repetition(args, current_repetition, args.results_dir_root)

    print('')
    print('Computing aggregate results...')
    _save_aggregated_figures(args, args.nofrepetitions)
    _save_test_metrics_summary(args, args.nofrepetitions)
    _save_train_val_xlsx(args, args.nofrepetitions)

    print('')
    print('+++ All repetitions completed.')
    print('Results saved in: %s' % args.results_dir_root)