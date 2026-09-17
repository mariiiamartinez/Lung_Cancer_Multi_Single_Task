"""Metric computation and persistence for both tasks.

- Classification: :func:`store_classification_results` writes the confusion
  matrix, per-image predictions and test metrics for each repetition.
- Segmentation: :func:`compute_segmentation_metrics` computes per-image
  dice/iou/precision/recall/accuracy, and :func:`store_segmentation_metrics_table`
  persists them into the per-repetition ``test_table.csv``.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def store_classification_results(y_true, y_pred, y_prob, rows, class_names, num_classes, results_dir):
    """Save the confusion matrix, test metrics and per-image predictions.

    ``results_dir`` is the per-repetition Screening folder, so filenames do
    not need a repetition suffix.
    """
    if class_names is None:
        class_names = ['class_%d' % i for i in range(num_classes)]
    labels = list(range(num_classes))
    os.makedirs(results_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=labels,
        display_labels=class_names,
        cmap=plt.cm.Blues,
        ax=ax,
        xticks_rotation=45,
    )
    ax.set_title('Confusion matrix')
    fig.tight_layout()
    fig.savefig('%s/CM.png' % results_dir)
    plt.close(fig)

    metrics_rows = []
    for average in ('micro', 'macro', 'weighted'):
        metrics_rows.append({
            'average': average,
            'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
            'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
            'f1': f1_score(y_true, y_pred, average=average, zero_division=0),
            'accuracy': accuracy_score(y_true, y_pred),
            'roc_auc': np.nan,
        })
    try:
        if num_classes == 2:
            auc = roc_auc_score(y_true, np.asarray(y_prob)[:, 1])
        else:
            auc = roc_auc_score(y_true, np.asarray(y_prob), multi_class='ovo')
        for row in metrics_rows:
            row['roc_auc'] = auc
    except Exception as exc:
        print('ROC-AUC could not be computed: %s' % exc)

    print(classification_report(y_true, y_pred, labels=labels, target_names=class_names, zero_division=0))
    pd.DataFrame(metrics_rows).to_csv(
        '%s/test_metrics.csv' % results_dir,
        index=False, na_rep='nan',
    )

    prob_columns = ['prob_%s' % class_name for class_name in class_names]
    pd.DataFrame(rows, columns=['image_name', 'true_label', 'pred_label'] + prob_columns).to_csv(
        '%s/predictions.csv' % results_dir,
        index=False,
    )


def compute_segmentation_metrics(pred_mask, gt_mask, threshold=0.5):
    """Compute per-image segmentation metrics.

    Thresholds both masks at ``threshold`` and returns a dict with dice,
    IoU, precision, recall and accuracy. The default ``threshold=0.5``
    matches the standard operating point used in SMP's built-in metrics.
    Note: the saved binary masks/overlays use a higher threshold (0.7).
    """
    p = (pred_mask > threshold).flatten().cpu().numpy().astype(bool)
    t = (gt_mask > threshold).flatten().cpu().numpy().astype(bool)

    tp = (p & t).sum()
    fp = (p & ~t).sum()
    fn = (~p & t).sum()
    tn = (~p & ~t).sum()

    dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)

    return {'dice': float(dice), 'iou': float(iou),
            'precision': float(precision), 'recall': float(recall),
            'accuracy': float(accuracy)}


def store_segmentation_metrics_table(metrics_rows, rep_dir):
    """Save per-image test metrics to ``test_table.csv`` inside the repetition folder."""
    if not metrics_rows:
        return
    df = pd.DataFrame(metrics_rows)
    df = df[['image'] + [column for column in df.columns if column != 'image']]
    stats = pd.DataFrame([{
        'image': 'Mean',
        **{k: df[k].mean() for k in df.columns if k != 'image'},
    }, {
        'image': 'Std',
        **{k: df[k].std() for k in df.columns if k != 'image'},
    }])
    df = pd.concat([df, stats], ignore_index=True)
    os.makedirs(rep_dir, exist_ok=True)
    df.to_csv(os.path.join(rep_dir, 'test_table.csv'), index=False)
