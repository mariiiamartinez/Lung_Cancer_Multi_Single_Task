"""Utility functions for lung segmentation / classification / multi-task training.

This module provides core training utilities including loss functions
(DiceLoss, FocalLoss, smoothed BCE), training orchestration (TrainingManager),
Grad-CAM visualisation support (GradCAMMapManager), and the trainers that
drive single-task classification, single-task segmentation and the joint
multi-task pipeline.
"""

import os
import pathlib
import random
import sys
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.utils as vutils

from util_metrics import (
    compute_segmentation_metrics,
    store_classification_results,
    store_segmentation_metrics_table,
)
from util_dataset import LungDataset
from util_models import UNetForSegmentationClassification, SMPUNetForSegmentationClassification

# Binarisation threshold for the predicted segmentation mask, matching the
# single-task pipeline (src_segmentation used 0.7). Kept high so the soft
# low-confidence rim around object edges does not add speckle to the overlays.
# This applies only to *saved* masks/overlays; segmentation *metrics* use
# their own threshold (0.5) in compute_segmentation_metrics (util_metrics).
MASK_THRESHOLD = 0.7


def _write_progress(current, total, prefix=''):
    """Overwrite the current terminal line with batch progress."""
    bar_len = 30
    filled = int(bar_len * current / total)
    bar = '#' * filled + ' ' * (bar_len - filled)
    sys.stdout.write(f'\r{prefix}  [{bar}] {current}/{total}')
    sys.stdout.flush()


class DiceLoss(nn.Module):
    """Dice loss for binary segmentation tasks.

    Computes the soft Dice loss (1 - Dice similarity coefficient)
    which is commonly used for imbalanced segmentation problems.
    """

    def __init__(self):
        """Initialise DiceLoss with a smoothing factor of 1.0."""
        super(DiceLoss, self).__init__()
        self.smooth = 1.0

    def dice_loss(self, y_pred, y_true):
        """Compute the soft Dice loss between predicted and target segmentation maps.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted segmentation map (sigmoid output).
        y_true : torch.Tensor
            Ground-truth segmentation map.

        Returns
        -------
        torch.Tensor
            Soft Dice loss value (1 - DSC).
        """
        assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dsc = (2.0 * intersection + self.smooth) / (
            y_pred.sum() + y_true.sum() + self.smooth
        )
        return 1.0 - dsc

    def get_loss_function(self):
        """Return the dice_loss callable for use as a loss function."""
        return self.dice_loss


def bce_segmentation_loss(y_pred, y_true):
    """Binary cross-entropy on the sigmoid segmentation output."""
    return F.binary_cross_entropy(y_pred, y_true)


class SegmentationFocalLoss(nn.Module):
    """Focal loss for binary segmentation, applied to the sigmoid output.

    The segmentation head already emits probabilities in ``[0, 1]``, so the
    focal weighting is computed directly on those probabilities (no second
    sigmoid like the classification :class:`FocalLoss`).
    """

    def __init__(self, gamma=2.0):
        """Initialise the loss with a gamma exponent."""
        super(SegmentationFocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, y_pred, y_true):
        """Compute the focal loss on the probability mask."""
        assert y_pred.size() == y_true.size()
        y_pred_flat = y_pred[:, 0].contiguous().view(-1)
        y_true_flat = y_true[:, 0].contiguous().view(-1)
        pt = y_pred_flat * y_true_flat + (1 - y_pred_flat) * (1 - y_true_flat)
        ce = F.binary_cross_entropy(y_pred_flat, y_true_flat, reduction='none')
        return ((1 - pt) ** self.gamma * ce).mean()


class SmoothBCEWithLogitsLoss(nn.Module):
    """Binary cross-entropy with optional label smoothing.

    Mirrors the label-smoothed binary loss used in ``src_screening``.  Raw
    targets (0/1) are smoothed to ``(1 - eps)/2 + eps * target`` before the
    logits are passed through :func:`torch.nn.functional.binary_cross_entropy_with_logits`.
    """

    def __init__(self, smoothing=0.0, class_weights=None):
        """Initialise the loss with a smoothing factor in [0, 1) and optional per-class weights."""
        super(SmoothBCEWithLogitsLoss, self).__init__()
        self.smoothing = smoothing
        self.class_weights = class_weights

    def forward(self, logits, target):
        """Compute the smoothed BCE loss.

        Parameters
        ----------
        logits : torch.Tensor
            Raw model logits of shape (B, 1).
        target : torch.Tensor
            Binary targets of shape (B, 1) with values in {0, 1}.

        Returns
        -------
        torch.Tensor
            Scalar smoothed BCE loss.
        """
        target_smoothed = target
        if self.smoothing > 0:
            target_smoothed = target * (1 - self.smoothing) + 0.5 * self.smoothing
        if self.class_weights is not None:
            sample_weights = self.class_weights.to(logits.device)[target.long()]
            return F.binary_cross_entropy_with_logits(logits, target_smoothed, weight=sample_weights)
        return F.binary_cross_entropy_with_logits(logits, target_smoothed)


class FocalLoss(nn.Module):
    """Focal loss supporting binary (single-logit) and multiclass targets.

    Binary inputs ([B, 1] logits or [B, 1, H, W] segmentation masks) use the
    sigmoid formulation; multiclass inputs ([B, C] logits) use the softmax
    formulation.
    """

    def __init__(self, gamma=2.0, class_weights=None):
        """Initialise the loss with a gamma exponent and optional per-class weights."""
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.class_weights = class_weights

    def forward(self, logits, targets):
        """Compute the focal loss.

        Parameters
        ----------
        logits : torch.Tensor
            Raw model logits (B, 1), (B, C) or a spatial mask (B, 1, H, W).
        targets : torch.Tensor
            Ground-truth targets matching the logits shape (classes or pixels).

        Returns
        -------
        torch.Tensor
            Scalar focal loss.
        """
        if logits.size(1) == 1:
            probs = torch.sigmoid(logits).view(-1)
            targets_flat = targets.float().view(-1)
            ce = F.binary_cross_entropy_with_logits(logits.view(-1), targets_flat, reduction='none')
            p_t = probs * targets_flat + (1 - probs) * (1 - targets_flat)
            losses = (1 - p_t) ** self.gamma * ce
            if self.class_weights is not None:
                sample_weights = self.class_weights.to(logits.device)[targets.long().view(-1)]
                losses = losses * sample_weights
            return losses.mean()
        ce_loss = F.cross_entropy(logits, targets, weight=self.class_weights, reduction='none')
        p_t = torch.exp(-ce_loss)
        return ((1 - p_t) ** self.gamma * ce_loss).mean()


class TrainingManager:
    """Orchestrates the training loop state including early stopping logic.

    Tracks the current epoch, validation loss, and epochs without
    improvement to support early-stopping and periodic model storage.
    """

    def __init__(self):
        """Initialise training state counters and best-loss tracker."""
        self.current_epoch = 0
        self.best_val_loss = None
        self.epochs_without_improvement = 0
        self.best_epoch_so_far = 0

    def get_current_epoch(self):
        """Return the current epoch number."""
        return self.current_epoch

    def show_final_training_report(self, args):
        """Print the best and final epoch numbers."""
        print('')
        print('Best epoch: %d' % self.best_epoch_so_far)
        print('Final epoch: %d' % self.current_epoch)

    def update_training_manager(self, args, current_repetition, model, val_loss, save_model_func):
        """Update training state after an epoch and save best model if validation loss improved.

        The best model (by validation loss) is always persisted per
        repetition, regardless of whether early stopping is enabled.  The
        ``epochs_without_improvement`` counter is only used when early
        stopping is active.

        Parameters
        ----------
        args : Namespace
            Experiment configuration arguments.
        current_repetition : int
            Current repetition index.
        model : nn.Module
            The model being trained.
        val_loss : float or None
            Validation loss for the current epoch.
        save_model_func : callable
            Function to persist the model state dict.
        """
        self.current_epoch += 1

        if val_loss is None:
            if args.early_stopping:
                self.epochs_without_improvement += 1
            return

        if self.best_val_loss is None or val_loss <= self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch_so_far = self.current_epoch
            model_path_to_save = os.path.join(
                args.rep_dir, 'Models', 'best.pkl',
            )
            os.makedirs(os.path.dirname(model_path_to_save), exist_ok=True)
            save_model_func(model, model_path_to_save)
            if args.early_stopping:
                self.epochs_without_improvement = 0
        else:
            if args.early_stopping:
                self.epochs_without_improvement += 1

    def continue_training(self, args):
        """Determine whether training should continue based on early stopping or max epochs."""
        if args.early_stopping:
            return self.epochs_without_improvement < args.nofepochs_or_patience
        return self.current_epoch < args.nofepochs_or_patience

    def store_model(self, args, model, save_model_func):
        """Periodically persist the model to disk based on store_model_frequency."""
        if args.store_model_frequency == -1:
            return
        if self.current_epoch != 0 and self.current_epoch % args.store_model_frequency == 0:
            model_path_to_save = os.path.join(
                args.rep_dir, 'trained_model_epoch_%d.pkl' % self.current_epoch,
            )
            save_model_func(model, model_path_to_save)


class GradCAMMapManager:
    """Manages generation of Grad-CAM visualisation maps (standard GradCAM).

    Only the basic Grad-CAM variant is produced, matching the single-task
    screening pipeline. The segmentation mask is a by-product of the same
    forward pass; the heatmap always targets the classification output.
    """

    class ClassificationTargetForGradCAM:
        """Target class that selects a class index from the classification output.

        Parameters
        ----------
        class_idx : int, optional
            Index of the target class. If None, uses the maximum activation.
        """

        def __init__(self, class_idx=None):
            self.class_idx = class_idx

        def __call__(self, model_output):
            if isinstance(model_output, tuple):
                model_output = model_output[1]
            if model_output.dim() == 1:
                model_output = model_output.view(1, -1)
            if self.class_idx is None:
                return model_output.max()
            if model_output.size(1) == 1:
                # Binary head: a single logit encodes the positive class.
                # For the positive class (index 1) maximise the logit; for the
                # negative class (index 0) maximise P(class 0) = 1 - sigmoid(logit),
                # i.e. minimise the logit.
                if self.class_idx == 0:
                    return (-model_output).sum()
                return model_output.sum()
            return model_output[:, self.class_idx].sum()

    def _get_gradcam_map_loop(self, args, gradcam_obj, input_dir_root, input_subset, preprocessing_method_obj, alpha=0.6):
        """Iterate over a dataset subset and save Grad-CAM heatmaps to disk."""
        subset_key, subset_value = input_subset
        gradcam_dir = os.path.join(args.rep_screening_dir, 'Gradcams_Test')
        os.makedirs(gradcam_dir, exist_ok=True)

        for current_input_image_name in subset_value:
            current_input_image = self._load_input_image(
                input_dir_root, current_input_image_name, preprocessing_method_obj, args.image_size,
            ).unsqueeze_(0).cuda()

            # Display image: raw (un-normalised) [0, 1] tensor, so the overlay
            # looks natural (mirrors the single-task screening pipeline).
            input_image_np = self._load_input_image_raw(
                input_dir_root, current_input_image_name, args.image_size,
            ).numpy().transpose((1, 2, 0))

            # Target the predicted class of this image (as in the single task).
            model_output = gradcam_obj.model(current_input_image)
            classification_output = model_output[1] if isinstance(model_output, tuple) else model_output
            if args.num_classes == 2:
                binary_probs = torch.sigmoid(classification_output)
                pred_class = int((binary_probs > 0.5).long().item())
            else:
                pred_class = int(torch.argmax(classification_output, dim=1).item())

            grayscale_gradcam = gradcam_obj(
                input_tensor=current_input_image,
                targets=[self.ClassificationTargetForGradCAM(pred_class)],
            )[0, :]
            visualization = show_cam_on_image(input_image_np, grayscale_gradcam, use_rgb=True)

            current_file_extension = pathlib.Path(current_input_image_name).suffix
            output_image_name = current_input_image_name.replace(
                current_file_extension,
                '_gradcam' + current_file_extension,
            )
            output_image_path = os.path.join(gradcam_dir, subset_key + '_' + output_image_name)
            plt.imsave(output_image_path, visualization)

    def _resolve_target_layers(self, model, layer_names):
        """Resolve Grad-CAM target layers from dotted module names.

        Each name is walked through ``model._modules`` (e.g. ``encoder.encoder5``).
        If the resolved module is a ``nn.Sequential`` its last sub-module is
        used, otherwise the module itself.
        """
        target_layers = []
        for layer_name in layer_names:
            module = model
            for part in layer_name.split('.'):
                module = module._modules[part]
            if isinstance(module, nn.Sequential) and len(module) > 0:
                module = module[-1]
            target_layers.append(module)
        return target_layers

    @staticmethod
    def _default_gradcam_layer(model):
        """Pick the last encoder layer as Grad-CAM target.

        SMP models: encoder.layer4 (last ResNet stage).
        Custom UNet: bottleneck (last encoder output).
        """
        if hasattr(model, 'encoder'):
            return 'encoder.layer4'
        return 'bottleneck'

    def get_gradcam_map(self, args, current_repetition_number, model, input_dir_root, splitting_obj, preprocessing_method_obj):
        """Compute and save standard Grad-CAM heatmaps for the test split.

        Heatmaps are saved into ``<rep>/Screening/Gradcams_Test`` and always
        use the basic Grad-CAM method on the classification output.
        """
        if args.layers:
            default_layers = args.layers
        else:
            default_layers = [self._default_gradcam_layer(model)]
        target_layers = self._resolve_target_layers(model, default_layers)
        module_names = {id(m): name for name, m in model.named_modules()}
        resolved_names = [module_names.get(id(m), repr(m)) for m in target_layers]
        print('[GradCAM] The following layers will be used: %s' % resolved_names)
        gradcam_obj = GradCAM(model=model, target_layers=target_layers)

        subsets_dict = {
            'test': splitting_obj.get_test_subset(),
        }
        for subset_key, subset_value in subsets_dict.items():
            self._get_gradcam_map_loop(
                args,
                gradcam_obj,
                input_dir_root,
                (subset_key, subset_value),
                preprocessing_method_obj,
            )


class BaseTrainer(GradCAMMapManager):
    """Base trainer with the shared data/model/plotting pipeline.

    Subclasses set ``self.use_segmentation`` and ``self.use_classification``
    so the same code path serves single-task classification, single-task
    segmentation and the joint multi-task model.
    """

    def __init__(self):
        """Initialise task flags and cached classification metadata."""
        self.classification_targets = None
        self.class_names = None
        self.aug_strength = None

    # ---- Model lifecycle -------------------------------------------------

    def init_model(self, args):
        """Initialise the model, disabling the heads not required by the task.

        With ``args.encoder`` set, an SMP-based UNet is created; otherwise a
        custom from-scratch UNet is used. The ``args.freeze_backbone`` flag is
        applied later (in :meth:`train`) since it needs the optimizer.
        """
        use_seg = self.use_segmentation
        use_cls = self.use_classification
        if args.encoder is not None:
            return SMPUNetForSegmentationClassification(
                encoder_name=args.encoder,
                encoder_weights=args.encoder_weights,
                num_classes=args.num_classes,
                use_segmentation=use_seg,
                use_classification=use_cls,
            )
        return UNetForSegmentationClassification(
            in_channels=3, out_channels=1, num_classes=args.num_classes,
            init_features=args.init_features,
            use_segmentation=use_seg,
            use_classification=use_cls,
        )

    def save_model(self, model, path_to_save):
        """Persist model state dict to disk in both .pkl and .pth formats."""
        torch.save(model.state_dict(), path_to_save)
        pth_path = os.path.splitext(path_to_save)[0] + '.pth'
        torch.save(model.state_dict(), pth_path)

    def load_model(self, model, path_to_load):
        """Load model state dict from disk and return the model."""
        model.load_state_dict(torch.load(path_to_load))
        return model

    # ---- Data loading helpers -------------------------------------------

    def _segmentation_target_path(self, args, current_input_image_name):
        """Build the full path to the ground-truth segmentation mask for a given input image."""
        stem = current_input_image_name.rsplit('.', 1)[0]
        return '%s/%s.png' % (args.lung_segm_root_dir, stem)

    def _load_input_image_raw(self, input_dir_root, current_input_image_name, image_size=None):
        """Load and resize an input image from disk, returning a raw [0, 1] tensor.

        Unlike :meth:`_load_input_image`, no preprocessing is applied. This
        lets data augmentation operate on the raw image before normalisation,
        mirroring ``src_segmentation`` (augment raw, then preprocess).
        """
        from PIL import ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        full_input_image_path = '%s/%s' % (input_dir_root, current_input_image_name)
        input_image = Image.open(full_input_image_path).convert('RGB')
        if image_size is not None:
            if isinstance(image_size, (int, float)):
                size = (int(image_size), int(image_size))
            else:
                size = (int(image_size[0]), int(image_size[1]))
            input_image = TF.resize(input_image, size)
        return TF.to_tensor(input_image)

    def _load_input_image(self, input_dir_root, current_input_image_name, preprocessing_method_obj, image_size=None):
        """Load, resize, and preprocess an input image from disk."""
        input_image = self._load_input_image_raw(input_dir_root, current_input_image_name, image_size)
        return preprocessing_method_obj.apply_preprocessing_on_image(input_image)

    def _load_segmentation_target(self, args, current_input_image_name, image_size=None):
        """Load the ground-truth segmentation mask for the given input image."""
        target_image_full_path = self._segmentation_target_path(args, current_input_image_name)
        if not os.path.exists(target_image_full_path):
            print('WARNING! The target segmentation image %s was not found!' % target_image_full_path)
            return None

        target_image = Image.open(target_image_full_path)
        target_image = ImageOps.grayscale(target_image)
        if image_size is not None:
            if isinstance(image_size, (int, float)):
                size = (int(image_size), int(image_size))
            else:
                size = (int(image_size[0]), int(image_size[1]))
            target_image = TF.resize(target_image, size)
        target_image = TF.to_tensor(target_image)
        return target_image.unsqueeze_(0)

    def _original_image_size(self, image_path):
        """Return the ``(height, width)`` of the input image on disk."""
        with Image.open(image_path) as image:
            width, height = image.size
        return height, width

    def _save_prediction_overlaps(self, image_path, pred_mask_tensor, gt_mask_path, overlapped_dir, stem):
        """Save the prediction overlays following ``src_segmentation.utils.overlap_images``."""
        from PIL import ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        pred_np = pred_mask_tensor.squeeze().cpu().numpy()

        # -- Overlay 1: prediction contour on the original image --
        pred_bin = (pred_np > MASK_THRESHOLD).astype(np.uint8)
        contours, _ = cv2.findContours(pred_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        original = np.asarray(Image.open(image_path).convert('RGB'))
        orig_h, orig_w = original.shape[:2]
        thickness = max(3, min(orig_h, orig_w) // 250)
        overlay = original.copy()
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), thickness)
        Image.fromarray(overlay).save(os.path.join(overlapped_dir, '%s.png' % stem))

        # -- Overlay 2: prediction (green) over the ground-truth mask --
        gt_mask = cv2.imread(gt_mask_path)
        if gt_mask is None:
            print('WARNING! Could not read GT mask %s — skipping GT overlay.' % gt_mask_path)
            return
        pred_bin = (pred_np > MASK_THRESHOLD).astype(np.uint8)
        color_mask = np.zeros_like(gt_mask)
        color_mask[pred_bin == 1] = [0, 255, 0]
        overlapped = cv2.addWeighted(gt_mask, 0.5, color_mask, 1.0, 0.0)
        cv2.imwrite(os.path.join(overlapped_dir, '%s_mask.png' % stem), overlapped)

    # ---- Plotting --------------------------------------------------------

    def _plot_train_val_curve(self, df, train_col, val_col, save_path, title, ylabel):
        """Plot a train vs validation metric curve."""
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(df['epoch'].astype(float), df[train_col].astype(float), 'r', label='Train')
        valid_rows = df[df[val_col] != 'N/A']
        if len(valid_rows) > 0:
            ax.plot(valid_rows['epoch'].astype(float), valid_rows[val_col].astype(float), 'b', label='Validation')
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)

    def _plot_training_curves(self, args, df_losses_log):
        """Save per-repetition training curves for the active tasks.

        Classification gets loss + accuracy; segmentation gets the six curves
        (loss, accuracy, dice, jaccard, precision, recall).
        """
        if self.use_classification:
            os.makedirs(args.rep_screening_dir, exist_ok=True)
            self._plot_train_val_curve(
                df_losses_log, 'train_loss', 'val_loss',
                os.path.join(args.rep_screening_dir, 'loss.png'),
                'Training vs Validation Loss', 'Loss',
            )
            self._plot_train_val_curve(
                df_losses_log, 'train_accuracy', 'val_accuracy',
                os.path.join(args.rep_screening_dir, 'accuracy.png'),
                'Training vs Validation Accuracy', 'Accuracy',
            )

        if self.use_segmentation:
            figures_dir = os.path.join(args.rep_segmentation_dir, 'Figures')
            os.makedirs(figures_dir, exist_ok=True)
            segm_metric_pairs = [
                ('train_loss', 'val_loss', 'loss.png',
                 'Training vs Validation Loss', 'Loss'),
                ('train_seg_accuracy', 'val_seg_accuracy', 'accuracy.png',
                 'Training vs Validation Accuracy', 'Pixel accuracy'),
                ('train_dice', 'val_dice', 'dice.png',
                 'Training vs Validation Dice', 'Dice'),
                ('train_iou', 'val_iou', 'jaccard.png',
                 'Training vs Validation Jaccard', 'Jaccard'),
                ('train_precision', 'val_precision', 'precision.png',
                 'Training vs Validation Precision', 'Precision'),
                ('train_recall', 'val_recall', 'recall.png',
                 'Training vs Validation Recall', 'Recall'),
            ]
            for train_col, val_col, fname, title, ylabel in segm_metric_pairs:
                self._plot_train_val_curve(
                    df_losses_log, train_col, val_col,
                    os.path.join(figures_dir, fname), title, ylabel,
                )

    # ---- Classification label helpers ------------------------------------

    def _read_classification_targets(self, args):
        """Read and cache classification labels from a CSV file."""
        if self.classification_targets is not None:
            return self.classification_targets

        df = pd.read_csv(args.classification_csv_file_path)
        image_column = args.classification_image_column
        label_column = args.classification_label_column

        if image_column not in df.columns:
            raise ValueError('The column %s was not found in %s' % (image_column, args.classification_csv_file_path))
        if label_column not in df.columns:
            raise ValueError('The column %s was not found in %s' % (label_column, args.classification_csv_file_path))

        raw_labels = df[label_column].tolist()
        if args.class_names is not None:
            self.class_names = args.class_names
        elif any(isinstance(label, str) for label in raw_labels):
            self.class_names = sorted([str(label) for label in set(raw_labels)])
        else:
            self.class_names = ['class_%d' % class_idx for class_idx in range(args.num_classes)]

        label_to_idx = {class_name: class_idx for class_idx, class_name in enumerate(self.class_names)}
        targets = {}
        for _, row in df.iterrows():
            image_name = str(row[image_column])
            label = row[label_column]
            if isinstance(label, str):
                label = label_to_idx[str(label)]
            label = int(label)
            targets[image_name] = label
            targets[image_name.replace('.png', '').replace('.jpg', '').replace('.jpeg', '')] = label

        self.classification_targets = targets
        return self.classification_targets

    def _get_classification_label(self, args, current_input_image_name):
        """Retrieve the integer class label for a given input image."""
        targets = self._read_classification_targets(args)
        image_key = current_input_image_name
        image_key_no_ext = current_input_image_name.replace('.png', '').replace('.jpg', '').replace('.jpeg', '')
        if image_key in targets:
            return targets[image_key]
        if image_key_no_ext in targets:
            return targets[image_key_no_ext]
        return None

    # ---- Loss construction -----------------------------------------------

    def _build_classification_loss(self, args):
        """Build the classification loss per ``args.cls_loss_func``."""
        class_weights = getattr(args, 'class_weights', None)
        if class_weights is not None and torch.cuda.is_available():
            class_weights = class_weights.cuda()
        lf = args.cls_loss_func

        if lf == 'smooth_bce':
            if args.num_classes != 2:
                print('[Loss] WARNING: smooth_bce requires 2 classes, got %d — using CrossEntropyLoss.' % args.num_classes)
                return nn.CrossEntropyLoss(label_smoothing=args.label_smoothing, weight=class_weights)
            return SmoothBCEWithLogitsLoss(smoothing=args.label_smoothing, class_weights=class_weights)

        if lf == 'ce':
            return nn.CrossEntropyLoss(label_smoothing=args.label_smoothing, weight=class_weights)

        return FocalLoss(gamma=args.focal_gamma, class_weights=class_weights)

    def _build_segmentation_loss(self, args):
        """Build the segmentation loss per ``args.segm_loss_func``."""
        lf = args.segm_loss_func
        if lf == 'bce':
            return bce_segmentation_loss
        if lf == 'focal':
            return SegmentationFocalLoss(gamma=args.focal_gamma)
        return DiceLoss().get_loss_function()

    def _obtain_loss_funcs(self, args):
        """Return the segmentation and/or classification loss functions."""
        segm_loss = self._build_segmentation_loss(args) if self.use_segmentation else None
        cls_loss = self._build_classification_loss(args) if self.use_classification else None
        return segm_loss, cls_loss

    # ---- Model output helpers --------------------------------------------

    def _split_outputs(self, output):
        """Split the model output into (segmentation, classification) tensors.

        Single-head models return a plain tensor; multi-head models return a
        tuple. ``None`` is returned for any inactive head.
        """
        if isinstance(output, tuple):
            segmentation = output[0] if self.use_segmentation else None
            classification = output[1] if self.use_classification else None
            return segmentation, classification
        if self.use_segmentation:
            return output, None
        return None, output

    def predict(self, model, input_image):
        """Run a forward pass and return (segmentation, classification) outputs."""
        return self._split_outputs(model(input_image))

    # ---- Loss computation and one training step ---------------------------

    def _compute_training_loss(self, input_image, target_segmentation, target_class, output, loss_funcs, loss_weights, num_classes):
        """Compute the task loss(es) and accuracy.

        Returns ``(loss, segm_loss, cls_loss, log_images, correct)``.  For
        single-task models ``loss`` is the only active head loss; ``None`` is
        returned for the inactive heads. ``correct`` counts correctly
        classified samples (0 when classification is inactive).
        """
        segm_loss_func, classification_loss_func = loss_funcs
        output_segmentation, output_classification = self._split_outputs(output)

        segm_loss = None
        cls_loss = None
        correct = 0

        if self.use_segmentation:
            segm_loss = segm_loss_func(output_segmentation, target_segmentation)

        if self.use_classification:
            if num_classes == 2:
                target = target_class.float().view(-1, 1)
                cls_loss = classification_loss_func(output_classification, target)
                correct = int((torch.sigmoid(output_classification).view(-1) > 0.5).long().eq(target_class).sum().item())
            else:
                cls_loss = classification_loss_func(output_classification, target_class)
                correct = int(torch.argmax(output_classification, dim=1).eq(target_class).sum().item())

        if self.use_segmentation and self.use_classification:
            loss = loss_weights['segm'] * segm_loss + loss_weights['classification'] * cls_loss
        elif self.use_segmentation:
            loss = segm_loss
        else:
            loss = cls_loss

        log_images = None
        if output_segmentation is not None:
            log_images = input_image[0], target_segmentation[0], output_segmentation.cpu()[0]

        return loss, segm_loss, cls_loss, log_images, correct

    def _train_batch(self, args, model, input_batch, target_seg_batch, target_class_batch, parameters):
        """Run a single batched training step (forward, backward, optimise)."""
        optimizer = parameters['optimizer']
        optimizer.zero_grad()

        model = model.cuda()
        input_batch = input_batch.cuda()
        target_seg_batch = target_seg_batch.cuda() if target_seg_batch is not None else None
        target_class_batch = target_class_batch.cuda() if target_class_batch is not None else None

        model_output = model(input_batch)
        loss_funcs = parameters['loss_func']
        loss, segmentation_loss, classification_loss, _, correct = self._compute_training_loss(
            input_batch, target_seg_batch, target_class_batch, model_output, loss_funcs,
            {'segm': args.weight_loss[0], 'classification': args.weight_loss[1]},
            num_classes=args.num_classes,
        )
        loss.backward()
        optimizer.step()

        output_segmentation, _ = self._split_outputs(model_output)
        dice = iou = precision = recall = segm_accuracy = 0.0
        if self.use_segmentation:
            output_seg_cpu = output_segmentation.detach().cpu()
            target_seg_cpu = target_seg_batch.cpu()
            for b in range(output_seg_cpu.size(0)):
                segm_metrics = compute_segmentation_metrics(output_seg_cpu[b], target_seg_cpu[b])
                dice += segm_metrics['dice']
                iou += segm_metrics['iou']
                precision += segm_metrics['precision']
                recall += segm_metrics['recall']
                segm_accuracy += segm_metrics['accuracy']

        return model, {
            'loss': loss,
            'segm_loss': segmentation_loss,
            'cls_loss': classification_loss,
            'correct': correct,
            'acc': None,
            'dice': dice,
            'iou': iou,
            'precision': precision,
            'recall': recall,
            'segm_accuracy': segm_accuracy,
            'n_images': input_batch.size(0),
        }

    # ---- Training loop ----------------------------------------------------

    def train(self, args, current_repetition, model, input_dir_root, training_manager_obj, data_augmentation_obj, splitting_obj, preprocessing_method_obj):
        """Run the full training loop for one repetition across multiple epochs."""
        # ---- Initialisation ----
        self.aug_strength = args.aug_strength

        if getattr(args, 'freeze_backbone', False):
            if hasattr(model, 'encoder'):
                n_frozen = 0
                for param in model.encoder.parameters():
                    param.requires_grad = False
                    n_frozen += 1
                print('[Model] Frozen %d encoder parameters (--freeze-backbone).' % n_frozen)
            else:
                print('[Model] WARNING: --freeze-backbone requires --encoder; nothing frozen.')

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=args.lr_patience, factor=args.lr_factor,
        )
        loss_func = self._obtain_loss_funcs(args)
        best_val_loss = float('inf')
        best_val_accuracy = 0.0

        log_headers = ['epoch', 'train_loss', 'val_loss']
        if self.use_segmentation:
            log_headers += ['segmentation_train_loss', 'segmentation_val_loss']
        if self.use_classification:
            log_headers += ['classification_train_loss', 'classification_val_loss',
                            'train_accuracy', 'val_accuracy']
        if self.use_segmentation:
            log_headers += ['train_dice', 'train_iou', 'train_precision', 'train_recall', 'train_seg_accuracy']
        if self.use_segmentation:
            log_headers += ['val_dice', 'val_iou', 'val_precision', 'val_recall', 'val_seg_accuracy']
        log_headers += ['learning_rate']
        losses_log_list = []

        # ---- Training loop ----
        max_epochs = args.nofepochs_or_patience
        while training_manager_obj.continue_training(args):
            epoch = training_manager_obj.get_current_epoch()
            begin_time = time.time()

            # ── Epoch header ────────────────────────
            print(f'\nEpoch {epoch + 1}/{max_epochs}\n' + '-' * 52)

            # ── Training ─────────────────────────────────────────────────
            total_training_loss = 0
            total_segmentation_loss = 0
            total_classification_loss = 0
            total_correct = 0
            total_nof_images = 0
            total_dice = 0.0
            total_iou = 0.0
            total_precision = 0.0
            total_recall = 0.0
            total_segm_accuracy = 0.0

            model.train()
            train_names = splitting_obj.get_training_subset()
            train_dataset = LungDataset(
                self, args, input_dir_root, train_names,
                data_augmentation_obj, preprocessing_method_obj,
                args.image_size, augment=True,
                with_segmentation=self.use_segmentation,
                with_classification=self.use_classification,
            )
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=True,
                num_workers=args.num_workers,
            )
            parameters = {'loss_func': loss_func, 'optimizer': optimizer}
            train_batches = len(train_loader)
            for batch_idx, batch_data in enumerate(train_loader):
                _write_progress(batch_idx + 1, train_batches, prefix='Train')
                input_batch, target_seg_batch, target_class_batch = self._unpack_batch(batch_data)
                _, train_results = self._train_batch(
                    args, model, input_batch, target_seg_batch, target_class_batch,
                    parameters,
                )
                total_training_loss += (train_results['loss'] * train_results['n_images'])
                if train_results['segm_loss'] is not None:
                    total_segmentation_loss += (train_results['segm_loss'] * train_results['n_images'])
                if train_results['cls_loss'] is not None:
                    total_classification_loss += (train_results['cls_loss'] * train_results['n_images'])
                total_correct += train_results['correct'] or 0
                total_dice += train_results['dice']
                total_iou += train_results['iou']
                total_precision += train_results['precision']
                total_recall += train_results['recall']
                total_segm_accuracy += train_results['segm_accuracy']
                total_nof_images += train_results['n_images']
            _write_progress(train_batches, train_batches, prefix='Train')
            print()

            mean_current_training_loss = total_training_loss / total_nof_images
            mean_segmentation_loss = total_segmentation_loss / total_nof_images if self.use_segmentation else None
            mean_classification_loss = total_classification_loss / total_nof_images if self.use_classification else None
            train_accuracy = total_correct / total_nof_images if self.use_classification else None
            train_dice = total_dice / total_nof_images
            train_iou = total_iou / total_nof_images
            train_precision = total_precision / total_nof_images
            train_recall = total_recall / total_nof_images
            train_segm_accuracy = total_segm_accuracy / total_nof_images

            train_loss_val = mean_current_training_loss.detach().cpu().numpy()
            if self.use_segmentation and self.use_classification:
                print(f'TRAIN  JOINT LOSS: {train_loss_val:.4f}  '
                      f'CLS LOSS: {mean_classification_loss.detach().cpu().numpy():.4f}  '
                      f'SEGM LOSS: {mean_segmentation_loss.detach().cpu().numpy():.4f}  '
                      f'CLS ACC: {train_accuracy:.4f}  '
                      f'SEGM DICE: {train_dice:.4f}')
            elif self.use_segmentation:
                print(f'TRAIN  SEGM LOSS: {train_loss_val:.4f}  '
                      f'SEGM DICE: {train_dice:.4f}')
            else:
                print(f'TRAIN  CLS LOSS: {train_loss_val:.4f}  '
                      f'CLS ACC: {train_accuracy:.4f}')

            # ── Validation ──────────────────────────────────────────────
            val_results = self.validation(
                args, model, input_dir_root, splitting_obj.get_validation_subset(),
                loss_func, preprocessing_method_obj,
            )
            val_loss_for_manager = val_results['loss']
            if val_loss_for_manager is None:
                print('VAL    No validation samples processed.')
            else:
                if self.use_segmentation and self.use_classification:
                    print(f'VAL    JOINT LOSS: {val_loss_for_manager:.4f}  '
                          f'CLS LOSS: {val_results["cls_loss"]:.4f}  '
                          f'SEGM LOSS: {val_results["segm_loss"]:.4f}  '
                          f'CLS ACC: {val_results["accuracy"]:.4f}  '
                          f'SEGM DICE: {val_results["dice"]:.4f}')
                elif self.use_segmentation:
                    print(f'VAL    SEGM LOSS: {val_loss_for_manager:.4f}  '
                          f'SEGM DICE: {val_results["dice"]:.4f}')
                else:
                    print(f'VAL    CLS LOSS: {val_loss_for_manager:.4f}  '
                          f'CLS ACC: {val_results["accuracy"]:.4f}')

                if val_loss_for_manager < best_val_loss:
                    best_val_loss = val_loss_for_manager
                    print(f'>>> New best validation loss: {best_val_loss:.4f}')
                if val_results['accuracy'] is not None and val_results['accuracy'] > best_val_accuracy:
                    best_val_accuracy = val_results['accuracy']
                    print(f'>>> New best validation accuracy: {best_val_accuracy:.4f}')

            current_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_loss_for_manager if val_loss_for_manager is not None else mean_current_training_loss)
            print(f'LR: {current_lr:.2e}  |  Time: {time.time() - begin_time:.2f}s')

            # ── Logging ─────────────────────────────────────────────────
            row = [
                epoch,
                str(train_loss_val),
                str(val_loss_for_manager) if val_loss_for_manager is not None else 'N/A',
            ]
            if self.use_segmentation:
                row += [
                    str(mean_segmentation_loss.detach().cpu().numpy()),
                    str(val_results['segm_loss']) if val_results['segm_loss'] is not None else 'N/A',
                ]
            if self.use_classification:
                row += [
                    str(mean_classification_loss.detach().cpu().numpy()),
                    str(val_results['cls_loss']) if val_results['cls_loss'] is not None else 'N/A',
                    train_accuracy,
                    val_results['accuracy'] if val_results['accuracy'] is not None else 'N/A',
                ]
            if self.use_segmentation:
                row += [train_dice, train_iou, train_precision, train_recall, train_segm_accuracy]
            if self.use_segmentation:
                row += [
                    val_results['dice'] if val_results['dice'] is not None else 'N/A',
                    val_results['iou'] if val_results['iou'] is not None else 'N/A',
                    val_results['precision'] if val_results['precision'] is not None else 'N/A',
                    val_results['recall'] if val_results['recall'] is not None else 'N/A',
                    val_results['segm_accuracy'] if val_results['segm_accuracy'] is not None else 'N/A',
                ]
            row.append('%.2e' % current_lr)
            losses_log_list.append(row)
            df_losses_log = pd.DataFrame(losses_log_list, columns=log_headers)
            df_losses_log.to_csv(os.path.join(args.rep_dir, 'loss_log.csv'), index=False)
            self._plot_training_curves(args, df_losses_log)

            training_manager_obj.store_model(args, model, self.save_model)
            training_manager_obj.update_training_manager(
                args, current_repetition, model, val_loss_for_manager, self.save_model,
            )

        training_manager_obj.show_final_training_report(args)
        if self.use_classification:
            print('Best validation accuracy across all epochs: %.4f' % best_val_accuracy)

        # Save the last model of the repetition.
        models_dir = os.path.join(args.rep_dir, 'Models')
        os.makedirs(models_dir, exist_ok=True)
        self.save_model(model, os.path.join(models_dir, 'last.pkl'))
        print('Saved last model: %s' % os.path.join(models_dir, 'last.pkl'))
        return model

    def _unpack_batch(self, batch_data):
        """Unpack a DataLoader batch into (image, seg_target, cls_target).

        ``None`` is returned for targets not required by the active task
        (segmentation-only / classification-only datasets have fewer outputs).
        """
        if self.use_segmentation and self.use_classification:
            input_batch, target_seg_batch, target_class_batch, _names = batch_data
            return input_batch, target_seg_batch, target_class_batch
        if self.use_segmentation:
            input_batch, target_seg_batch, _names = batch_data
            return input_batch, target_seg_batch, None
        input_batch, target_class_batch, _names = batch_data
        return input_batch, None, target_class_batch

    # ---- Evaluation loop --------------------------------------------------

    def validation(self, args, model, input_dir_root, input_dataset, loss_funcs, preprocessing_method_obj):
        """Run validation over a dataset subset without storing per-image outputs."""
        return self._evaluation_loop(
            args, -1, model, input_dir_root, input_dataset,
            loss_funcs, preprocessing_method_obj, store_outputs=False,
        )

    def _evaluation_loop(self, args, current_repetition_number, model, input_dir_root, input_dataset, loss_funcs, preprocessing_method_obj, store_outputs=True):
        """Iterate over a dataset split, compute losses, and optionally store outputs.

        Returns a dict with 'loss', 'segm_loss', 'cls_loss', 'accuracy' and
        the mean segmentation metrics (dice/iou/precision/recall/segm_accuracy).
        """
        total_nof_images = 0
        total_loss = 0
        total_segmentation_loss = 0
        total_classification_loss = 0
        total_correct = 0
        total_dice = 0.0
        total_iou = 0.0
        total_precision = 0.0
        total_recall = 0.0
        total_segm_accuracy = 0.0
        y_true, y_pred, y_prob, rows = [], [], [], []
        segm_metrics_rows = []

        rep_dir = result_dir = overlapped_dir = None
        if store_outputs and self.use_segmentation:
            rep_dir = args.rep_segmentation_dir
            result_dir = os.path.join(rep_dir, 'Mask')
            overlapped_dir = os.path.join(rep_dir, 'Overlapped')
            os.makedirs(result_dir, exist_ok=True)
            os.makedirs(overlapped_dir, exist_ok=True)

        model.eval()
        eval_desc = 'Test' if store_outputs else 'Val'
        eval_dataset = LungDataset(
            self, args, input_dir_root, input_dataset,
            None, preprocessing_method_obj, args.image_size, augment=False,
            with_segmentation=self.use_segmentation,
            with_classification=self.use_classification,
        )
        eval_loader = torch.utils.data.DataLoader(
            eval_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers,
        )
        with torch.no_grad():
            eval_batches = len(eval_loader)
            for batch_idx, batch_data in enumerate(eval_loader):
                _write_progress(batch_idx + 1, eval_batches, prefix=eval_desc)
                input_batch, target_seg_batch, target_class_batch = self._unpack_batch(batch_data)
                batch_size = len(input_batch)

                input_batch = input_batch.cuda()
                target_seg_batch_cuda = target_seg_batch.cuda() if target_seg_batch is not None else None
                target_class_batch_cuda = target_class_batch.cuda() if target_class_batch is not None else None

                model_output = model(input_batch)
                loss, segmentation_loss, classification_loss, _, correct = self._compute_training_loss(
                    input_batch, target_seg_batch_cuda, target_class_batch_cuda, model_output, loss_funcs,
                    {'segm': args.weight_loss[0], 'classification': args.weight_loss[1]},
                    num_classes=args.num_classes,
                )
                output_segmentation, output_classification = self._split_outputs(model_output)

                total_loss += loss.cpu() * batch_size
                if segmentation_loss is not None:
                    total_segmentation_loss += segmentation_loss.cpu() * batch_size
                if classification_loss is not None:
                    total_classification_loss += classification_loss.cpu() * batch_size
                total_correct += correct
                total_nof_images += batch_size

                for b in range(batch_size):
                    current_input_image_name = batch_data[-1][b]
                    if self.use_classification:
                        if args.num_classes == 2:
                            binary_probs = torch.sigmoid(output_classification[b])
                            probs = torch.cat([1 - binary_probs, binary_probs]).view(1, 2)
                            pred_class = int((binary_probs > 0.5).long().item())
                        else:
                            probs = F.softmax(output_classification[b], dim=0).view(1, -1)
                            pred_class = int(torch.argmax(probs, dim=1).item())
                        true_class = int(target_class_batch[b].item())

                        y_true.append(true_class)
                        y_pred.append(pred_class)
                        y_prob.append(probs.cpu().numpy()[0])
                        rows.append([current_input_image_name, true_class, pred_class] + probs.cpu().numpy()[0].tolist())

                    if self.use_segmentation:
                        segm_metrics = compute_segmentation_metrics(output_segmentation.cpu()[b], target_seg_batch[b])
                        segm_metrics['image'] = current_input_image_name
                        segm_metrics_rows.append(segm_metrics)
                        total_dice += segm_metrics['dice']
                        total_iou += segm_metrics['iou']
                        total_precision += segm_metrics['precision']
                        total_recall += segm_metrics['recall']
                        total_segm_accuracy += segm_metrics['accuracy']

                    if store_outputs and self.use_segmentation:
                        stem = current_input_image_name.rsplit('.', 1)[0]

                        # Predicted mask resized back to the original image size
                        input_image_path = os.path.join(input_dir_root, current_input_image_name)
                        orig_h, orig_w = self._original_image_size(input_image_path)
                        pred_full = F.interpolate(
                            output_segmentation.cpu()[b].unsqueeze(0), size=(orig_h, orig_w),
                            mode='bilinear', align_corners=False,
                        )
                        pred_binary = (pred_full > MASK_THRESHOLD).float()
                        vutils.save_image(pred_binary, os.path.join(result_dir, '%s.png' % stem))

                        self._save_prediction_overlaps(
                            input_image_path,
                            pred_full,
                            os.path.join(args.lung_segm_root_dir, '%s.png' % stem),
                            overlapped_dir,
                            stem,
                        )
            _write_progress(eval_batches, eval_batches, prefix=eval_desc)
            print()

        model.train()

        # ---- Cleanup ----
        if total_nof_images == 0:
            return {
                'loss': None, 'segm_loss': None, 'cls_loss': None, 'accuracy': None,
                'dice': None, 'iou': None, 'precision': None, 'recall': None,
                'segm_accuracy': None,
            }

        if store_outputs:
            if self.use_classification:
                store_classification_results(
                    y_true, y_pred, y_prob, rows,
                    self.class_names, args.num_classes,
                    args.rep_screening_dir,
                )
            if self.use_segmentation:
                store_segmentation_metrics_table(segm_metrics_rows, rep_dir)

        return {
            'loss': (total_loss / total_nof_images).item(),
            'segm_loss': (total_segmentation_loss / total_nof_images).item() if self.use_segmentation else None,
            'cls_loss': (total_classification_loss / total_nof_images).item() if self.use_classification else None,
            'accuracy': (total_correct / total_nof_images) if self.use_classification else None,
            'dice': total_dice / total_nof_images,
            'iou': total_iou / total_nof_images,
            'precision': total_precision / total_nof_images,
            'recall': total_recall / total_nof_images,
            'segm_accuracy': total_segm_accuracy / total_nof_images,
        }

    def test(self, args, current_repetition_number, model, input_dir_root, input_dataset, preprocessing_method_obj):
        """Evaluate the model on the test set and return the averaged results dict."""
        loss_funcs = self._obtain_loss_funcs(args)
        return self._evaluation_loop(
            args, current_repetition_number, model, input_dir_root, input_dataset,
            loss_funcs, preprocessing_method_obj, store_outputs=True,
        )


class ClassificationOnlyTrainer(BaseTrainer):
    """Single-task classification trainer (segmentation head disabled)."""

    def __init__(self):
        super(ClassificationOnlyTrainer, self).__init__()
        self.use_segmentation = False
        self.use_classification = True
        print('[Approach] Single-task classification training.')


class SegmentationOnlyTrainer(BaseTrainer):
    """Single-task segmentation trainer (classification head disabled)."""

    def __init__(self):
        super(SegmentationOnlyTrainer, self).__init__()
        self.use_segmentation = True
        self.use_classification = False
        print('[Approach] Single-task segmentation training.')


class MultitaskTrainer(BaseTrainer):
    """Multi-task trainer combining segmentation and classification."""

    def __init__(self):
        super(MultitaskTrainer, self).__init__()
        self.use_segmentation = True
        self.use_classification = True
        print('[Approach] Multi-task training with segmentation and classification.')