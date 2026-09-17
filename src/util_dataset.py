"""Dataset utilities: splitting, augmentation, preprocessing.

Classes
-------
NoDataAugmentation / DataAugmentation
    Pass-through or stochastic augmentation (colour jitter, rotation, flip).
NoPreprocessing / Preprocessing
    Identity transform or channel-wise (x - mean) / std normalization.
LungDataset
    Dataset yielding the image plus the segmentation mask and/or class label
    required by the active task.
HoldoutSplitting
    Patient-level train/val/test split with optional class balancing.
"""
import os
import random
from collections import defaultdict

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


def compute_normalization_stats(input_dir_root, image_names, image_size):
    """Compute per-channel mean/std over the given images (raw, resized, [0, 1]).

    Returns ``(mean, std)`` as lists of length 3 ready for :class:`Preprocessing`.
    """
    from PIL import Image

    mean_acc = torch.zeros(3, dtype=torch.float64)
    square_acc = torch.zeros(3, dtype=torch.float64)
    total_pixels = 0
    for name in image_names:
        full_path = os.path.join(input_dir_root, name)
        image = Image.open(full_path).convert('RGB')
        if image_size is not None:
            size = (int(image_size[0]), int(image_size[1]))
            image = TF.resize(image, size)
        tensor = TF.to_tensor(image)
        channel_flat = tensor.reshape(3, -1).double()
        mean_acc += channel_flat.sum(dim=1)
        square_acc += (channel_flat ** 2).sum(dim=1)
        total_pixels += channel_flat.shape[1]

    mean = mean_acc / total_pixels
    variance = square_acc / total_pixels - mean ** 2
    std = variance.clamp(min=0.0).sqrt()
    return mean.tolist(), std.tolist()


class NoDataAugmentation:
    """Identity data augmentation — images pass through unchanged."""

    def __init__(self, aug_strength=None):
        print('[Augmentation] No data augmentation was selected.')

    def apply_transforms(self, input_image, target_segmentation=None):
        if target_segmentation is not None:
            return input_image, target_segmentation
        return input_image, None


class DataAugmentation:
    """Albumentations-based augmentation, same transforms for image and mask.

    Applies the exact same geometric augmentation (HFlip + Rotate) to the
    input image and the segmentation mask so both stay pixel-aligned, while
    colour/noise transforms touch only the image. Intensity levels:

    - ``light``:  HFlip + rotation ``[-8, 8]``.
    - ``medium``: HFlip + rotation ``[-12, 12]`` + colour jitter.
    - ``strong``: HFlip + rotation ``[-20, 20]`` + colour jitter +
      Gaussian noise (p=0.2) + blur (p=0.2).

    No vertical flip is used (anatomically meaningless for lung images).
    """

    STRENGTHS = {
        'light': {'rotation': 8, 'color_jitter': None},
        'medium': {'rotation': 12, 'color_jitter': (0.12, 0.12)},
        'strong': {'rotation': 20, 'color_jitter': (0.2, 0.2)},
    }

    def __init__(self, aug_strength='medium'):
        aug_strength = str(aug_strength).lower()
        if aug_strength not in self.STRENGTHS:
            raise ValueError(
                'Unknown aug_strength %r. Choose from %s.'
                % (aug_strength, ', '.join(sorted(self.STRENGTHS)))
            )
        self.aug_strength = aug_strength
        config = self.STRENGTHS[aug_strength]

        aug_list = [
            A.HorizontalFlip(p=0.5),
            A.Rotate(
                limit=config['rotation'],
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                fill_mask=0,
                p=1.0,
            ),
        ]
        if config['color_jitter'] is not None:
            aug_list.append(
                A.ColorJitter(
                    brightness=config['color_jitter'][0],
                    contrast=config['color_jitter'][1],
                )
            )
        if aug_strength == 'strong':
            aug_list.append(A.GaussNoise(std_range=(0.02, 0.06), mean_range=(0.0, 0.0), p=0.2))
            aug_list.append(A.Blur(blur_limit=(3, 7), p=0.2))

        self.pipeline = A.Compose(aug_list, additional_targets={'mask': 'mask'})

    def apply_transforms(self, input_image, target_segmentation=None):
        """Apply the same geometric transforms to the image and optionally its mask."""
        image_np = (input_image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        if target_segmentation is None:
            augmented = self.pipeline(image=image_np)
            out_image = (
                torch.from_numpy(augmented['image'].copy()).float().div(255).permute(2, 0, 1)
            )
            return out_image, None

        mask_np = np.squeeze(target_segmentation.numpy())
        augmented = self.pipeline(image=image_np, mask=mask_np)
        out_image = (
            torch.from_numpy(augmented['image'].copy()).float().div(255).permute(2, 0, 1)
        )
        out_mask = torch.from_numpy(np.ascontiguousarray(augmented['mask'])).float()
        if out_mask.ndim == 2:
            out_mask = out_mask.unsqueeze(0)
        return out_image, out_mask


class NoPreprocessing:
    """Identity preprocessing — tensor passes through unchanged."""

    def __init__(self):
        pass

    def apply_preprocessing_on_image(self, input_image):
        return input_image


class Preprocessing:
    """Normalisation: ``(x - mean) / std`` with given channel-wise statistics.

    Parameters
    ----------
    mean, std : list | None
        Channel-wise statistics. They come from ImageNet when a pretrained
        encoder is used, or are computed from the training data otherwise.
    """

    def __init__(self, mean=None, std=None):
        if mean is None or std is None:
            raise ValueError(
                'Preprocessing requires both mean and std (ImageNet encoder '
                'stats or stats computed from the train set).'
            )
        self.mean = mean
        self.std = std

    def apply_preprocessing_on_image(self, input_image):
        """Normalise channels: ``(input_image - mean) / std``."""
        mean_t = torch.tensor(self.mean, device=input_image.device).view(3, 1, 1)
        std_t = torch.tensor(self.std, device=input_image.device).view(3, 1, 1)
        return (input_image - mean_t) / std_t


class LungDataset(torch.utils.data.Dataset):
    """Dataset yielding image plus the targets required by the active task.

    Loads the raw image, applies data augmentation (training only) and the
    preprocessing pipeline, and returns the tensors without a batch dimension
    so that a :class:`torch.utils.data.DataLoader` can stack them into batches.
    Samples whose required segmentation mask or classification label is missing
    are filtered out at construction time.

    Parameters
    ----------
    with_segmentation : bool
        Load a segmentation mask and return it alongside the image.
    with_classification : bool
        Load a class label and return it alongside the image.
    """

    def __init__(self, trainer, args, input_dir_root, image_names, data_augmentation_obj,
                 preprocessing_method_obj, image_size, augment=False,
                 with_segmentation=False, with_classification=False):
        self.trainer = trainer
        self.args = args
        self.input_dir_root = input_dir_root
        self.data_augmentation_obj = data_augmentation_obj
        self.preprocessing_method_obj = preprocessing_method_obj
        self.image_size = image_size
        self.augment = augment
        self.with_segmentation = with_segmentation
        self.with_classification = with_classification
        self.image_names = [
            name for name in image_names
            if self._has_required_targets(name)
        ]

    def _has_required_targets(self, name):
        """Return True if the sample has all targets required by the active task."""
        if self.with_segmentation and not os.path.exists(self.trainer._segmentation_target_path(self.args, name)):
            print('WARNING! Skipping %s: segmentation target not found.' % name)
            return False
        if self.with_classification and self.trainer._get_classification_label(self.args, name) is None:
            print('WARNING! Skipping %s: classification label not found.' % name)
            return False
        return True

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        name = self.image_names[idx]
        input_image = self.trainer._load_input_image_raw(self.input_dir_root, name, self.image_size)

        target_segmentation = None
        if self.with_segmentation:
            target_segmentation = self.trainer._load_segmentation_target(self.args, name, self.image_size)
            target_segmentation = target_segmentation.squeeze(0)

        classification_label = None
        if self.with_classification:
            classification_label = self.trainer._get_classification_label(self.args, name)

        if self.augment:
            input_image, target_segmentation = self.data_augmentation_obj.apply_transforms(
                input_image, target_segmentation,
            )

        input_image = self.preprocessing_method_obj.apply_preprocessing_on_image(input_image)

        if self.with_segmentation and self.with_classification:
            return input_image, target_segmentation, int(classification_label), name
        if self.with_segmentation:
            return input_image, target_segmentation, name
        return input_image, int(classification_label), name


class HoldoutSplitting:
    """Split image names into train / validation / test subsets.

    When classification labels are available the split is class-balanced
    and performed at patient level (via ``GroupShuffleSplit``) to avoid
    data leakage.  Otherwise a simple random shuffle is used.

    Parameters
    ----------
    train_pct : float
        Fraction of data used for training.
    val_pct : float
        Fraction of data used for validation.
    test_pct : float
        Fraction of data used for testing.
    """

    def __init__(self, train_pct=0.8, val_pct=0.0, test_pct=0.2):
        self._train_pct = train_pct
        self._val_pct = val_pct
        self._test_pct = test_pct
        self._class_weights = None

        print('\n+++ HOLDOUT SPLITTING')
        print('Training percentage -> %.2f' % self._train_pct)
        print('Validation percentage -> %.2f' % self._val_pct)
        print('Test percentage -> %.2f' % self._test_pct)

    def _load_classification_labels(self, args, image_names):
        """Read classification labels from CSV and align them with *image_names*.

        Returns
        -------
        list
            Labels (as strings) in the same order as *image_names*.
            Missing images get the label ``'unknown'``.
        """
        df = pd.read_csv(args.classification_csv_file_path)
        image_column = args.classification_image_column
        label_column = args.classification_label_column

        label_map = {}
        for _, row in df.iterrows():
            img_name = str(row[image_column])
            label_map[img_name] = str(row[label_column])
            label_map[img_name.replace('.png', '').replace('.jpg', '').replace('.jpeg', '')] = str(row[label_column])

        labels = []
        for img_name in image_names:
            key = img_name
            key_no_ext = img_name.replace('.png', '').replace('.jpg', '').replace('.jpeg', '')
            if key in label_map:
                labels.append(label_map[key])
            elif key_no_ext in label_map:
                labels.append(label_map[key_no_ext])
            else:
                labels.append('unknown')

        print('Loaded classification labels for %d images (%d unique classes).' % (
            len(labels),
            len(set(labels)),
        ))
        return labels

    def _balance_classes(self, image_names, labels, seed=0):
        """Subsample majority classes so each class has the same number of images.

        * 2 classes → majority subsampled to minority count (minority kept intact).
        * 3 classes → majority subsampled to the average of the two minority counts.
        * Otherwise → no balancing.
        """
        rng = random.Random(seed)
        class_indices = defaultdict(list)
        for i, label in enumerate(labels):
            class_indices[label].append(i)

        num_classes = len(class_indices)
        balanced = []

        if num_classes == 2:
            min_count = min(len(v) for v in class_indices.values())
            print('Balancing 2 classes: sampling majority to %d images.' % min_count)
            for indices in class_indices.values():
                if len(indices) > min_count:
                    balanced.extend(rng.sample(indices, min_count))
                else:
                    balanced.extend(indices)
        elif num_classes == 3:
            counts = {c: len(v) for c, v in class_indices.items()}
            sorted_c = sorted(counts, key=counts.get)
            minority = sorted_c[:-1]
            majority = sorted_c[-1]
            target = int(sum(counts[c] for c in minority) / len(minority))
            print('Balancing 3 classes: sampling majority to %d images.' % target)
            for c in minority:
                balanced.extend(class_indices[c])
            balanced.extend(rng.sample(class_indices[majority], min(target, counts[majority])))
        else:
            print('No class balancing applied (%d classes).' % num_classes)
            balanced = list(range(len(image_names)))

        return balanced

    def _extract_patient_ids(self, image_names):
        """Extract patient identifier from the first token of each filename."""
        return [name.split('_')[0] for name in image_names]

    def _patient_split(self, labels, patients, random_state):
        """Two-stage ``GroupShuffleSplit``: train+val vs test, then train vs val.

        Parameters
        ----------
        labels : list of str
            Class labels for each sample.
        patients : list of str
            Patient identifiers (same length as *labels*).
        random_state : int
            Seed for the shuffle split.

        Returns
        -------
        tuple of np.ndarray
            ``(train_idx, val_idx, test_idx)`` — index arrays into the balanced
            dataset.
        """
        from sklearn.model_selection import GroupShuffleSplit

        unique_labels = sorted(set(labels))
        label_to_idx = {l: i for i, l in enumerate(unique_labels)}
        y = np.array([label_to_idx[l] for l in labels])
        groups = np.array(patients)

        per_train = self._train_pct
        per_val = self._val_pct
        per_test = self._test_pct
        per_train_val = per_train + per_val

        gss1 = GroupShuffleSplit(n_splits=1, test_size=per_test, random_state=random_state)
        train_val_idx, test_idx = next(gss1.split(X=y, y=y, groups=groups))

        gss2 = GroupShuffleSplit(n_splits=1, test_size=per_val / per_train_val, random_state=random_state)
        train_rel_idx, val_rel_idx = next(
            gss2.split(X=train_val_idx, y=y[train_val_idx], groups=groups[train_val_idx])
        )

        train_idx = train_val_idx[train_rel_idx]
        val_idx = train_val_idx[val_rel_idx]

        return train_idx, val_idx, test_idx

    def _class_distribution_string(self, labels, class_names, label_to_idx, indices, global_counts, global_total):
        """Format per-class image counts of a split, with split and global percentages."""
        split_counts = [0] * len(class_names)
        for i in indices:
            split_counts[label_to_idx[labels[i]]] += 1
        total = len(indices)
        parts = []
        for class_idx, class_name in enumerate(class_names):
            split_pct = 100.0 * split_counts[class_idx] / total if total else 0.0
            global_pct = 100.0 * global_counts[class_idx] / global_total if global_total else 0.0
            parts.append('%s=%d (split %.2f%% | global %.2f%%)' % (
                class_name, split_counts[class_idx], split_pct, global_pct))
        return ', '.join(parts)

    def _patient_class_distribution_string(self, labels, patients, class_names, indices):
        """Format per-class sample/patient counts of a split."""
        parts = []
        for class_name in class_names:
            class_samples = 0
            class_patients = set()
            for i in indices:
                if labels[i] == class_name:
                    class_samples += 1
                    class_patients.add(patients[i])
            parts.append('%s=%d samples / %d patients' % (class_name, class_samples, len(class_patients)))
        return ', '.join(parts)

    def get_class_weights(self):
        """Return the normalised per-class weights (numpy array), or None if not computed."""
        return self._class_weights

    def load_dataset_with_random_shuffling(self, args, iteration, seed=0):
        """Shuffle and partition image names into train/val/test subsets.

        The split is stored internally and accessed via
        :meth:`get_training_subset`, :meth:`get_validation_subset`, and
        :meth:`get_test_subset`.

        Parameters
        ----------
        args:
            Parsed CLI arguments (needed for CSV path and column names).
        iteration:
            Current repetition index — combined with *seed* for a unique but
            reproducible split per repetition.
        seed:
            Base random seed.
        """
        image_names = [f for f in os.listdir(args.input_dir_root) if not f.startswith('.')]
        print('Total images found: %d' % len(image_names))

        # ── Load classification labels when available ────────────────────── #
        classification_labels = None
        if hasattr(args, 'classification_csv_file_path') and args.classification_csv_file_path is not None:
            classification_labels = self._load_classification_labels(args, image_names)

        if classification_labels is not None and len(classification_labels) == len(image_names):
            if getattr(args, 'balanced', True):
                print('\n+++ BALANCED PATIENT-LEVEL SPLIT')
                balanced_indices = self._balance_classes(image_names, classification_labels, seed=seed)
                balanced_images = [image_names[i] for i in balanced_indices]
                balanced_labels = [classification_labels[i] for i in balanced_indices]
            else:
                print('\n+++ PATIENT-LEVEL SPLIT (no balancing)')
                balanced_images = image_names
                balanced_labels = classification_labels

            patients = self._extract_patient_ids(balanced_images)
            train_idx, val_idx, test_idx = self._patient_split(balanced_labels, patients, seed + iteration)

            train_img = [balanced_images[i] for i in train_idx]
            val_img = [balanced_images[i] for i in val_idx]
            test_img = [balanced_images[i] for i in test_idx]

            n_train_patients = len(set(patients[i] for i in train_idx))
            n_val_patients = len(set(patients[i] for i in val_idx))
            n_test_patients = len(set(patients[i] for i in test_idx))

            total_split = len(train_img) + len(val_img) + len(test_img)
            print('Images -> train=%d (%.2f%%), val=%d (%.2f%%), test=%d (%.2f%%)' % (
                len(train_img), 100.0 * len(train_img) / total_split,
                len(val_img), 100.0 * len(val_img) / total_split,
                len(test_img), 100.0 * len(test_img) / total_split))
            print('Patients -> train=%d, val=%d, test=%d' % (n_train_patients, n_val_patients, n_test_patients))

            class_names = list(args.class_names) if getattr(args, 'class_names', None) is not None \
                else sorted(set(balanced_labels))
            label_to_idx = {class_name: class_idx for class_idx, class_name in enumerate(class_names)}
            global_counts = [0] * len(class_names)
            for label in balanced_labels:
                global_counts[label_to_idx[label]] += 1
            global_total = len(balanced_labels)

            print('Class distribution -> train: %s' % self._class_distribution_string(
                balanced_labels, class_names, label_to_idx, train_idx, global_counts, global_total))
            print('Class distribution -> val: %s' % self._class_distribution_string(
                balanced_labels, class_names, label_to_idx, val_idx, global_counts, global_total))
            print('Class distribution -> test: %s' % self._class_distribution_string(
                balanced_labels, class_names, label_to_idx, test_idx, global_counts, global_total))
            print('Patient class distribution -> train: %s' % self._patient_class_distribution_string(
                balanced_labels, patients, class_names, train_idx))
            print('Patient class distribution -> val: %s' % self._patient_class_distribution_string(
                balanced_labels, patients, class_names, val_idx))
            print('Patient class distribution -> test: %s' % self._patient_class_distribution_string(
                balanced_labels, patients, class_names, test_idx))

            counts = np.array(global_counts, dtype=np.float32)
            class_weights = np.zeros_like(counts, dtype=np.float32)
            mask = counts > 0
            if mask.any():
                class_weights[mask] = global_total / (mask.sum() * counts[mask])
                class_weights /= class_weights.sum()
            self._class_weights = class_weights
        else:
            print('\n+++ RANDOM SHUFFLE SPLIT')
            rng = random.Random(seed + iteration)
            rng.shuffle(image_names)
            total = len(image_names)
            n_train = int(total * self._train_pct)
            n_val = int(total * self._val_pct)
            remainder = total - n_train - n_val
            n_train += remainder

            train_img = image_names[:n_train]
            val_img = image_names[n_train:n_train + n_val]
            test_img = image_names[n_train + n_val:]

            print('Images -> train=%d (%.2f%%), val=%d (%.2f%%), test=%d (%.2f%%)' % (
                len(train_img), 100.0 * len(train_img) / total,
                len(val_img), 100.0 * len(val_img) / total,
                len(test_img), 100.0 * len(test_img) / total))

        self._random_input_images_names_list = train_img + val_img + test_img
        self._nof_samples_training = len(train_img)
        self._nof_samples_validation = len(val_img)
        self._nof_samples_test = len(test_img)
        self._total_dataset_size = len(self._random_input_images_names_list)

    def get_training_subset(self):
        """Return the list of training image names."""
        return self._random_input_images_names_list[0:self._nof_samples_training]

    def get_validation_subset(self):
        """Return the list of validation image names."""
        return self._random_input_images_names_list[
            self._nof_samples_training:self._nof_samples_training + self._nof_samples_validation
        ]

    def get_test_subset(self):
        """Return the list of test image names."""
        return self._random_input_images_names_list[
            self._nof_samples_training + self._nof_samples_validation:self._total_dataset_size
        ]