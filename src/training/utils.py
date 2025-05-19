"""Utility functions for the waste classification model training pipeline.

This module provides helper functions for:
- Managing training sessions and directories
- Loading and transforming training data
- Creating training summary reports
"""

import os
import time
import datetime
import math
import random
from collections import defaultdict
from typing import Tuple, List, Dict, Any, Optional, Callable, Iterator, Set

import numpy as np
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Sampler

# Type alias for transforms
Transform = Callable[[Any], torch.Tensor]


def get_next_session_number(base_output_dir: str = "output") -> int:
    """Determine the next available session number for training.

    Examines existing session directories in the base output directory
    and determines the next available session number to use.

    Args:
        base_output_dir: Base directory where session folders are stored

    Returns:
        The next available session number
    """
    os.makedirs(base_output_dir, exist_ok=True)

    session_dirs = []
    for d in os.listdir(base_output_dir):
        if os.path.isdir(os.path.join(base_output_dir, d)) and d.startswith(
            "session_"
        ):
            try:
                session_number = int(d.split("_")[1])
                session_dirs.append(session_number)
            except ValueError:
                pass

    if not session_dirs:
        return 1
    else:
        return max(session_dirs) + 1


def get_data_transforms() -> transforms.Compose:
    """Get data transformations for training and validation datasets.

    Creates preprocessing transforms for model input images.
    Only tensor conversion and normalization are applied, since images are
    already resized and processed during dataset preparation.

    Returns:
        A transforms.Compose object with minimal image preprocessing steps.
    """
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )


class BalancedClassSampler(Sampler[int]):
    """Sampler that ensures balanced class distribution in each epoch.

    Ensures each class contributes the same number of samples in each epoch,
    equal to the size of the smallest class. For larger classes, different
    samples are selected in each epoch until all samples have been used.

    Args:
        dataset: Dataset to sample from (ImageFolder or Subset of ImageFolder)
        epoch: Current epoch number (incremented for each epoch)
    """

    def __init__(self, dataset, epoch: int = 0):
        self.dataset = dataset
        self.epoch = epoch
        self.num_samples = len(dataset)

        # Handle both ImageFolder and Subset cases
        if hasattr(dataset, "samples"):
            # Direct ImageFolder
            self._init_from_image_folder(dataset)
        elif hasattr(dataset, "dataset") and hasattr(
            dataset.dataset, "samples"
        ):
            # Subset of ImageFolder
            self._init_from_subset(dataset)
        else:
            raise ValueError(
                "Dataset must be either an ImageFolder or a Subset of ImageFolder"
            )

        # Calculate total number of samples per epoch (balanced)
        self.samples_per_epoch = self.min_class_size * self.num_classes

        # Print diagnostic information
        print("\nClass distribution in dataset:")
        print("-" * 40)

        # Get class names if available
        class_names = getattr(dataset, "classes", None)
        if class_names is None and hasattr(dataset, "dataset"):
            class_names = getattr(dataset.dataset, "classes", None)

        for class_idx, indices in self.class_indices.items():
            class_name = (
                f"Class {class_idx}"
                if class_names is None
                else class_names[class_idx]
            )
            print(f"{class_name:<15}: {len(indices):>5} samples")

        print("-" * 40)
        print(f"Minimum class size: {self.min_class_size}")
        print(f"Samples per balanced epoch: {self.samples_per_epoch}")
        print(
            f"Each class will contribute exactly {self.min_class_size} samples per epoch"
        )
        print("-" * 40)

    def _init_from_image_folder(self, dataset):
        """Initialize class indices from an ImageFolder dataset."""
        self.class_indices: Dict[int, List[int]] = defaultdict(list)
        for idx, (_, class_idx) in enumerate(dataset.samples):
            self.class_indices[class_idx].append(idx)

        # Find the size of the smallest class
        self.min_class_size = min(
            len(indices) for indices in self.class_indices.values()
        )
        self.num_classes = len(self.class_indices)

    def _init_from_subset(self, subset):
        """Initialize class indices from a Subset of an ImageFolder dataset."""
        self.class_indices: Dict[int, List[int]] = defaultdict(list)
        image_folder = subset.dataset

        # Map from subset indices to original indices to class indices
        for subset_idx, orig_idx in enumerate(subset.indices):
            _, class_idx = image_folder.samples[orig_idx]
            self.class_indices[class_idx].append(subset_idx)

        # Find the size of the smallest class
        self.min_class_size = min(
            len(indices) for indices in self.class_indices.values()
        )
        self.num_classes = len(self.class_indices)

    def __iter__(self) -> Iterator[int]:
        # Set random seed based on epoch for reproducibility but different each epoch
        g = torch.Generator()
        g.manual_seed(42 + self.epoch)

        indices = []
        for class_idx, class_samples in self.class_indices.items():
            # For each class, select min_class_size samples
            # If there are more samples than min_class_size, rotate through them based on epoch
            if len(class_samples) > self.min_class_size:
                # Calculate starting position based on epoch (to rotate through samples)
                start_idx = (self.epoch * self.min_class_size) % len(
                    class_samples
                )
                # Take min_class_size samples, wrapping around if needed
                selected_samples = (
                    class_samples[start_idx:] + class_samples[:start_idx]
                )
                selected_samples = selected_samples[: self.min_class_size]
            else:
                # If class size is equal to or smaller than min_class_size, use all samples
                selected_samples = class_samples

            indices.extend(selected_samples)

        # Shuffle the indices (while maintaining class balance)
        shuffled_indices = torch.randperm(len(indices), generator=g).tolist()
        return iter([indices[i] for i in shuffled_indices])

    def __len__(self) -> int:
        return self.samples_per_epoch

    def set_epoch(self, epoch: int) -> None:
        """Update the epoch counter for this sampler.

        This should be called at the beginning of each epoch to ensure
        different samples are selected from larger classes.

        Args:
            epoch: New epoch number
        """
        if epoch != self.epoch:
            print(f"\nRotating samples for epoch {epoch}")
            print(
                f"This will use different samples from larger classes while maintaining class balance"
            )
        self.epoch = epoch


def load_datasets(
    data_dir: str = "src/dataset", batch_size: int = 32, num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """Load training, validation and test datasets and create dataloaders.

    Loads image datasets from specified directory structure and creates
    dataloaders for training, validation, and testing.

    For the training dataloader, uses a BalancedClassSampler to ensure each class
    contributes the same number of samples in each epoch, equal to the size of the
    smallest class.

    Args:
        data_dir: Path to the root dataset directory
        batch_size: Number of samples per batch
        num_workers: Number of worker processes for data loading

    Returns:
        Tuple containing:
            - Training dataloader
            - Validation dataloader
            - Test dataloader
            - List of class names
    """
    transforms_obj = get_data_transforms()

    train_dir = os.path.join(data_dir, "training")
    val_dir = os.path.join(data_dir, "validation")
    test_dir = os.path.join(data_dir, "test")

    train_dataset = datasets.ImageFolder(train_dir, transform=transforms_obj)
    val_dataset = datasets.ImageFolder(val_dir, transform=transforms_obj)
    test_dataset = datasets.ImageFolder(test_dir, transform=transforms_obj)

    persistent_workers = num_workers > 0

    # Create a balanced sampler for the training dataset
    balanced_sampler = BalancedClassSampler(train_dataset)

    # Use sampler instead of shuffle for training
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=balanced_sampler,  # Use our custom sampler
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
    )

    class_names = train_dataset.classes

    return train_dataloader, val_dataloader, test_dataloader, class_names


def create_training_summary(
    session_dir: str,
    trained_model_path: str,
    training_start_time: float,
    training_args: Any,
    evaluation_results: Optional[Dict[str, Any]] = None,
    model: Optional[Any] = None,
) -> str:
    """Create a comprehensive training summary text file.

    Generates a detailed summary of the training run, including model
    configuration, training parameters, and evaluation results.

    Args:
        session_dir: Path to the session directory
        trained_model_path: Path to the saved model checkpoint
        training_start_time: Timestamp when training began
        training_args: Arguments used for training
        evaluation_results: Optional results from model evaluation
        model: Optional trained model object

    Returns:
        Path to the created summary file
    """
    summary_path = os.path.join(session_dir, "training_summary.txt")

    elapsed_time = time.time() - training_start_time
    elapsed_str = str(datetime.timedelta(seconds=int(elapsed_time)))

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("WASTE CLASSIFICATION MODEL - TRAINING SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Session Directory: {session_dir}\n")
        f.write(f"Model Path: {trained_model_path}\n")
        f.write(f"Total Training Time: {elapsed_str} (hh:mm:ss)\n\n")

        f.write("TRAINING PARAMETERS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Learning Rate: {training_args.lr}\n")
        f.write(f"Weight Decay: {training_args.weight_decay}\n")
        f.write(f"Batch Size: {training_args.batch_size}\n")
        f.write(f"Max Epochs: {training_args.max_epochs}\n")
        f.write(
            f"Cross Validation: {'Yes' if training_args.use_cross_validation else 'No'}\n"
        )
        if training_args.use_cross_validation:
            f.write(f"Number of Folds: {training_args.n_splits}\n")
        f.write(f"Workers: {training_args.num_workers}\n\n")

        if model:
            f.write("MODEL ARCHITECTURE\n")
            f.write("-" * 80 + "\n")
            f.write("Base Model: EfficientNet-B0 (pretrained on ImageNet)\n")
            f.write("Fine-tuning: Final layers (_fc, _conv_head, _bn1)\n")
            f.write(
                "Classification Head: Linear(1280→512) → ReLU → Dropout(0.3) → Linear(512→num_classes)\n"
            )

            try:
                hparams = model.hparams
                f.write("\nHYPERPARAMETERS\n")
                f.write("-" * 80 + "\n")
                for key, value in vars(hparams).items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")
            except Exception:
                f.write("Hyperparameters not available\n\n")

        if evaluation_results:
            f.write("EVALUATION RESULTS\n")
            f.write("-" * 80 + "\n")
            f.write(
                f"Test Accuracy (Primary Metric): {evaluation_results['test_accuracy']:.4f}\n"
            )
            if "test_batch_wise_accuracy" in evaluation_results:
                f.write(
                    f"Test Batch-Wise Accuracy (Lightning): {evaluation_results['test_batch_wise_accuracy']:.4f}\n"
                )
            f.write(f"Test Loss: {evaluation_results['test_loss']:.4f}\n\n")

            f.write("CLASSIFICATION REPORT\n")
            f.write("-" * 80 + "\n")
            report = evaluation_results["classification_report"]

            f.write(
                f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<12}\n"
            )
            f.write("-" * 80 + "\n")

            for class_name in evaluation_results["class_names"]:
                if class_name in report:
                    class_report = report[class_name]
                    f.write(
                        f"{class_name:<20} {class_report['precision']:<12.4f} {class_report['recall']:<12.4f} {class_report['f1-score']:<12.4f} {class_report['support']:<12}\n"
                    )

            f.write("-" * 80 + "\n")
            if "macro avg" in report:
                macro = report["macro avg"]
                f.write(
                    f"{'macro avg':<20} {macro['precision']:<12.4f} {macro['recall']:<12.4f} {macro['f1-score']:<12.4f} {macro['support']:<12}\n"
                )
            if "weighted avg" in report:
                weighted = report["weighted avg"]
                f.write(
                    f"{'weighted avg':<20} {weighted['precision']:<12.4f} {weighted['recall']:<12.4f} {weighted['f1-score']:<12.4f} {weighted['support']:<12}\n"
                )

        f.write("\n" + "=" * 80 + "\n")
        f.write("End of Summary\n")
        f.write("=" * 80 + "\n")

    print(f"Training summary saved to: {summary_path}")
    return summary_path
