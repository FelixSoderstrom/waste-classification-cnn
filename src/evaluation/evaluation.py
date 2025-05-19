"""Model evaluation functionality for waste classification models.

This module provides functions to evaluate trained waste classification models
on test datasets, measuring performance metrics like accuracy, precision, recall,
and F1-score. It includes functionality to:
- Load test datasets
- Load and run inference with trained models
- Generate and format evaluation metrics
- Support GPU acceleration when available
"""

import os
import sys
from typing import Dict, List, Tuple, Any, Optional

import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import datasets
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from network.network import WasteClassifier
from training.utils import get_data_transforms
from evaluation.utils import get_default_model_path


def load_test_data(
    data_dir: str = "src/dataset", batch_size: int = 32, num_workers: int = 4
) -> Tuple[DataLoader, List[str]]:
    """Load the test dataset for model evaluation.

    Prepares a DataLoader for the test dataset with appropriate transforms
    and settings for efficient evaluation.

    Args:
        data_dir: Path to the root dataset directory
        batch_size: Number of samples per batch
        num_workers: Number of worker processes for data loading

    Returns:
        A tuple containing:
            - DataLoader for the test dataset
            - List of class names
    """
    transforms_obj = get_data_transforms()

    test_dir = os.path.join(data_dir, "test")
    test_dataset = datasets.ImageFolder(test_dir, transform=transforms_obj)

    persistent_workers = num_workers > 0

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
    )

    class_names = test_dataset.classes

    return test_dataloader, class_names


def calculate_overall_accuracy(
    predictions: List[int], targets: List[int]
) -> float:
    """Calculate overall accuracy across all samples.

    Unlike the batch-wise accuracy calculated by PyTorch Lightning,
    this calculates accuracy across the entire dataset, which better
    handles class imbalance.

    Args:
        predictions: List of model predictions
        targets: List of ground truth labels

    Returns:
        Overall accuracy as a float
    """
    return accuracy_score(targets, predictions)


def evaluate_model(
    model_path: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 10,
) -> Dict[str, Any]:
    """Evaluate a trained waste classification model on the test set.

    Loads a trained model, runs inference on the test dataset, and computes
    various evaluation metrics including accuracy, precision, recall, and F1-score.

    Args:
        model_path: Path to the trained model checkpoint file
        batch_size: Number of samples per batch during evaluation
        num_workers: Number of worker processes for data loading

    Returns:
        Dictionary containing comprehensive evaluation results:
            - test_accuracy: Overall accuracy on test set (calculated across all samples)
            - test_batch_wise_accuracy: Accuracy as calculated by PyTorch Lightning (batch-wise average)
            - test_loss: Loss on test set
            - classification_report: Per-class precision, recall, F1
            - confusion_matrix: Matrix showing predicted vs actual class counts
            - predictions: Raw model predictions
            - targets: Ground truth labels
            - class_names: Names of the classes
            - model: Loaded model instance
            - device: Device used for evaluation
            - model_path: Path to the model checkpoint
            - session_dir: Directory of the training session
            - primary_metric: Indicating the primary metric

    Raises:
        FileNotFoundError: If the specified model path does not exist
    """
    if model_path is None:
        model_path = get_default_model_path()
        print(f"Using automatically detected model: {model_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    pl.seed_everything(42)

    torch.set_float32_matmul_precision("high")
    print("Enabled Tensor Core optimizations with 'high' precision")

    if torch.cuda.is_available():
        print("CUDA is available! Using GPU for evaluation.")
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU being used: {gpu_name}")
    else:
        print(
            "WARNING: CUDA is not available. Evaluation on CPU will be much slower."
        )
        device = torch.device("cpu")

    print(f"Using device: {device}")

    test_dataloader, class_names = load_test_data(
        batch_size=batch_size, num_workers=num_workers
    )

    model = WasteClassifier.load_from_checkpoint(
        model_path, map_location=device
    )
    model.eval()
    model = model.to(device)

    mismatched_params = 0
    for param in model.parameters():
        if param.device != device:
            mismatched_params += 1
            param.data = param.data.to(device)

    if mismatched_params > 0:
        print(
            f"Moved {mismatched_params} parameters from {param.device} to {device}"
        )

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else "auto",
        logger=None,
        deterministic=True,
    )

    test_results = trainer.test(model, test_dataloader)

    print(f"Batch-wise test accuracy: {test_results[0]['test_acc']:.4f}")
    print(f"Test loss: {test_results[0]['test_loss']:.4f}")

    all_preds: List[int] = []
    all_targets: List[int] = []

    with torch.no_grad():
        for batch in test_dataloader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)

            if next(model.parameters()).device != device:
                model = model.to(device)

            logits = model(x)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    # Calculate overall accuracy across all samples
    overall_accuracy = calculate_overall_accuracy(all_preds, all_targets)
    print(
        f"Overall test accuracy (across all samples): {overall_accuracy:.4f}"
    )
    print(f"(This is the primary evaluation metric)")
    print(
        f"Batch-wise test accuracy (Lightning): {test_results[0]['test_acc']:.4f}"
    )

    report = classification_report(
        all_targets, all_preds, target_names=class_names, output_dict=True
    )

    df_report = pd.DataFrame(report).transpose()
    print("\nClassification Report:")
    print(df_report)

    cm = confusion_matrix(all_targets, all_preds)

    model_dir = os.path.dirname(model_path)
    session_dir = model_dir

    if os.path.basename(model_dir).startswith("fold_"):
        session_dir = os.path.dirname(model_dir)

    results = {
        "test_accuracy": overall_accuracy,  # Overall accuracy across all samples
        "test_batch_wise_accuracy": test_results[0][
            "test_acc"
        ],  # Original batch-wise accuracy
        "test_loss": test_results[0]["test_loss"],
        "classification_report": report,
        "confusion_matrix": cm,
        "predictions": all_preds,
        "targets": all_targets,
        "class_names": class_names,
        "model": model,
        "device": device,
        "model_path": model_path,
        "session_dir": session_dir,
        "primary_metric": "test_accuracy",  # Indicating the primary metric
    }

    return results


def main() -> Dict[str, Any]:
    """Run model evaluation from command line arguments.

    Parses command line arguments and runs model evaluation with the
    specified parameters.

    Returns:
        Dictionary containing the evaluation results
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate waste classification model"
    )

    default_model_path = get_default_model_path()
    parser.add_argument(
        "--model_path",
        type=str,
        default=default_model_path,
        help=f"Path to the trained model checkpoint (default: {default_model_path})",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers"
    )

    args = parser.parse_args()

    results = evaluate_model(
        model_path=args.model_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    return results


if __name__ == "__main__":
    main()
