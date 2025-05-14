import os
import torch
import pytorch_lightning as pl
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

# import pandas as pd
import glob
import re

import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from network.network import WasteClassifier
from training.trainer import get_data_transforms


def find_latest_session_dir(base_dir="output"):
    """
    Find the latest session directory based on its number.

    Args:
        base_dir: Base directory to search in

    Returns:
        str: Path to the latest session directory, or None if no session directories found
    """
    if not os.path.exists(base_dir):
        return None

    session_dirs = []
    for d in os.listdir(base_dir):
        full_path = os.path.join(base_dir, d)
        if os.path.isdir(full_path) and d.startswith("session_"):
            try:
                # Extract session number from directory name
                session_num = int(d.split("_")[1])
                session_dirs.append((session_num, full_path))
            except (IndexError, ValueError):
                continue

    if not session_dirs:
        return None

    # Sort by session number (descending) and return the latest
    latest_session = sorted(session_dirs, key=lambda x: x[0], reverse=True)[
        0
    ][1]
    return latest_session


def find_checkpoint_in_dir(session_dir):
    """
    Find a checkpoint file in the given directory.
    Prioritizes 'trained_model.ckpt' if it exists.

    Args:
        session_dir: Directory to search in

    Returns:
        str: Path to a checkpoint file, or None if no checkpoint found
    """
    if not os.path.exists(session_dir):
        return None

    # First, check if trained_model.ckpt exists (top priority)
    trained_model_path = os.path.join(session_dir, "trained_model.ckpt")
    if os.path.exists(trained_model_path):
        return trained_model_path

    # Otherwise, look for any .ckpt file in the directory
    checkpoint_files = glob.glob(os.path.join(session_dir, "*.ckpt"))

    # If no checkpoints in the main directory, look in fold subdirectories
    if not checkpoint_files:
        for d in os.listdir(session_dir):
            fold_dir = os.path.join(session_dir, d)
            if os.path.isdir(fold_dir) and d.startswith("fold_"):
                checkpoint_files.extend(
                    glob.glob(os.path.join(fold_dir, "*.ckpt"))
                )

    if checkpoint_files:
        # Sort by modification time to get the latest
        return sorted(checkpoint_files, key=os.path.getmtime)[-1]

    return None


def get_default_model_path():
    """
    Automatically find the latest model checkpoint to use as default.

    Returns:
        str: Path to the latest model checkpoint, or fallback default if none found
    """
    fallback_path = "output/trained_model.ckpt"

    latest_session = find_latest_session_dir()
    if not latest_session:
        return fallback_path

    checkpoint_path = find_checkpoint_in_dir(latest_session)
    if not checkpoint_path:
        return fallback_path

    return checkpoint_path


def load_test_data(data_dir="src/dataset", batch_size=32, num_workers=4):
    """
    Load the test dataset.

    Args:
        data_dir: Path to the dataset directory
        batch_size: Batch size for the dataloader
        num_workers: Number of workers for the dataloader

    Returns:
        tuple: (test_dataloader, class_names)
    """
    _, val_transforms = get_data_transforms()

    # Load test dataset
    test_dir = os.path.join(data_dir, "test")
    test_dataset = datasets.ImageFolder(test_dir, transform=val_transforms)

    # Create dataloader
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Get class names
    class_names = test_dataset.classes

    return test_dataloader, class_names


def evaluate_model(model_path=None, batch_size=32, num_workers=4):
    """
    Evaluate a trained model on the test set.

    Args:
        model_path: Path to the trained model checkpoint
        batch_size: Batch size for the dataloader
        num_workers: Number of workers for the dataloader

    Returns:
        dict: Evaluation results including accuracy, precision, recall, f1-score
    """
    # Use default path finder if no path provided
    if model_path is None:
        model_path = get_default_model_path()
        print(f"Using automatically detected model: {model_path}")

    # Check if model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    # Set random seed for reproducibility
    pl.seed_everything(42)

    # Check if GPU is available - with more robust detection
    if torch.cuda.is_available():
        print("CUDA is available! Using GPU for evaluation.")
        device = torch.device("cuda")
        # Print GPU information
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU being used: {gpu_name}")
    else:
        print(
            "WARNING: CUDA is not available. Evaluation on CPU will be much slower."
        )
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Load test data
    test_dataloader, class_names = load_test_data(
        batch_size=batch_size, num_workers=num_workers
    )

    # Load trained model
    model = WasteClassifier.load_from_checkpoint(model_path)
    model.eval()
    model.to(device)

    # Set up trainer for testing with explicit GPU configuration
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else "auto",
        logger=None,
        deterministic=True,
    )

    # Test the model
    test_results = trainer.test(model, test_dataloader)

    print(f"Test accuracy: {test_results[0]['test_acc']:.4f}")
    print(f"Test loss: {test_results[0]['test_loss']:.4f}")

    # Get predictions
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in test_dataloader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    # Calculate metrics
    report = classification_report(
        all_targets, all_preds, target_names=class_names, output_dict=True
    )

    # Convert classification report to DataFrame for easier viewing
    df_report = pd.DataFrame(report).transpose()
    print("\nClassification Report:")
    print(df_report)

    # Calculate confusion matrix
    cm = confusion_matrix(all_targets, all_preds)

    # Return evaluation results
    results = {
        "test_accuracy": test_results[0]["test_acc"],
        "test_loss": test_results[0]["test_loss"],
        "classification_report": report,
        "confusion_matrix": cm,
        "predictions": all_preds,
        "targets": all_targets,
        "class_names": class_names,
    }

    return results


def main():
    """Main function to evaluate the model."""
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

    # Evaluate model
    results = evaluate_model(
        model_path=args.model_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    return results


if __name__ == "__main__":
    main()
