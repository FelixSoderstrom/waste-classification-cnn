import os
import torch
import pytorch_lightning as pl
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import pandas as pd

import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from network.network import WasteClassifier
from training.trainer import get_data_transforms


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


def evaluate_model(
    model_path="output/trained_model.ckpt", batch_size=32, num_workers=4
):
    """
    Evaluate a trained model on the test set.

    Args:
        model_path: Path to the trained model checkpoint
        batch_size: Batch size for the dataloader
        num_workers: Number of workers for the dataloader

    Returns:
        dict: Evaluation results including accuracy, precision, recall, f1-score
    """
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
    parser.add_argument(
        "--model_path",
        type=str,
        default="output/trained_model.ckpt",
        help="Path to the trained model checkpoint",
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
