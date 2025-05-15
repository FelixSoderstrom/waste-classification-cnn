import os
import torch
import pytorch_lightning as pl
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import pandas as pd
import sys
import glob

from network.network import WasteClassifier
from training.utils import get_data_transforms
from evaluation.utils import get_default_model_path

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


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

    test_dir = os.path.join(data_dir, "test")
    test_dataset = datasets.ImageFolder(test_dir, transform=val_transforms)

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


def evaluate_model(model_path=None, batch_size=32, num_workers=10):
    """
    Evaluate a trained model on the test set.

    Args:
        model_path: Path to the trained model checkpoint
        batch_size: Batch size for the dataloader
        num_workers: Number of workers for the dataloader

    Returns:
        dict: Evaluation results including accuracy, precision, recall, f1-score,
              along with the loaded model and device for visualization
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

    print(f"Test accuracy: {test_results[0]['test_acc']:.4f}")
    print(f"Test loss: {test_results[0]['test_loss']:.4f}")

    all_preds = []
    all_targets = []

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
        "test_accuracy": test_results[0]["test_acc"],
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

    results = evaluate_model(
        model_path=args.model_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    return results


if __name__ == "__main__":
    main()
