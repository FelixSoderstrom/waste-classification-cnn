import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch
import torchvision
from PIL import Image
import random

import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.evaluation import evaluate_model
from network.network import WasteClassifier


def plot_confusion_matrix(
    cm, class_names, save_path="output/confusion_matrix.png"
):
    """
    Plot and save the confusion matrix.

    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save the confusion matrix plot
    """
    plt.figure(figsize=(12, 10))

    # Normalize confusion matrix
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    # Create heatmap
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Normalized Confusion Matrix")
    plt.tight_layout()

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save figure
    plt.savefig(save_path)
    print(f"Confusion matrix saved to {save_path}")
    plt.close()


def plot_sample_predictions(
    model,
    data_dir="src/dataset/test",
    num_samples=10,
    save_path="output/sample_predictions.png",
):
    """
    Plot sample images with their true and predicted labels.

    Args:
        model: Trained PyTorch model
        data_dir: Path to the test dataset
        num_samples: Number of sample images to plot
        save_path: Path to save the predictions plot
    """
    # Check if GPU is available - with more robust detection
    if torch.cuda.is_available():
        print("CUDA is available! Using GPU for predictions.")
        device = torch.device("cuda")
    else:
        print(
            "WARNING: CUDA is not available. Predictions on CPU will be slower."
        )
        device = torch.device("cpu")

    print(f"Using device: {device}")

    model.to(device)
    model.eval()

    # Load class names
    class_names = [
        d
        for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ]

    # Get transform for test images (no augmentation - same as in trainer.py)
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    # Collect sample images from all classes
    all_samples = []
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        image_files = [f for f in os.listdir(class_dir) if f.endswith(".jpg")]

        if len(image_files) > 0:
            # Select a random image from this class
            sample_file = random.choice(image_files)
            sample_path = os.path.join(class_dir, sample_file)

            # Load and transform the image
            image = Image.open(sample_path).convert("RGB")
            tensor = transform(image).unsqueeze(0)

            all_samples.append(
                {
                    "image": image,
                    "tensor": tensor,
                    "true_class": class_idx,
                    "true_label": class_name,
                }
            )

    # Randomly select num_samples from all collected samples
    random.shuffle(all_samples)
    selected_samples = all_samples[: min(num_samples, len(all_samples))]

    # Make predictions
    with torch.no_grad():
        for sample in selected_samples:
            tensor = sample["tensor"].to(device)
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)
            pred_class = torch.argmax(logits, dim=1).item()
            confidence = probs[0][pred_class].item()

            sample["pred_class"] = pred_class
            sample["pred_label"] = (
                class_names[pred_class]
                if pred_class < len(class_names)
                else "other"
            )
            sample["confidence"] = confidence

    # Plot the results
    fig, axes = plt.subplots(
        min(num_samples, len(selected_samples)),
        1,
        figsize=(12, 4 * min(num_samples, len(selected_samples))),
    )

    if min(num_samples, len(selected_samples)) == 1:
        axes = [axes]

    for i, sample in enumerate(selected_samples):
        ax = axes[i]

        # Display image
        ax.imshow(sample["image"])

        # Set title with true and predicted labels
        correct = sample["true_class"] == sample["pred_class"]
        color = "green" if correct else "red"

        title = f"True: {sample['true_label']}\nPred: {sample['pred_label']} ({sample['confidence']:.2f})"
        ax.set_title(title, color=color)

        # Remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save figure
    plt.savefig(save_path)
    print(f"Sample predictions saved to {save_path}")
    plt.close()


def plot_class_distribution(
    data_dir="src/dataset", save_path="output/class_distribution.png"
):
    """
    Plot the class distribution across training, validation, and test sets.

    Args:
        data_dir: Path to the dataset directory
        save_path: Path to save the class distribution plot
    """
    # Get class names from training set
    train_dir = os.path.join(data_dir, "training")
    class_names = [
        d
        for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d))
    ]

    # Count samples in each split
    counts = {"train": [], "val": [], "test": []}

    for split in ["training", "validation", "test"]:
        split_key = (
            "train"
            if split == "training"
            else "val"
            if split == "validation"
            else "test"
        )
        split_dir = os.path.join(data_dir, split)

        for class_name in class_names:
            class_dir = os.path.join(split_dir, class_name)
            if os.path.exists(class_dir):
                count = len(
                    [f for f in os.listdir(class_dir) if f.endswith(".jpg")]
                )
                counts[split_key].append(count)
            else:
                counts[split_key].append(0)

    # Plot distribution
    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot bars for each split
    ax.bar(x - width, counts["train"], width, label="Training")
    ax.bar(x, counts["val"], width, label="Validation")
    ax.bar(x + width, counts["test"], width, label="Test")

    # Add labels and legend
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of Samples")
    ax.set_title("Class Distribution Across Splits")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.legend()

    plt.tight_layout()

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save figure
    plt.savefig(save_path)
    print(f"Class distribution plot saved to {save_path}")
    plt.close()


def visualize_model_performance(model_path="output/trained_model.ckpt"):
    """
    Visualize the model's performance by plotting confusion matrix,
    sample predictions, and class distribution.

    Args:
        model_path: Path to the trained model checkpoint
    """
    # Check if GPU is available
    if torch.cuda.is_available():
        print("CUDA is available! Using GPU for visualization.")
        device = torch.device("cuda")
        # Print GPU information
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU being used: {gpu_name}")
    else:
        print(
            "WARNING: CUDA is not available. Visualization on CPU will be slower."
        )
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Load the model
    model = WasteClassifier.load_from_checkpoint(model_path)
    model.eval()
    model.to(device)

    # Evaluate the model
    results = evaluate_model(model_path=model_path)

    # Plot confusion matrix
    plot_confusion_matrix(
        cm=results["confusion_matrix"],
        class_names=results["class_names"],
        save_path="output/confusion_matrix.png",
    )

    # Plot sample predictions
    plot_sample_predictions(
        model=model,
        data_dir="src/dataset/test",
        num_samples=10,
        save_path="output/sample_predictions.png",
    )

    # Plot class distribution
    plot_class_distribution(
        data_dir="src/dataset", save_path="output/class_distribution.png"
    )


def main():
    """Main function to visualize model performance."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize model performance"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="output/trained_model.ckpt",
        help="Path to the trained model checkpoint",
    )

    args = parser.parse_args()

    # Visualize model performance
    visualize_model_performance(model_path=args.model_path)


if __name__ == "__main__":
    main()
