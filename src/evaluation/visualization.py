import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torchvision
from PIL import Image
import random
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from evaluation.utils import get_default_model_path
from training.utils import get_data_transforms


def plot_confusion_matrix(cm, class_names, save_path):
    """
    Plot and save the confusion matrix.

    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save the confusion matrix plot
    """
    plt.figure(figsize=(12, 10))

    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

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

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.savefig(save_path)
    print(f"Confusion matrix saved to {save_path}")
    plt.close()


def plot_sample_predictions(
    model,
    device,
    data_dir="src/dataset/test",
    num_samples=10,
    save_path="output/sample_predictions.png",
):
    """
    Plot sample images with their true and predicted labels.

    Args:
        model: Trained PyTorch model
        device: Device to run inference on
        data_dir: Path to the test dataset
        num_samples: Number of sample images to plot
        save_path: Path to save the predictions plot
    """
    model = model.to(device)
    model.eval()

    class_names = [
        d
        for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ]

    _, transform = get_data_transforms()

    all_samples = []
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        image_files = [f for f in os.listdir(class_dir) if f.endswith(".jpg")]

        if len(image_files) > 0:
            sample_file = random.choice(image_files)
            sample_path = os.path.join(class_dir, sample_file)

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

    random.shuffle(all_samples)
    selected_samples = all_samples[: min(num_samples, len(all_samples))]

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

    fig, axes = plt.subplots(
        min(num_samples, len(selected_samples)),
        1,
        figsize=(12, 4 * min(num_samples, len(selected_samples))),
    )

    if min(num_samples, len(selected_samples)) == 1:
        axes = [axes]

    for i, sample in enumerate(selected_samples):
        ax = axes[i]
        ax.imshow(sample["image"])

        correct = sample["true_class"] == sample["pred_class"]
        color = "green" if correct else "red"

        title = f"True: {sample['true_label']}\nPred: {sample['pred_label']} ({sample['confidence']:.2f})"
        ax.set_title(title, color=color)

        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

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
    train_dir = os.path.join(data_dir, "training")
    class_names = [
        d
        for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d))
    ]

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

    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 8))

    ax.bar(x - width, counts["train"], width, label="Training")
    ax.bar(x, counts["val"], width, label="Validation")
    ax.bar(x + width, counts["test"], width, label="Test")

    ax.set_xlabel("Class")
    ax.set_ylabel("Number of Samples")
    ax.set_title("Class Distribution Across Splits")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.legend()

    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.savefig(save_path)
    print(f"Class distribution plot saved to {save_path}")
    plt.close()


def create_visualizations(
    evaluation_results, model, device, session_dir=None
):
    """
    Create visualizations based on evaluation results.

    Args:
        evaluation_results: Results from evaluate_model
        model: Trained model
        device: Device to run inference on
        session_dir: Path to the session directory
    """
    if session_dir is None:
        model_path = evaluation_results.get("model_path", "")
        model_dir = os.path.dirname(model_path)
        session_dir = model_dir

        if os.path.basename(model_dir).startswith("fold_"):
            session_dir = os.path.dirname(model_dir)

    class_names = evaluation_results["class_names"]

    plots_dir = os.path.join(session_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    print(f"Creating visualizations in: {plots_dir}")

    confusion_matrix_path = os.path.join(plots_dir, "confusion_matrix.png")
    sample_predictions_path = os.path.join(
        plots_dir, "sample_predictions.png"
    )
    class_distribution_path = os.path.join(
        plots_dir, "class_distribution.png"
    )

    plot_confusion_matrix(
        cm=evaluation_results["confusion_matrix"],
        class_names=class_names,
        save_path=confusion_matrix_path,
    )

    plot_sample_predictions(
        model=model,
        device=device,
        data_dir="src/dataset/test",
        num_samples=10,
        save_path=sample_predictions_path,
    )

    plot_class_distribution(
        data_dir="src/dataset", save_path=class_distribution_path
    )

    return {
        "confusion_matrix_path": confusion_matrix_path,
        "sample_predictions_path": sample_predictions_path,
        "class_distribution_path": class_distribution_path,
    }
