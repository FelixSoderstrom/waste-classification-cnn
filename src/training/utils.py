import os
import time
import datetime
from torchvision import transforms, datasets
from torch.utils.data import DataLoader


def get_next_session_number(base_output_dir="output"):
    """
    Determine the next session number by examining existing directories.

    Args:
        base_output_dir: Base output directory

    Returns:
        int: Next session number
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


def get_data_transforms():
    """
    Get data transformations for training and validation/testing.
    No augmentation is applied since the data is already augmented.

    Returns:
        tuple: (train_transforms, val_transforms)
    """
    train_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    val_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    return train_transforms, val_transforms


def load_datasets(data_dir="src/dataset", batch_size=32, num_workers=4):
    """
    Load the training, validation, and test datasets.

    Args:
        data_dir: Path to the dataset directory
        batch_size: Batch size for the dataloaders
        num_workers: Number of workers for the dataloaders

    Returns:
        tuple: (train_dataloader, val_dataloader, test_dataloader, class_names)
    """
    train_transforms, val_transforms = get_data_transforms()

    train_dir = os.path.join(data_dir, "training")
    val_dir = os.path.join(data_dir, "validation")
    test_dir = os.path.join(data_dir, "test")

    train_dataset = datasets.ImageFolder(
        train_dir, transform=train_transforms
    )
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=val_transforms)

    persistent_workers = num_workers > 0

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
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
    session_dir,
    trained_model_path,
    training_start_time,
    training_args,
    evaluation_results=None,
    model=None,
):
    """
    Create a summary text file for the training session.

    Args:
        session_dir: Path to the session directory
        trained_model_path: Path to the trained model checkpoint
        training_start_time: Time when training started
        training_args: Arguments used for training
        evaluation_results: Results from model evaluation
        model: The trained model
    """
    summary_path = os.path.join(session_dir, "training_summary.txt")

    elapsed_time = time.time() - training_start_time
    elapsed_str = str(datetime.timedelta(seconds=int(elapsed_time)))

    with open(summary_path, "w") as f:
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
            f.write("Base Model: ResNet50 (pretrained on ImageNet)\n")
            f.write("Fine-tuning: Last layer and FC layers\n")
            f.write(
                "Classification Head: Linear(2048→512) → ReLU → Dropout(0.3) → Linear(512→num_classes)\n"
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
                f"Test Accuracy: {evaluation_results['test_accuracy']:.4f}\n"
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
