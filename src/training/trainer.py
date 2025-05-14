import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from sklearn.model_selection import KFold
import re
import time
import wandb

import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from network.network import WasteClassifier


def get_next_session_number(base_output_dir="output"):
    """
    Determine the next session number by examining existing directories.

    Args:
        base_output_dir: Base output directory

    Returns:
        int: Next session number
    """
    os.makedirs(base_output_dir, exist_ok=True)

    # Get all directories that follow the pattern 'session_X'
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
    # Training transforms without augmentation
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

    # Validation/testing transforms (same as training now)
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

    # Load datasets
    train_dir = os.path.join(data_dir, "training")
    val_dir = os.path.join(data_dir, "validation")
    test_dir = os.path.join(data_dir, "test")

    train_dataset = datasets.ImageFolder(
        train_dir, transform=train_transforms
    )
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=val_transforms)

    # Create dataloaders with persistent workers for better performance
    persistent_workers = num_workers > 0

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,  # Keep workers alive between iterations
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,  # Keep workers alive between iterations
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,  # Keep workers alive between iterations
    )

    # Get class names from the training dataset
    class_names = train_dataset.classes

    return train_dataloader, val_dataloader, test_dataloader, class_names


def train_model(
    lr=0.001,
    weight_decay=0.0005,
    batch_size=32,
    max_epochs=10,
    num_workers=4,
    project_name="waste-material-classification",
    use_cross_validation=True,
    n_splits=5,
):
    """
    Train the waste classification model.

    Args:
        lr: Learning rate
        weight_decay: Weight decay
        batch_size: Batch size
        max_epochs: Maximum number of epochs
        num_workers: Number of workers for the dataloaders
        project_name: Name of the project for W&B logging
        use_cross_validation: Whether to use cross-validation
        n_splits: Number of cross-validation splits

    Returns:
        pl.LightningModule: The trained model
    """
    # Set random seed for reproducibility
    pl.seed_everything(42)

    # Enable Tensor Core optimizations for RTX GPUs
    torch.set_float32_matmul_precision(
        "medium"
    )  # Balance between speed and precision
    print("Enabled Tensor Core optimizations with 'medium' precision")

    # Check if GPU is available - with more robust detection
    if torch.cuda.is_available():
        print("CUDA is available! Using GPU for training.")
        device = torch.device("cuda")
        # Print GPU information
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU being used: {gpu_name}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
    else:
        print(
            "WARNING: CUDA is not available. Training on CPU will be much slower."
        )
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Load datasets
    train_dataloader, val_dataloader, test_dataloader, class_names = (
        load_datasets(batch_size=batch_size, num_workers=num_workers)
    )

    # Create session directory based on next available session number
    session_number = get_next_session_number()
    session_dir = f"output/session_{session_number}"
    os.makedirs(session_dir, exist_ok=True)
    print(f"Created session directory: {session_dir}")

    if use_cross_validation:
        # Cross-validation training
        return train_with_cross_validation(
            train_dataloader,
            val_dataloader,
            test_dataloader,
            class_names,
            lr=lr,
            weight_decay=weight_decay,
            max_epochs=max_epochs,
            project_name=project_name,
            n_splits=n_splits,
            session_number=session_number,
            session_dir=session_dir,
        )
    else:
        # Standard training with early stopping
        return train_standard(
            train_dataloader,
            val_dataloader,
            test_dataloader,
            class_names,
            lr=lr,
            weight_decay=weight_decay,
            max_epochs=max_epochs,
            project_name=project_name,
            session_number=session_number,
            session_dir=session_dir,
        )


def train_standard(
    train_dataloader,
    val_dataloader,
    test_dataloader,
    class_names,
    lr=0.001,
    weight_decay=0.0005,
    max_epochs=10,
    project_name="waste-material-classification",
    session_number=1,
    session_dir="output/session_1",
):
    """
    Train the model using standard training with early stopping.

    Args:
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
        test_dataloader: Test data loader
        class_names: List of class names
        lr: Learning rate
        weight_decay: Weight decay
        max_epochs: Maximum number of epochs
        project_name: Name of the project for W&B logging
        session_number: Current session number
        session_dir: Directory for session outputs

    Returns:
        pl.LightningModule: The trained model
    """
    # Make sure any existing wandb run is finished
    if wandb.run is not None:
        wandb.finish()

    # Initialize the model
    model = WasteClassifier(
        lr=lr, weight_decay=weight_decay, class_names=class_names
    )

    # Set up Weights & Biases logger with session-specific run name
    run_name = f"session_{session_number}-standard"
    unique_id = f"{run_name}_{int(time.time())}"

    wandb_logger = WandbLogger(
        project=project_name,
        name=run_name,
        id=unique_id,
        group=f"session_{session_number}",
    )

    # Set up callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=session_dir,
        filename="waste-classifier-{epoch:02d}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
    )

    early_stopping_callback = EarlyStopping(
        monitor="val_loss", patience=3, mode="min", verbose=True
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Set up trainer with explicit GPU configuration
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else "auto",
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stopping_callback, lr_monitor],
        log_every_n_steps=10,
        deterministic=True,
    )

    # Train the model
    trainer.fit(model, train_dataloader, val_dataloader)

    # Test the model on the test set
    trainer.test(model, test_dataloader)

    # Save the final model
    final_model_path = os.path.join(session_dir, "trained_model.ckpt")
    trainer.save_checkpoint(final_model_path)
    print(f"Model saved to: {final_model_path}")

    # Make sure to finish the wandb run
    wandb.finish()

    return model


def train_with_cross_validation(
    train_dataloader,
    val_dataloader,
    test_dataloader,
    class_names,
    lr=0.001,
    weight_decay=0.0005,
    max_epochs=10,
    project_name="waste-material-classification",
    n_splits=5,
    session_number=1,
    session_dir="output/session_1",
):
    """
    Train the model using cross-validation.

    Args:
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
        test_dataloader: Test data loader
        class_names: List of class names
        lr: Learning rate
        weight_decay: Weight decay
        max_epochs: Maximum number of epochs
        project_name: Name of the project for W&B logging
        n_splits: Number of cross-validation splits
        session_number: Current session number
        session_dir: Directory for session outputs

    Returns:
        pl.LightningModule: The best model from cross-validation
    """
    # Get the training dataset
    train_dataset = train_dataloader.dataset

    # Initialize KFold
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Initialize list to store validation scores
    val_scores = []

    # Initialize best model and score
    best_model_path = None
    best_val_score = 0

    # Check if we can use persistent workers
    persistent_workers = train_dataloader.num_workers > 0

    # Loop through folds
    for fold, (train_idx, val_idx) in enumerate(
        kfold.split(range(len(train_dataset)))
    ):
        # Make sure any existing wandb run is finished before starting a new one
        if wandb.run is not None:
            wandb.finish()

        fold_number = fold + 1
        print(f"Fold {fold_number}/{n_splits}")

        # Create fold directory
        fold_dir = os.path.join(session_dir, f"fold_{fold_number}")
        os.makedirs(fold_dir, exist_ok=True)

        # Create dataloaders for this fold
        fold_train_subset = torch.utils.data.Subset(train_dataset, train_idx)
        fold_val_subset = torch.utils.data.Subset(train_dataset, val_idx)

        fold_train_loader = DataLoader(
            fold_train_subset,
            batch_size=train_dataloader.batch_size,
            shuffle=True,
            num_workers=train_dataloader.num_workers,
            pin_memory=True,
            persistent_workers=persistent_workers,  # Keep workers alive between iterations
        )

        fold_val_loader = DataLoader(
            fold_val_subset,
            batch_size=train_dataloader.batch_size,
            shuffle=False,
            num_workers=train_dataloader.num_workers,
            pin_memory=True,
            persistent_workers=persistent_workers,  # Keep workers alive between iterations
        )

        # Initialize the model for this fold
        model = WasteClassifier(
            lr=lr, weight_decay=weight_decay, class_names=class_names
        )

        # Set up Weights & Biases logger with unique run ID to prevent overwriting
        run_name = f"session_{session_number}-fold_{fold_number}"
        unique_id = f"{run_name}_{int(time.time())}"

        wandb_logger = WandbLogger(
            project=project_name,
            name=run_name,
            id=unique_id,
            group=f"session_{session_number}",  # Group all folds from this session
        )

        # Set up callbacks
        checkpoint_callback = ModelCheckpoint(
            dirpath=fold_dir,
            filename="waste-classifier-{epoch:02d}-{val_loss:.2f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        )

        early_stopping_callback = EarlyStopping(
            monitor="val_loss", patience=3, mode="min", verbose=True
        )

        # Set up trainer for this fold with explicit GPU configuration
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1 if torch.cuda.is_available() else "auto",
            logger=wandb_logger,
            callbacks=[checkpoint_callback, early_stopping_callback],
            log_every_n_steps=10,
            deterministic=True,
        )

        # Train the model for this fold
        trainer.fit(model, fold_train_loader, fold_val_loader)

        # Validate on the validation set
        val_result = trainer.validate(model, val_dataloader)
        val_score = val_result[0]["val_acc"]
        val_scores.append(val_score)

        # Update best model if this fold is better
        if val_score > best_val_score:
            best_val_score = val_score
            best_model_path = checkpoint_callback.best_model_path

        # Explicitly finish the wandb run
        wandb.finish()

    # Make sure any existing wandb run is finished before creating final one
    if wandb.run is not None:
        wandb.finish()

    # Load the best model from cross-validation
    best_model = WasteClassifier.load_from_checkpoint(best_model_path)

    # Create a final trainer for testing with explicit GPU configuration
    best_run_name = f"session_{session_number}-best-model"
    best_run_id = f"{best_run_name}_{int(time.time())}"

    test_trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else "auto",
        logger=WandbLogger(
            project=project_name,
            name=best_run_name,
            id=best_run_id,
            group=f"session_{session_number}",  # Same group as the folds
        ),
        deterministic=True,
    )

    # Test the best model on the test set
    test_trainer.test(best_model, test_dataloader)

    # Save the final model
    final_model_path = os.path.join(session_dir, "trained_model.ckpt")
    test_trainer.save_checkpoint(final_model_path)
    print(f"Best model saved to: {final_model_path}")

    # Make sure to finish the final wandb run
    wandb.finish()

    return best_model


if __name__ == "__main__":
    print(
        "This file is not meant to be executed directly.\n"
        "If you want to train a new model, run train.py instead.\n"
        "You can also run it with the '--h' flag to see the available arguments."
    )
