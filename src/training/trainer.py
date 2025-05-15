"""Training functionality for waste classification models.

This module contains functions to train waste classification models using
either standard training or k-fold cross-validation. It handles:
- Setting up model parameters and training environments
- Configuring callbacks like early stopping and checkpointing
- Integrating with Weights & Biases for experiment tracking
- GPU acceleration when available
"""

import os
import sys
import time
from typing import List, Optional

import torch
import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold

# Add parent directory to path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from network.network import WasteClassifier
from training.utils import (
    get_next_session_number,
    load_datasets,
)


def train_model(
    lr: float = 0.001,
    weight_decay: float = 0.0005,
    batch_size: int = 32,
    max_epochs: int = 10,
    num_workers: int = 4,
    project_name: str = "waste-material-classification",
    use_cross_validation: bool = True,
    n_splits: int = 5,
) -> pl.LightningModule:
    """Train the waste classification model.

    Prepares the environment and training settings, then delegates to either
    standard training or cross-validation depending on configuration.

    Args:
        lr: Learning rate for optimizer
        weight_decay: Weight decay coefficient for optimizer
        batch_size: Number of samples per batch
        max_epochs: Maximum number of training epochs
        num_workers: Number of workers for the dataloaders
        project_name: Name of the project for W&B logging
        use_cross_validation: Whether to use cross-validation
        n_splits: Number of cross-validation splits

    Returns:
        The trained PyTorch Lightning model
    """
    pl.seed_everything(42)

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        print("CUDA is available. Using GPU for training.")
        print("Enabled Tensor Core optimizations with 'high' precision")
        print(f"GPU being used: {gpu_name}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
    else:
        print(
            "WARNING: CUDA is not available. Training on CPU will be much slower."
        )
        device = torch.device("cpu")

    print(f"Using device: {device}")

    train_dataloader, val_dataloader, test_dataloader, class_names = (
        load_datasets(batch_size=batch_size, num_workers=num_workers)
    )

    session_number = get_next_session_number()
    session_dir = f"output/session_{session_number}"
    os.makedirs(session_dir, exist_ok=True)
    print(f"Created session directory: {session_dir}")

    if use_cross_validation:
        return train_with_cross_validation(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            test_dataloader=test_dataloader,
            class_names=class_names,
            lr=lr,
            weight_decay=weight_decay,
            max_epochs=max_epochs,
            project_name=project_name,
            n_splits=n_splits,
            session_number=session_number,
            session_dir=session_dir,
        )
    return train_standard(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        class_names=class_names,
        lr=lr,
        weight_decay=weight_decay,
        max_epochs=max_epochs,
        project_name=project_name,
        session_number=session_number,
        session_dir=session_dir,
    )


def train_standard(
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    test_dataloader: DataLoader,
    class_names: List[str],
    lr: float = 0.001,
    weight_decay: float = 0.0005,
    max_epochs: int = 10,
    project_name: str = "waste-material-classification",
    session_number: int = 1,
    session_dir: Optional[str] = None,
) -> pl.LightningModule:
    """Train model using standard training with early stopping.

    Configures the model, trainer, and callbacks for a standard training run
    (no cross-validation). Includes early stopping and model checkpointing.

    Args:
        train_dataloader: DataLoader for training data
        val_dataloader: DataLoader for validation data
        test_dataloader: DataLoader for test data
        class_names: List of class names for the dataset
        lr: Learning rate for optimizer
        weight_decay: Weight decay coefficient for optimizer
        max_epochs: Maximum number of training epochs
        project_name: Name of the project for W&B logging
        session_number: Current session number for organization
        session_dir: Directory where outputs will be saved

    Returns:
        The trained PyTorch Lightning model
    """
    if wandb.run is not None:
        wandb.finish()

    model = WasteClassifier(
        lr=lr, weight_decay=weight_decay, class_names=class_names
    )

    run_name = f"session_{session_number}-standard"
    unique_id = f"{run_name}_{int(time.time())}"

    wandb_logger = WandbLogger(
        project=project_name,
        name=run_name,
        id=unique_id,
        group=f"session_{session_number}",
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=session_dir,
        filename="waste-classifier-{epoch:02d}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
    )

    early_stopping_callback = EarlyStopping(
        monitor="val_loss", patience=5, mode="min", verbose=True
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else "auto",
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stopping_callback, lr_monitor],
        log_every_n_steps=10,
        deterministic=True,
    )

    trainer.fit(model, train_dataloader, val_dataloader)
    trainer.test(model, test_dataloader)

    final_model_path = os.path.join(session_dir, "trained_model.ckpt")
    trainer.save_checkpoint(final_model_path)
    print(f"Model saved to: {final_model_path}")

    wandb.finish()

    return model


def train_with_cross_validation(
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    test_dataloader: DataLoader,
    class_names: List[str],
    lr: float = 0.001,
    weight_decay: float = 0.0005,
    max_epochs: int = 10,
    project_name: str = "waste-material-classification",
    n_splits: int = 5,
    session_number: int = 1,
    session_dir: Optional[str] = None,
) -> pl.LightningModule:
    """Train model using k-fold cross-validation.

    Performs k-fold cross-validation by splitting the training data into
    train and validation sets for each fold. Returns the best model based
    on validation score across folds.

    Args:
        train_dataloader: DataLoader for training data
        val_dataloader: DataLoader for validation data
        test_dataloader: DataLoader for test data
        class_names: List of class names for the dataset
        lr: Learning rate for optimizer
        weight_decay: Weight decay coefficient for optimizer
        max_epochs: Maximum number of training epochs
        project_name: Name of the project for W&B logging
        n_splits: Number of cross-validation folds
        session_number: Current session number for organization
        session_dir: Directory where outputs will be saved

    Returns:
        The best model from cross-validation based on validation accuracy
    """
    val_scores: List[float] = []
    best_model_path: Optional[str] = None
    best_val_score: float = 0.0

    train_dataset = train_dataloader.dataset
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    persistent_workers = train_dataloader.num_workers > 0

    for fold, (train_idx, val_idx) in enumerate(
        kfold.split(range(len(train_dataset)))
    ):
        if wandb.run is not None:
            wandb.finish()

        fold_number = fold + 1
        print(f"Fold {fold_number}/{n_splits}")

        fold_dir = os.path.join(session_dir, f"fold_{fold_number}")
        os.makedirs(fold_dir, exist_ok=True)

        fold_train_subset = Subset(train_dataset, train_idx)
        fold_val_subset = Subset(train_dataset, val_idx)

        fold_train_loader = DataLoader(
            fold_train_subset,
            batch_size=train_dataloader.batch_size,
            shuffle=True,
            num_workers=train_dataloader.num_workers,
            pin_memory=True,
            persistent_workers=persistent_workers,
        )

        fold_val_loader = DataLoader(
            fold_val_subset,
            batch_size=train_dataloader.batch_size,
            shuffle=False,
            num_workers=train_dataloader.num_workers,
            pin_memory=True,
            persistent_workers=persistent_workers,
        )

        model = WasteClassifier(
            lr=lr, weight_decay=weight_decay, class_names=class_names
        )

        run_name = f"session_{session_number}-fold_{fold_number}"
        unique_id = f"{run_name}_{int(time.time())}"

        wandb_logger = WandbLogger(
            project=project_name,
            name=run_name,
            id=unique_id,
            group=f"session_{session_number}",
        )

        checkpoint_callback = ModelCheckpoint(
            dirpath=fold_dir,
            filename="waste-classifier-{epoch:02d}-{val_loss:.2f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        )

        early_stopping_callback = EarlyStopping(
            monitor="val_loss", patience=5, mode="min", verbose=True
        )

        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1 if torch.cuda.is_available() else "auto",
            logger=wandb_logger,
            callbacks=[checkpoint_callback, early_stopping_callback],
            log_every_n_steps=10,
            deterministic=True,
        )

        trainer.fit(model, fold_train_loader, fold_val_loader)

        val_result = trainer.validate(model, val_dataloader)
        val_score = val_result[0]["val_acc"]
        val_scores.append(val_score)

        if val_score > best_val_score:
            best_val_score = val_score
            best_model_path = checkpoint_callback.best_model_path

        wandb.finish()

    if wandb.run is not None:
        wandb.finish()

    best_model = WasteClassifier.load_from_checkpoint(best_model_path)

    best_run_name = f"session_{session_number}-best-model"
    best_run_id = f"{best_run_name}_{int(time.time())}"

    test_trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else "auto",
        logger=WandbLogger(
            project=project_name,
            name=best_run_name,
            id=best_run_id,
            group=f"session_{session_number}",
        ),
        deterministic=True,
    )

    test_trainer.test(best_model, test_dataloader)

    final_model_path = os.path.join(session_dir, "trained_model.ckpt")
    test_trainer.save_checkpoint(final_model_path)
    print(f"Best model saved to: {final_model_path}")

    wandb.finish()

    return best_model


if __name__ == "__main__":
    print(
        "This file is not meant to be executed directly.\n"
        "If you want to train a new model, run train.py instead.\n"
        "You can also run it with the '--h' flag to see the available arguments."
    )
