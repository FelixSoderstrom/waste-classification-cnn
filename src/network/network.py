import torch
import torch.nn as nn
import torchvision.models as models
import pytorch_lightning as pl
import numpy as np
from typing import Dict, Any, List, Optional, Union


class WasteClassifier(pl.LightningModule):
    """
    Transfer learning model based on ResNet50 for waste classification.
    The model is trained on 10 categories of waste materials.
    """

    def __init__(
        self,
        lr: float = 0.001,
        weight_decay: float = 0.0005,
        num_classes: int = 10,
        class_names: list = None,
        class_weights: Optional[Union[List[float], torch.Tensor]] = None,
    ):
        """
        Initialize the WasteClassifier model.

        Args:
            lr: Learning rate for the optimizer
            weight_decay: Weight decay for the optimizer
            num_classes: Number of output classes
            class_names: List of class names
            class_weights: Optional tensor of class weights for loss function.
                           If None, weights will be calculated automatically
                           from the class distribution.
        """
        super().__init__()

        # Save hyperparameters
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay

        # Store class weights if provided
        self.class_weights = class_weights

        # Load pretrained ResNet50 model
        self.model = models.resnet50(weights="IMAGENET1K_V2")

        # Freeze all layers except the last few
        for name, param in self.model.named_parameters():
            if "layer4" not in name and "fc" not in name:
                param.requires_grad = False

        # Replace the final fully connected layer for our classification task
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

        # Class names
        self.class_names = (
            class_names
            if class_names
            else [
                "battery",
                "biological",
                "cardboard",
                "clothes",
                "glass",
                "metal",
                "paper",
                "plastic",
                "shoes",
                "trash",
            ]
        )

        # Initialize loss function without weights, will be set in on_train_start
        # This allows us to calculate weights automatically from dataset if not provided
        self.criterion = nn.CrossEntropyLoss()

        # Flag to track if weights have been applied to the criterion
        self.weights_applied = False

    def on_save_checkpoint(self, checkpoint):
        """Save custom attributes when saving checkpoint."""
        # Store class weights if they exist
        if hasattr(self, "class_weights") and self.class_weights is not None:
            checkpoint["class_weights"] = self.class_weights.detach().cpu()

        # This helps us know if weights were applied
        checkpoint["weights_applied"] = self.weights_applied

        return checkpoint

    def on_load_checkpoint(self, checkpoint):
        """Load custom attributes when loading checkpoint."""
        # Load class weights if they exist in the checkpoint
        if "class_weights" in checkpoint:
            self.class_weights = checkpoint["class_weights"]

            # If weights were applied to criterion in the saved model
            if checkpoint.get("weights_applied", False):
                # Move weights to the correct device
                device = (
                    self.device
                    if hasattr(self, "device")
                    else torch.device("cpu")
                )
                self.class_weights = self.class_weights.to(device)

                # Re-create criterion with weights
                self.criterion = nn.CrossEntropyLoss(
                    weight=self.class_weights
                )
                self.weights_applied = True

    def on_train_start(self):
        """
        Called at the beginning of training.
        If class weights weren't provided, calculate them from the dataset.
        """
        # Skip if weights have already been applied
        if self.weights_applied:
            return

        # If weights already manually set, skip automatic calculation
        if self.class_weights is not None:
            # Convert to tensor if it's a list
            if isinstance(self.class_weights, list):
                self.class_weights = torch.tensor(self.class_weights)

            # Move to same device as model
            self.class_weights = self.class_weights.to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
            self.weights_applied = True
            print(f"Using provided class weights: {self.class_weights}")
            return

        # Use a pre-defined distribution based on the current dataset structure
        # (Could also be calculated dynamically from the dataset if needed)
        class_counts = {
            "battery": 3376,
            "biological": 3588,
            "cardboard": 3248,
            "clothes": 4926,
            "glass": 2654,
            "metal": 3664,
            "paper": 2956,
            "plastic": 3552,
            "shoes": 3554,
            "trash": 3356,
        }

        # Get counts in the same order as class_names
        counts = [class_counts.get(name, 1) for name in self.class_names]

        # Calculate inverse frequencies and normalize
        frequencies = np.array(counts) / sum(counts)
        inv_frequencies = 1.0 / (
            frequencies + 1e-8
        )  # Add small epsilon to avoid division by zero
        weights = inv_frequencies / inv_frequencies.sum()

        # Convert to tensor and move to device
        self.class_weights = torch.tensor(weights, dtype=torch.float32).to(
            self.device
        )

        # Update loss function with calculated weights
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        self.weights_applied = True

        print(f"Automatically calculated class weights: {self.class_weights}")

    def forward(self, x):
        """Forward pass through the network."""
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """Training step for the model."""
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # Log training metrics
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True
        )
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step for the model."""
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # Log validation metrics
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

        return {"val_loss": loss, "val_acc": acc}

    def test_step(self, batch, batch_idx):
        """Test step for the model."""
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # Log test metrics
        preds = torch.argmax(logits, dim=1)
        # This is batch-wise accuracy, not the primary evaluation metric
        # The overall accuracy across all samples is calculated in evaluation.py
        acc = (preds == y).float().mean()
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True, add_dataloader_idx=False)
        self.log(
            "test_batch_acc", acc, prog_bar=False
        )  # Explicitly named batch accuracy

        # Return predictions and targets for later use in confusion matrix and overall accuracy
        return {
            "test_loss": loss,
            "test_acc": acc,
            "preds": preds,
            "targets": y,
        }

    def predict_step(self, batch, batch_idx):
        """Prediction step for the model."""
        x, y = batch
        logits = self(x)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)

        return {
            "logits": logits,
            "probs": probs,
            "preds": preds,
            "targets": y,
        }

    def configure_optimizers(self):
        """Configure the optimizer for training."""
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer
