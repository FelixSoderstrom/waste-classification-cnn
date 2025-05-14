import torch
import torch.nn as nn
import torchvision.models as models
import pytorch_lightning as pl
from typing import Dict, Any


class WasteClassifier(pl.LightningModule):
    """
    Transfer learning model based on ResNet50 for waste classification.
    The model is trained on 10 categories of waste materials plus an 'other' category.
    """

    def __init__(
        self,
        lr: float = 0.001,
        weight_decay: float = 0.0005,
        num_classes: int = 11,
        class_names: list = None,
    ):
        """
        Initialize the WasteClassifier model.

        Args:
            lr: Learning rate for the optimizer
            weight_decay: Weight decay for the optimizer
            num_classes: Number of output classes
            class_names: List of class names
        """
        super().__init__()

        # Save hyperparameters
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay

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
                "other",
            ]
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

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
        acc = (preds == y).float().mean()
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)

        # Return predictions and targets for later use in confusion matrix
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
