"""Model definition for mushroom classification."""

from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchmetrics.classification import (
    Accuracy,
    F1Score,
    Precision,
    Recall,
    ConfusionMatrix,
)
from torchvision import models


class MushroomClassifier(LightningModule):
    """Lightning module for mushroom classification."""

    def __init__(
        self,
        model_name: str = "resnet50",
        num_classes: int = 10,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        optimizer: str = "adam",
        scheduler: Optional[Dict[str, Any]] = None,
        pretrained: bool = True,
        dropout_rate: float = 0.2,
        feature_extract: bool = False,  # Only update the final layer params
    ):
        """
        Initialize the model.

        Args:
            model_name: Name of the backbone model
            num_classes: Number of output classes
            learning_rate: Learning rate for optimization
            weight_decay: Weight decay for regularization
            optimizer: Name of the optimizer to use
            scheduler: Scheduler configuration
            pretrained: Whether to use pretrained weights
            dropout_rate: Dropout rate for regularization
            feature_extract: Whether to only train the final layer
        """
        super().__init__()
        self.save_hyperparameters()

        # Initialize model
        self.model = self._initialize_model(model_name, num_classes, pretrained, feature_extract)

        # Add dropout for regularization
        if hasattr(self.model, "fc"):
            in_features = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(in_features, num_classes),
            )
        elif hasattr(self.model, "classifier"):
            if isinstance(self.model.classifier, nn.Linear):
                in_features = self.model.classifier.in_features
                self.model.classifier = nn.Sequential(
                    nn.Dropout(dropout_rate),
                    nn.Linear(in_features, num_classes),
                )
            else:
                # For MobileNet and similar architectures
                self.model.classifier[-1] = nn.Linear(
                    self.model.classifier[-1].in_features, num_classes
                )

        # Define metrics
        metrics = self._define_metrics(num_classes)
        self.train_metrics = metrics.copy()
        self.val_metrics = metrics.copy()
        self.test_metrics = metrics.copy()

    def _initialize_model(self, model_name, num_classes, pretrained, feature_extract):
        """Initialize the backbone model."""
        # Based on the selected architecture
        if model_name == "resnet18":
            model = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        elif model_name == "resnet34":
            model = models.resnet34(weights="IMAGENET1K_V1" if pretrained else None)
        elif model_name == "resnet50":
            model = models.resnet50(weights="IMAGENET1K_V1" if pretrained else None)
        elif model_name == "efficientnet_b0":
            model = models.efficientnet_b0(weights="IMAGENET1K_V1" if pretrained else None)
        elif model_name == "mobilenet_v2":
            model = models.mobilenet_v2(weights="IMAGENET1K_V1" if pretrained else None)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        # Freeze parameters for feature extraction only
        if feature_extract:
            for param in model.parameters():
                param.requires_grad = False

        return model

    def _define_metrics(self, num_classes):
        """Define metrics for evaluation."""
        return {
            "accuracy": Accuracy(task="multiclass", num_classes=num_classes),
            "f1": F1Score(task="multiclass", num_classes=num_classes, average="macro"),
            "precision": Precision(task="multiclass", num_classes=num_classes, average="macro"),
            "recall": Recall(task="multiclass", num_classes=num_classes, average="macro"),
        }

    def forward(self, x):
        """Forward pass through the model."""
        return self.model(x)

    def _calculate_loss(self, logits, targets):
        """Calculate the loss."""
        return F.cross_entropy(logits, targets)

    def training_step(self, batch, batch_idx):
        """Training step."""
        x, y = batch
        logits = self(x)
        loss = self._calculate_loss(logits, y)

        # Calculate and log metrics
        preds = torch.argmax(logits, dim=1)
        for name, metric in self.train_metrics.items():
            value = metric(preds, y)
            self.log(f"train_{name}", value, prog_bar=True)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        x, y = batch
        logits = self(x)
        loss = self._calculate_loss(logits, y)

        # Calculate and log metrics
        preds = torch.argmax(logits, dim=1)
        for name, metric in self.val_metrics.items():
            value = metric(preds, y)
            self.log(f"val_{name}", value, prog_bar=True)

        self.log("val_loss", loss, prog_bar=True)
        return {"val_loss": loss, "preds": preds, "targets": y}

    def test_step(self, batch, batch_idx):
        """Test step."""
        x, y = batch
        logits = self(x)
        loss = self._calculate_loss(logits, y)

        # Calculate metrics
        preds = torch.argmax(logits, dim=1)
        for name, metric in self.test_metrics.items():
            value = metric(preds, y)
            self.log(f"test_{name}", value)

        self.log("test_loss", loss)
        return {"test_loss": loss, "preds": preds, "targets": y}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """Prediction step."""
        if isinstance(batch, tuple):
            x = batch[0]
        else:
            x = batch

        logits = self(x)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        return {"probabilities": probs, "predictions": preds}

    def configure_optimizers(self):
        """Configure optimizers and schedulers."""
        # Get trainable parameters
        parameters = filter(lambda p: p.requires_grad, self.parameters())

        # Choose optimizer
        if self.hparams.optimizer == "adam":
            optimizer = torch.optim.Adam(
                parameters,
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
            )
        elif self.hparams.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                parameters,
                lr=self.hparams.learning_rate,
                momentum=0.9,
                weight_decay=self.hparams.weight_decay,
            )
        elif self.hparams.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                parameters,
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
            )
        else:
            raise ValueError(
                f"Unsupported optimizer: {self.hparams.optimizer}. "
                "Supported optimizers: adam, sgd, adamw"
            )

        # Configure scheduler
        if self.hparams.scheduler is not None:
            scheduler_config = self.hparams.scheduler
            scheduler_name = scheduler_config.get("name", "cosine")

            if scheduler_name == "cosine":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=self.trainer.max_epochs, eta_min=1e-6
                )
            elif scheduler_name == "step":
                step_size = scheduler_config.get("step_size", 10)
                gamma = scheduler_config.get("gamma", 0.1)
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=step_size, gamma=gamma
                )
            elif scheduler_name == "reduce_on_plateau":
                patience = scheduler_config.get("patience", 5)
                factor = scheduler_config.get("gamma", 0.1)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", factor=factor, patience=patience, verbose=True
                )
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "monitor": "val_loss",
                    },
                }
            else:
                raise ValueError(
                    f"Unsupported scheduler: {scheduler_name}. "
                    "Supported schedulers: cosine, step, reduce_on_plateau"
                )

            return [optimizer], [scheduler]

        return optimizer
