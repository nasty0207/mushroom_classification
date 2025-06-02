"""Training script for mushroom classification."""

import os
from pathlib import Path
from typing import Dict, Optional

import hydra
import mlflow
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import MLFlowLogger, TensorBoardLogger

from mushroom_classifier.data import MushroomDataModule
from mushroom_classifier.model import MushroomClassifier


def train(config: DictConfig) -> Dict:
    """Train the model based on the given configuration.

    Args:
        config: Hydra configuration

    Returns:
        Dictionary containing training metrics
    """
    # Create directories
    os.makedirs(config.directories.checkpoints, exist_ok=True)
    os.makedirs(config.directories.logs, exist_ok=True)

    # Create data module
    data_module = MushroomDataModule(
        data_dir=config.directories.data,
        batch_size=config.training.dataloader.batch_size,
        num_workers=config.training.dataloader.num_workers,
        pin_memory=config.training.dataloader.pin_memory,
    )

    # Get the number of classes
    data_module.setup()
    num_classes = data_module.num_classes

    # Create model
    model = MushroomClassifier(
        model_name=config.model.name,
        num_classes=num_classes,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        optimizer=config.training.optimizer,
        scheduler={
            "name": config.training.scheduler.name,
            "step_size": config.training.scheduler.step_size,
            "gamma": config.training.scheduler.gamma,
            "patience": config.training.scheduler.patience,
        },
        pretrained=config.model.pretrained,
        dropout_rate=config.model.dropout_rate,
        feature_extract=config.model.feature_extract,
    )

    # Load checkpoint if specified
    if config.model.checkpoint_path is not None:
        model = MushroomClassifier.load_from_checkpoint(config.model.checkpoint_path)

    # Configure callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.directories.checkpoints,
        filename="mushroom-{epoch:02d}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=config.training.early_stopping_patience,
        mode="min",
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # Configure loggers
    tensorboard_logger = TensorBoardLogger(
        save_dir=config.directories.logs,
        name="tensorboard",
        version=config.experiment.name,
    )

    mlflow_logger = MLFlowLogger(
        experiment_name=config.experiment.name,
        tracking_uri=config.experiment.mlflow.tracking_uri,
        tags=config.experiment.tags,
        save_dir=config.experiment.mlflow.tracking_uri,
    )

    # Set random seed for reproducibility
    pl.seed_everything(config.seed, workers=True)

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        accelerator=config.devices.accelerator,
        devices=config.devices.devices,
        precision=config.devices.precision,
        callbacks=[checkpoint_callback, early_stopping, lr_monitor],
        logger=[tensorboard_logger, mlflow_logger],
        deterministic=True,
    )

    # Train model
    trainer.fit(model=model, datamodule=data_module)

    # Test model
    test_results = trainer.test(model=model, datamodule=data_module)[0]

    # Save path to best model
    best_model_path = checkpoint_callback.best_model_path
    print(f"Best model saved at: {best_model_path}")

    # Log model to MLflow
    if config.experiment.log_model:
        mlflow_logger.experiment.log_artifact(
            mlflow_logger.run_id, best_model_path, "model"
        )

    return {"best_model_path": best_model_path, "test_metrics": test_results}


@hydra.main(config_path="../configs", config_name="training")
def main(config: DictConfig) -> None:
    """Main function for training."""
    print(f"Configuration:\n{config}")
    train_results = train(config)
    print(f"Training completed. Test metrics: {train_results['test_metrics']}")


if __name__ == "__main__":
    main()
