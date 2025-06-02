"""Data loading and processing module for mushroom classification."""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
from PIL import Image
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class MushroomDataset(Dataset):
    """Dataset for mushroom images."""

    def __init__(
        self,
        image_paths: List[str],
        labels: Optional[List[int]] = None,
        transform=None,
        class_mapping: Optional[Dict[str, int]] = None,
    ):
        """
        Initialize the dataset.

        Args:
            image_paths: List of paths to images
            labels: List of labels corresponding to images
            transform: Transforms to apply to images
            class_mapping: Mapping from class names to indices
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.class_mapping = class_mapping

    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, int], torch.Tensor]:
        """Return an example from the dataset."""
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        if self.labels is not None:
            return img, self.labels[idx]
        return img


class MushroomDataModule(LightningDataModule):
    """Lightning data module for mushroom classification."""

    def __init__(
        self,
        data_dir: str = "data/processed",
        batch_size: int = 32,
        num_workers: int = 4,
        image_size: Tuple[int, int] = (224, 224),
        augmentation: bool = True,
        pin_memory: bool = True,
    ):
        """
        Initialize the data module.

        Args:
            data_dir: Directory containing the data
            batch_size: Batch size for the dataloaders
            num_workers: Number of workers for the dataloaders
            image_size: Image dimensions (height, width)
            augmentation: Whether to use data augmentation
            pin_memory: Whether to pin memory for the dataloaders
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.augmentation = augmentation
        self.pin_memory = pin_memory

        # Load class mapping
        class_mapping_path = self.data_dir.parent / "class_mapping.json"
        if class_mapping_path.exists():
            with open(class_mapping_path, "r") as f:
                self.class_mapping = json.load(f)
            self.num_classes = len(self.class_mapping)
        else:
            self.class_mapping = None
            self.num_classes = None

        self.train_transform = None
        self.val_transform = None

    def setup(self, stage: Optional[str] = None):
        """Set up the datasets for the data module."""
        # Define transformations
        self.train_transform = transforms.Compose(
            [
                transforms.Resize(self.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.val_transform = transforms.Compose(
            [
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # Prepare datasets
        if stage == "fit" or stage is None:
            self._setup_train_val()

        if stage == "test" or stage is None:
            self._setup_test()

        if stage == "predict" or stage is None:
            self._setup_predict()

    def _setup_train_val(self):
        """Set up training and validation datasets."""
        train_dir = self.data_dir / "train"
        val_dir = self.data_dir / "val"

        # Get class directories
        train_class_dirs = [d for d in train_dir.iterdir() if d.is_dir()]
        val_class_dirs = [d for d in val_dir.iterdir() if d.is_dir()]

        # Collect image paths and labels
        train_image_paths, train_labels = self._collect_images_and_labels(train_class_dirs)
        val_image_paths, val_labels = self._collect_images_and_labels(val_class_dirs)

        # Create datasets
        self.train_dataset = MushroomDataset(
            train_image_paths,
            train_labels,
            transform=self.train_transform,
            class_mapping=self.class_mapping,
        )
        self.val_dataset = MushroomDataset(
            val_image_paths,
            val_labels,
            transform=self.val_transform,
            class_mapping=self.class_mapping,
        )

    def _setup_test(self):
        """Set up test dataset."""
        test_dir = self.data_dir / "test"
        test_class_dirs = [d for d in test_dir.iterdir() if d.is_dir()]

        # Collect image paths and labels
        test_image_paths, test_labels = self._collect_images_and_labels(test_class_dirs)

        # Create dataset
        self.test_dataset = MushroomDataset(
            test_image_paths,
            test_labels,
            transform=self.val_transform,
            class_mapping=self.class_mapping,
        )

    def _setup_predict(self):
        """Set up prediction dataset."""
        # This could be the same as test dataset or another custom dataset
        # For now, just use the test dataset
        if not hasattr(self, "test_dataset"):
            self._setup_test()
        self.predict_dataset = self.test_dataset

    def _collect_images_and_labels(self, class_dirs):
        """Collect image paths and labels from class directories."""
        image_paths = []
        labels = []

        for class_dir in class_dirs:
            class_name = class_dir.name
            class_idx = self.class_mapping.get(class_name) if self.class_mapping else None

            # Collect images
            for img_path in class_dir.glob("*.jpg"):
                image_paths.append(str(img_path))
                if class_idx is not None:
                    labels.append(class_idx)

            # Also check for PNG images
            for img_path in class_dir.glob("*.png"):
                image_paths.append(str(img_path))
                if class_idx is not None:
                    labels.append(class_idx)

        return image_paths, labels

    def train_dataloader(self):
        """Return the training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        """Return the validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        """Return the test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def predict_dataloader(self):
        """Return the prediction dataloader."""
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )


def prepare_data(config):
    """Prepare data for training."""
    # Implementation for data preparation script
    # This would be used by the prepare_data.py script
    pass
