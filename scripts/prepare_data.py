#!/usr/bin/env python
"""Script to prepare and preprocess data for mushroom classification."""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from mushroom_classifier.utils import generate_class_mapping, save_class_mapping


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Prepare data for mushroom classification.")
    parser.add_argument(
        "--raw-data", type=str, required=True, help="Path to raw data directory"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to output processed data"
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.7, help="Ratio of training data"
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.15, help="Ratio of validation data"
    )
    parser.add_argument(
        "--test-ratio", type=float, default=0.15, help="Ratio of test data"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--resize", type=int, nargs=2, default=[224, 224], help="Size to resize images to (H W)"
    )
    return parser.parse_args()


def create_directory_structure(output_dir: Path) -> Tuple[Path, Path, Path]:
    """Create directory structure for processed data."""
    # Create main output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create subdirectories for train, val, test
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    test_dir = output_dir / "test"

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    return train_dir, val_dir, test_dir


def collect_image_paths(data_dir: Path) -> Dict[str, List[Path]]:
    """Collect image paths organized by class."""
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Find all class directories
    class_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    print(f"Found {len(class_dirs)} classes")

    # Collect image paths for each class
    image_paths = {}
    for class_dir in class_dirs:
        class_name = class_dir.name
        # Collect both .jpg and .png images
        jpg_paths = list(class_dir.glob("*.jpg"))
        png_paths = list(class_dir.glob("*.png"))
        all_paths = jpg_paths + png_paths
        if all_paths:
            image_paths[class_name] = all_paths
            print(f"Class '{class_name}': {len(all_paths)} images")
        else:
            print(f"Warning: No images found for class '{class_name}'")

    return image_paths


def split_and_copy_data(
    image_paths: Dict[str, List[Path]],
    train_dir: Path,
    val_dir: Path,
    test_dir: Path,
    train_ratio: float,
    val_ratio: float,
    resize: Optional[Tuple[int, int]] = None,
    seed: int = 42,
):
    """Split data into train/val/test sets and copy to respective directories."""
    np.random.seed(seed)

    for class_name, paths in image_paths.items():
        print(f"\nProcessing class: {class_name}")

        # Create class directories in train, val, test
        train_class_dir = train_dir / class_name
        val_class_dir = val_dir / class_name
        test_class_dir = test_dir / class_name

        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)

        # Split data
        train_paths, temp_paths = train_test_split(
            paths, train_size=train_ratio, random_state=seed
        )
        # Split remaining data into val and test
        remaining_ratio = val_ratio / (val_ratio + test_ratio)
        val_paths, test_paths = train_test_split(
            temp_paths, train_size=remaining_ratio, random_state=seed
        )

        print(f"Training: {len(train_paths)} images")
        print(f"Validation: {len(val_paths)} images")
        print(f"Test: {len(test_paths)} images")

        # Process and copy images
        process_and_copy_images(train_paths, train_class_dir, resize)
        process_and_copy_images(val_paths, val_class_dir, resize)
        process_and_copy_images(test_paths, test_class_dir, resize)


def process_and_copy_images(paths: List[Path], output_dir: Path, resize: Optional[Tuple[int, int]]):
    """Process and copy images to the target directory."""
    for idx, path in enumerate(tqdm(paths, desc="Processing")):
        try:
            # Load image
            img = Image.open(path).convert("RGB")

            # Resize if specified
            if resize is not None:
                img = img.resize(resize)

            # Save to output directory
            output_path = output_dir / f"{idx:05d}{path.suffix}"
            img.save(output_path)

        except Exception as e:
            print(f"Error processing {path}: {e}")


def main():
    """Main function."""
    args = parse_args()

    raw_data_dir = Path(args.raw_data)
    output_dir = Path(args.output)

    # Create directory structure
    train_dir, val_dir, test_dir = create_directory_structure(output_dir)
    print(f"Created output directories in {output_dir}")

    # Collect image paths
    image_paths = collect_image_paths(raw_data_dir)

    # Generate class mapping
    class_mapping = generate_class_mapping(raw_data_dir)
    mapping_path = output_dir / "class_mapping.json"
    save_class_mapping(class_mapping, mapping_path)

    # Split and copy data
    resize = tuple(args.resize) if args.resize else None
    split_and_copy_data(
        image_paths,
        train_dir,
        val_dir,
        test_dir,
        args.train_ratio,
        args.val_ratio,
        resize=resize,
        seed=args.seed,
    )

    print("\nData preparation completed!")
    print(f"Processed data saved to {output_dir}")


if __name__ == "__main__":
    main()
