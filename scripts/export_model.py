#!/usr/bin/env python
"""Script to export trained PyTorch models to ONNX format."""

import argparse
import os
from pathlib import Path

import torch

from mushroom_classifier.model import MushroomClassifier
from mushroom_classifier.utils import export_to_onnx


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Export model to ONNX format")
    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to PyTorch model checkpoint"
    )
    parser.add_argument(
        "--output-path", type=str, required=True, help="Path to save the ONNX model"
    )
    parser.add_argument(
        "--input-shape",
        type=int,
        nargs="+",
        default=[1, 3, 224, 224],
        help="Input shape for the model (batch, channels, height, width)",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device to use for export (cpu or cuda)"
    )
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    model_path = Path(args.model_path)
    output_path = Path(args.output_path)

    # Check if model exists
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    # Ensure output directory exists
    os.makedirs(output_path.parent, exist_ok=True)

    # Load model
    print(f"Loading model from {model_path}...")
    model = MushroomClassifier.load_from_checkpoint(str(model_path), map_location=args.device)
    model.eval()
    model.to(args.device)

    # Export to ONNX
    print(f"Exporting model to ONNX format at {output_path}...")
    export_to_onnx(
        model=model,
        save_path=output_path,
        input_shape=tuple(args.input_shape),
        device=args.device,
    )

    print("Model export completed successfully!")


if __name__ == "__main__":
    main()
