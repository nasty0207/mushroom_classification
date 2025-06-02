"""Utility functions for mushroom classification."""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report


def generate_class_mapping(data_dir: Union[str, Path]) -> Dict[str, int]:
    """
    Generate class mapping from directory structure.

    Args:
        data_dir: Directory containing class subdirectories

    Returns:
        Dictionary mapping class names to indices
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Find all class directories
    class_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    class_dirs.sort()  # Sort for reproducibility

    # Create mapping from class name to index
    class_mapping = {d.name: i for i, d in enumerate(class_dirs)}
    return class_mapping


def save_class_mapping(class_mapping: Dict[str, int], output_path: Union[str, Path]) -> None:
    """
    Save class mapping to JSON file.

    Args:
        class_mapping: Dictionary mapping class names to indices
        output_path: Path to save the mapping
    """
    output_path = Path(output_path)
    os.makedirs(output_path.parent, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(class_mapping, f, indent=2)

    print(f"Class mapping saved to {output_path}")


def load_class_mapping(mapping_path: Union[str, Path]) -> Dict[str, int]:
    """
    Load class mapping from JSON file.

    Args:
        mapping_path: Path to the class mapping JSON file

    Returns:
        Dictionary mapping class names to indices
    """
    mapping_path = Path(mapping_path)
    if not mapping_path.exists():
        raise FileNotFoundError(f"Class mapping file not found: {mapping_path}")

    with open(mapping_path, "r") as f:
        class_mapping = json.load(f)

    # Convert keys to strings and values to integers if needed
    class_mapping = {str(k): int(v) for k, v in class_mapping.items()}
    return class_mapping


def visualize_predictions(
    images: List[Union[str, Image.Image]],
    predictions: List[Dict],
    class_mapping: Optional[Dict[str, int]] = None,
    save_path: Optional[Union[str, Path]] = None,
):
    """
    Visualize model predictions.

    Args:
        images: List of image paths or PIL Images
        predictions: List of prediction dictionaries
        class_mapping: Dictionary mapping class names to indices
        save_path: Path to save the visualization
    """
    num_images = len(images)
    cols = min(5, num_images)
    rows = (num_images + cols - 1) // cols

    plt.figure(figsize=(cols * 4, rows * 4))

    for i, (img, pred) in enumerate(zip(images, predictions)):
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")

        # Plot image
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.axis("off")

        # Add prediction info as title
        title = f"{pred['class_name']}\nConf: {pred['confidence']:.2f}"
        plt.title(title)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        os.makedirs(save_path.parent, exist_ok=True)
        plt.savefig(save_path)
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()


def export_to_onnx(model, save_path, input_shape=(1, 3, 224, 224), device="cpu"):
    """
    Export PyTorch model to ONNX format.

    Args:
        model: PyTorch model to export
        save_path: Path to save the ONNX model
        input_shape: Shape of the input tensor
        device: Device to run the export on
    """
    # Ensure directory exists
    save_path = Path(save_path)
    os.makedirs(save_path.parent, exist_ok=True)

    # Create dummy input
    dummy_input = torch.randn(input_shape, device=device)

    # Set model to eval mode
    model.eval()
    model.to(device)

    # Export model to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    print(f"Model exported to ONNX format at {save_path}")


def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    class_names: Optional[List[str]] = None,
    save_path: Optional[Union[str, Path]] = None,
):
    """
    Plot confusion matrix for model evaluation.

    Args:
        y_true: List of true class indices
        y_pred: List of predicted class indices
        class_names: List of class names
        save_path: Path to save the plot
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # If class names are not provided, use indices
    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]

    # Plot
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()

    # Add axis labels
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                horizontalalignment="center",
                verticalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    if save_path:
        save_path = Path(save_path)
        os.makedirs(save_path.parent, exist_ok=True)
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()

    # Print classification report
    report = classification_report(
        y_true, y_pred, target_names=class_names, digits=3
    )
    print("Classification Report:\n")
    print(report)
