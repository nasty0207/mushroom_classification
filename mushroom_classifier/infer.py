"""Inference code for mushroom classification."""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from torchvision import transforms

from mushroom_classifier.model import MushroomClassifier


class MushroomInferenceModel:
    """Model for mushroom classification inference."""

    def __init__(
        self,
        model_path: str,
        class_mapping_path: Optional[str] = None,
        device: str = "cpu",
    ):
        """
        Initialize the inference model.

        Args:
            model_path: Path to the model file (pytorch or onnx)
            class_mapping_path: Path to the class mapping file
            device: Device to run inference on
        """
        self.model_path = Path(model_path)
        self.device = device
        self.model = None
        self.class_mapping = None
        self.idx_to_class = None
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # Load class mapping if provided
        if class_mapping_path is not None:
            self._load_class_mapping(class_mapping_path)

        # Load model based on file extension
        self._load_model()

    def _load_class_mapping(self, class_mapping_path: str):
        """Load class mapping from JSON file."""
        path = Path(class_mapping_path)
        if not path.exists():
            print(f"Warning: Class mapping file not found at {path}")
            return

        with open(path, "r") as f:
            self.class_mapping = json.load(f)

        # Create reverse mapping (index to class name)
        self.idx_to_class = {int(v): k for k, v in self.class_mapping.items()}

    def _load_model(self):
        """Load model based on file extension."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found at {self.model_path}")

        if self.model_path.suffix.lower() == ".onnx":
            # Load ONNX model
            providers = [
                "CUDAExecutionProvider" if self.device == "cuda" else "CPUExecutionProvider"
            ]
            self.model = ort.InferenceSession(str(self.model_path), providers=providers)
            self.model_type = "onnx"
        elif self.model_path.suffix.lower() == ".pt" or self.model_path.suffix.lower() == ".pth":
            # Load PyTorch model
            self.model = MushroomClassifier.load_from_checkpoint(
                self.model_path, map_location=self.device
            )
            self.model.eval()
            self.model.to(self.device)
            self.model_type = "pytorch"
        else:
            raise ValueError(
                f"Unsupported model format: {self.model_path.suffix}. "
                "Supported formats: .onnx, .pt, .pth"
            )

    def preprocess(self, image: Union[str, Image.Image]) -> torch.Tensor:
        """Preprocess an image for inference."""
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            raise TypeError("Image must be a PIL Image or a path to an image.")

        return self.transform(image).unsqueeze(0)

    def predict(
        self, image: Union[str, Image.Image], return_probabilities: bool = False
    ) -> Dict:
        """Run inference on an image."""
        # Preprocess image
        img_tensor = self.preprocess(image)

        if self.model_type == "pytorch":
            with torch.no_grad():
                outputs = self.model(img_tensor.to(self.device))
                probabilities = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
        elif self.model_type == "onnx":
            # Run ONNX inference
            ort_inputs = {
                self.model.get_inputs()[0].name: img_tensor.cpu().numpy(),
            }
            outputs = self.model.run(None, ort_inputs)
            probabilities = outputs[0]

        # Get the predicted class
        class_idx = np.argmax(probabilities, axis=1)[0]
        class_prob = float(probabilities[0, class_idx])

        # Get class name if available
        if self.idx_to_class is not None:
            class_name = self.idx_to_class.get(class_idx, f"Class {class_idx}")
        else:
            class_name = f"Class {class_idx}"

        result = {
            "class_index": int(class_idx),
            "class_name": class_name,
            "confidence": class_prob,
        }

        if return_probabilities:
            result["probabilities"] = {
                self.idx_to_class.get(i, f"Class {i}"): float(p)
                for i, p in enumerate(probabilities[0])
            }

        return result

    def predict_batch(
        self, images: List[Union[str, Image.Image]], return_probabilities: bool = False
    ) -> List[Dict]:
        """Run inference on a batch of images."""
        return [self.predict(img, return_probabilities) for img in images]


def main():
    """Main function for inference."""
    parser = argparse.ArgumentParser(description="Run inference on mushroom images")
    parser.add_argument("--image-path", type=str, required=True, help="Path to input image")
    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to model file (.pt or .onnx)"
    )
    parser.add_argument(
        "--class-mapping",
        type=str,
        default=None,
        help="Path to class mapping JSON file",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device to run inference on")
    args = parser.parse_args()

    # Create inference model
    model = MushroomInferenceModel(
        model_path=args.model_path,
        class_mapping_path=args.class_mapping,
        device=args.device,
    )

    # Run inference
    result = model.predict(args.image_path, return_probabilities=True)

    # Print results
    print(f"Predicted class: {result['class_name']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print("\nClass probabilities:")
    if "probabilities" in result:
        for class_name, prob in sorted(
            result["probabilities"].items(), key=lambda x: x[1], reverse=True
        ):
            print(f"{class_name}: {prob:.4f}")


if __name__ == "__main__":
    main()
