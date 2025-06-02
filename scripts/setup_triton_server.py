#!/usr/bin/env python
"""Script to setup Triton Inference Server with mushroom classifier model."""

import argparse
import json
import os
import shutil
from pathlib import Path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Setup Triton Inference Server with model")
    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to the model file"
    )
    parser.add_argument(
        "--model-format",
        type=str,
        default="onnx",
        choices=["onnx", "tensorrt", "pytorch"],
        help="Format of the model",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="mushroom_classifier",
        help="Name for the model in Triton",
    )
    parser.add_argument(
        "--model-version",
        type=int,
        default=1,
        help="Version number for the model",
    )
    parser.add_argument(
        "--repository-path",
        type=str,
        default="./triton_models",
        help="Path to Triton model repository",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=32,
        help="Maximum batch size for the model",
    )
    parser.add_argument(
        "--start-server",
        action="store_true",
        help="Start Triton server after setup",
    )
    parser.add_argument(
        "--http-port",
        type=int,
        default=8000,
        help="HTTP port for Triton server",
    )
    parser.add_argument(
        "--grpc-port",
        type=int,
        default=8001,
        help="gRPC port for Triton server",
    )
    return parser.parse_args()


def create_model_repository(args):
    """Create model repository structure for Triton."""
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")

    # Create model repository directory structure
    repository_path = Path(args.repository_path)
    model_repo_dir = repository_path / args.model_name
    model_version_dir = model_repo_dir / str(args.model_version)

    # Create directories
    os.makedirs(model_version_dir, exist_ok=True)

    # Copy the model file to the version directory
    if args.model_format == "onnx":
        target_path = model_version_dir / "model.onnx"
        shutil.copy(model_path, target_path)
        backend = "onnxruntime"
    elif args.model_format == "tensorrt":
        target_path = model_version_dir / "model.plan"
        shutil.copy(model_path, target_path)
        backend = "tensorrt"
    elif args.model_format == "pytorch":
        # PyTorch models require a model directory containing model.pt and special handling
        os.makedirs(model_version_dir / "model", exist_ok=True)
        target_path = model_version_dir / "model" / "model.pt"
        shutil.copy(model_path, target_path)
        backend = "pytorch"

    print(f"Copied model to {target_path}")

    # Create config.pbtxt file
    create_model_config(
        model_repo_dir, args.model_name, backend, args.max_batch_size, args.model_format
    )

    return repository_path


def create_model_config(model_dir, model_name, backend, max_batch_size, model_format):
    """Create config.pbtxt file for Triton model."""
    config_file = model_dir / "config.pbtxt"

    # Basic config template
    config = f"""name: "{model_name}"
    platform: "{backend}"
    max_batch_size: {max_batch_size}
    
    input [
      {{
        name: "input"
        data_type: TYPE_FP32
        format: FORMAT_NCHW
        dims: [ 3, 224, 224 ]
      }}
    ]
    
    output [
      {{
        name: "output"
        data_type: TYPE_FP32
        dims: [ -1 ]  # Number of classes, will be adjusted dynamically
      }}
    ]
    
    instance_group [
      {{
        count: 1
        kind: KIND_GPU
      }}
    ]
    """

    # Add backend-specific settings
    if backend == "onnxruntime":
        config += """optimization { execution_accelerators {
            gpu_execution_accelerator : [ { name : "tensorrt" } ]
        }}
        """
    
    elif backend == "pytorch":
        if model_format == "pytorch":
            config += """parameters { key: "ENABLE_PYTORCH_PROFILER" value: { string_value: "true" } }
            """

    # Write config to file
    with open(config_file, "w") as f:
        f.write(config)

    print(f"Created model configuration at {config_file}")


def start_triton_server(repository_path, http_port=8000, grpc_port=8001):
    """Start Triton Inference Server with the configured model repository."""
    try:
        import subprocess

        cmd = [
            "docker",
            "run",
            "--gpus=all",
            "-it",
            "--rm",
            "-p", f"{http_port}:{http_port}",
            "-p", f"{grpc_port}:{grpc_port}",
            "-v", f"{repository_path}:/models",
            "nvcr.io/nvidia/tritonserver:23.04-py3",
            "tritonserver",
            "--model-repository=/models",
            "--http-port", str(http_port),
            "--grpc-port", str(grpc_port),
            "--log-verbose=1"
        ]

        print("Starting Triton Inference Server...")
        print(f"Command: {' '.join(cmd)}")
        subprocess.run(cmd)

    except Exception as e:
        print(f"Error starting Triton server: {e}")
        print("Please ensure Docker is installed and running, and that you have nvidia-docker setup if using GPU.")
        print("You can start the server manually with:")
        print(f"docker run --gpus=all -it --rm -p {http_port}:{http_port} -p {grpc_port}:{grpc_port} \
              -v {repository_path}:/models nvcr.io/nvidia/tritonserver:23.04-py3 \
              tritonserver --model-repository=/models")


def main():
    """Main function."""
    args = parse_args()

    try:
        # Create model repository
        repository_path = create_model_repository(args)
        print(f"Triton model repository created at {repository_path}")

        # Start server if requested
        if args.start_server:
            start_triton_server(repository_path, args.http_port, args.grpc_port)
        else:
            print("\nModel repository setup complete. To start Triton Inference Server:")
            print(f"docker run --gpus=all -it --rm -p {args.http_port}:{args.http_port} -p {args.grpc_port}:{args.grpc_port} \
                  -v {repository_path.absolute()}:/models nvcr.io/nvidia/tritonserver:23.04-py3 \
                  tritonserver --model-repository=/models")

    except Exception as e:
        print(f"Error setting up Triton server: {e}")


if __name__ == "__main__":
    main()
