#!/usr/bin/env python
"""Script to convert ONNX model to TensorRT format."""

import argparse
import os
from pathlib import Path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Convert ONNX model to TensorRT format")
    parser.add_argument(
        "--onnx-path", type=str, required=True, help="Path to ONNX model file"
    )
    parser.add_argument(
        "--tensorrt-path", type=str, required=True, help="Path to save TensorRT model"
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp16",
        choices=["fp32", "fp16", "int8"],
        help="Precision to use for TensorRT model",
    )
    parser.add_argument(
        "--workspace",
        type=int,
        default=8,
        help="Workspace size in GB for TensorRT builder",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=32,
        help="Maximum batch size for TensorRT engine",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output during conversion",
    )
    return parser.parse_args()


def convert_onnx_to_tensorrt(args):
    """Convert ONNX model to TensorRT format."""
    try:
        # Import TensorRT here to avoid dependency errors for users who don't have it
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit

        # Create TensorRT logger
        TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE if args.verbose else trt.Logger.WARNING)

        # Create builder and network
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, TRT_LOGGER)

        # Parse ONNX model
        with open(args.onnx_path, "rb") as model:
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                raise ValueError("Failed to parse ONNX model")

        # Create config
        config = builder.create_builder_config()
        config.max_workspace_size = args.workspace * 1 << 30  # Convert GB to bytes

        # Set precision
        if args.precision == "fp16" and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("Building with FP16 precision")
        elif args.precision == "int8" and builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            print("Building with INT8 precision")
        else:
            print("Building with FP32 precision")

        # Set optimization profile
        profile = builder.create_optimization_profile()
        input_name = network.get_input(0).name
        shape = network.get_input(0).shape
        min_shape = (1, shape[1], shape[2], shape[3])
        opt_shape = (args.max_batch_size // 2, shape[1], shape[2], shape[3])
        max_shape = (args.max_batch_size, shape[1], shape[2], shape[3])
        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)

        # Build engine
        print(f"Building TensorRT engine for {args.onnx_path}...")
        engine = builder.build_engine(network, config)
        print("Build completed!")

        # Save engine
        with open(args.tensorrt_path, "wb") as f:
            f.write(engine.serialize())

        print(f"TensorRT engine saved to {args.tensorrt_path}")

    except ImportError:
        print("Error: TensorRT or CUDA Python bindings not found.")
        print("Please install TensorRT and PyCUDA before running this script.")
        return


def main():
    """Main function."""
    args = parse_args()

    onnx_path = Path(args.onnx_path)
    tensorrt_path = Path(args.tensorrt_path)

    # Check if ONNX model exists
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found at {onnx_path}")

    # Ensure output directory exists
    os.makedirs(tensorrt_path.parent, exist_ok=True)

    # Convert model
    convert_onnx_to_tensorrt(args)


if __name__ == "__main__":
    main()
