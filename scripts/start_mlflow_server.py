#!/usr/bin/env python
"""Script to start MLflow tracking server."""

import argparse
import os
import subprocess
from pathlib import Path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Start MLflow tracking server")
    parser.add_argument(
        "--backend-store-uri",
        type=str,
        default="./logs/mlflow",
        help="URI for backing MLflow tracking server",
    )
    parser.add_argument(
        "--default-artifact-root",
        type=str,
        default="./logs/mlflow/artifacts",
        help="Directory for storing artifacts",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind the server to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to run the server on",
    )
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    # Create directories if they don't exist
    backend_store_path = Path(args.backend_store_uri)
    artifact_root_path = Path(args.default_artifact_root)

    os.makedirs(backend_store_path, exist_ok=True)
    os.makedirs(artifact_root_path, exist_ok=True)

    # Build MLflow command
    cmd = [
        "mlflow",
        "server",
        "--backend-store-uri",
        args.backend_store_uri,
        "--default-artifact-root",
        args.default_artifact_root,
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]

    print(f"Starting MLflow server at http://{args.host}:{args.port}")
    print(f"Backend store: {args.backend_store_uri}")
    print(f"Artifact store: {args.default_artifact_root}")

    # Run MLflow server
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nShutting down MLflow server...")
    except Exception as e:
        print(f"Error starting MLflow server: {e}")


if __name__ == "__main__":
    main()
