"""Download datasets from Kaggle."""

import os
import sys
import zipfile
import argparse

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# Add project root to path
sys.path.insert(0, os.path.dirname(DATA_DIR))
from src.config import DATASETS, DATASET_NAME


def download_dataset(dataset_name: str = None):
    """Download dataset from Kaggle (requires kaggle.json configured)."""
    dataset_name = dataset_name or DATASET_NAME
    ds = DATASETS[dataset_name]

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("Install kaggle: pip install kaggle")
        print("Then place kaggle.json in ~/.kaggle/")
        return

    api = KaggleApi()
    api.authenticate()

    slug = ds["kaggle_slug"]
    print(f"Downloading {ds['description']}...")
    print(f"Kaggle dataset: {slug}")
    api.dataset_download_files(slug, path=DATA_DIR, unzip=False)

    # Find and extract zip
    for f in os.listdir(DATA_DIR):
        if f.endswith(".zip"):
            zip_path = os.path.join(DATA_DIR, f)
            print(f"Extracting {f}...")
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(DATA_DIR)
            os.remove(zip_path)

    print(f"Dataset extracted to {DATA_DIR}/")
    print(f"\nRun the pipeline with:")
    print(f"  python main.py --dataset {dataset_name} --data-dir data/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download NIDS dataset from Kaggle")
    parser.add_argument("--dataset", type=str, default=None,
                        choices=list(DATASETS.keys()),
                        help=f"Dataset to download (default: {DATASET_NAME})")
    args = parser.parse_args()
    download_dataset(args.dataset)
