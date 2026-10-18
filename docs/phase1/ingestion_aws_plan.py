"""
Data ingestion and preparation for the Cloud-Native MLOps Pipeline.

DVC Quick-Start (run once after cloning):
    dvc init
    dvc remote add -d s3remote s3://<your-bucket>/dvc-store
    dvc add data/raw/dataset.csv
    git add data/raw/dataset.csv.dvc .dvcignore .dvc/config
    git commit -m "track dataset with DVC"
    dvc push

In CI/CD, DVC pulls the exact versioned dataset before training:
    dvc pull data/raw/dataset.csv
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("mlops.ingestion")


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest or generate the raw training dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--bucket-name",
        type=str,
        default="",
        help="S3 bucket name (used for future upload steps; not required for local mock).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Local directory where dataset.csv will be saved.",
    )
    parser.add_argument(
        "--mock-size",
        type=int,
        default=1000,
        help="Number of rows to generate when producing a mock dataset.",
    )
    return parser.parse_args()


def generate_mock_dataset(n_samples: int, random_state: int = 42) -> pd.DataFrame:
    """
    Generate a reproducible binary-classification dataset.

    Features are continuous; target is 0 or 1.
    """
    logger.info("Generating mock dataset with %d samples.", n_samples)

    X, y = make_classification(
        n_samples=n_samples,
        n_features=8,
        n_informative=5,
        n_redundant=2,
        n_clusters_per_class=1,
        flip_y=0.03,
        random_state=random_state,
    )

    feature_cols = {f"feature_{i + 1}": X[:, i] for i in range(X.shape[1])}
    df = pd.DataFrame(feature_cols)
    df["target"] = y.astype(int)

    logger.info(
        "Dataset shape: %s | target distribution: %s",
        df.shape,
        df["target"].value_counts().to_dict(),
    )
    return df


def validate_dataset(df: pd.DataFrame) -> None:
    """Raise if the dataset fails basic sanity checks."""
    if df.empty:
        raise ValueError("Dataset is empty.")
    if "target" not in df.columns:
        raise ValueError("Dataset is missing required 'target' column.")
    if df.isnull().any().any():
        null_counts = df.isnull().sum()
        raise ValueError(f"Dataset contains null values:\n{null_counts[null_counts > 0]}")
    logger.info("Dataset validation passed.")


def save_dataset(df: pd.DataFrame, output_dir: str, filename: str = "dataset.csv") -> Path:
    """Persist the DataFrame to *output_dir/filename*, creating dirs as needed."""
    out_path = Path(output_dir) / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("Dataset saved to %s (%d bytes).", out_path, out_path.stat().st_size)
    return out_path


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    logger.info("Starting data ingestion.")
    logger.info("Output directory : %s", args.output_dir)
    logger.info("S3 bucket        : %s", args.bucket_name or "(not specified)")
    logger.info("Mock size        : %d rows", args.mock_size)

    raw_path = Path(args.output_dir) / "dataset.csv"

    if raw_path.exists():
        logger.info("Existing dataset found at %s — loading instead of regenerating.", raw_path)
        df = pd.read_csv(raw_path)
    else:
        df = generate_mock_dataset(n_samples=args.mock_size)

    validate_dataset(df)
    saved_path = save_dataset(df, output_dir=args.output_dir)

    logger.info("Ingestion complete. Dataset ready at %s.", saved_path)


if __name__ == "__main__":
    main()
