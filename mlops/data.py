"""Ingestion, validation and content-addressed versioning of training data.

Source: the Elec2 electricity-market dataset (Harries 1999; Gama), OpenML id 151,
licence "Public". 45,312 half-hour records from the New South Wales market; the
label says whether the price went UP or DOWN relative to the last 24 hours. Prices
follow demand, seasons and market rules that change over time, so a model trained
once goes stale - which is exactly what a retraining pipeline should handle.

The batch "uploads" are a replay of that real history, one week per file.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

logger = logging.getLogger("mlops.data")

SOURCE_MD5 = "8ca97867d960ae029ae3a9ac2c923d34"
COLUMNS = ["date", "day", "period", "nswprice", "nswdemand", "vicprice", "vicdemand", "transfer", "label"]
RAW_FEATURES = ["day", "period", "nswprice", "nswdemand", "vicprice", "vicdemand", "transfer"]
# Lag features use only the previous half-hour, which is known when predicting the next one.
FEATURES = RAW_FEATURES + ["prev_label", "nswprice_delta"]
ROWS_PER_WEEK = 48 * 7


class ValidationError(ValueError):
    pass


def read_arff(path: Path) -> pd.DataFrame:
    digest = hashlib.md5(path.read_bytes()).hexdigest()
    if digest != SOURCE_MD5:
        raise ValidationError(f"unexpected source checksum {digest}")
    lines = path.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip().lower() == "@data") + 1
    rows = [line.split(",") for line in lines[start:] if line.strip()]
    df = pd.DataFrame(rows, columns=COLUMNS)
    for c in COLUMNS[:-1]:
        df[c] = pd.to_numeric(df[c])
    df["day"] = df["day"].astype(int)
    df["label"] = (df["label"] == "UP").astype(int)
    return df


def write_weekly_batches(df: pd.DataFrame, out_dir: Path, first: int, count: int) -> list[Path]:
    """Split the history into week-sized files, as if each week were uploaded separately."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for w in range(first, first + count):
        chunk = df.iloc[w * ROWS_PER_WEEK:(w + 1) * ROWS_PER_WEEK]
        if len(chunk) < ROWS_PER_WEEK:
            break
        p = out_dir / f"week_{w:03d}.csv"
        chunk.to_csv(p, index=False)
        paths.append(p)
    return paths


def validate(df: pd.DataFrame) -> None:
    """Reject a batch before it can influence a model. Every rule names what failed."""
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValidationError(f"missing columns: {missing}")
    if len(df) < 200:
        raise ValidationError(f"batch too small: {len(df)} rows (need >= 200)")
    nulls = df[COLUMNS].isnull().sum()
    if nulls.any():
        raise ValidationError(f"null values: {nulls[nulls > 0].to_dict()}")
    normalized = ["period", "nswprice", "nswdemand", "vicprice", "vicdemand", "transfer"]
    out_of_range = [c for c in normalized if (df[c] < 0).any() or (df[c] > 1).any()]
    if out_of_range:
        raise ValidationError(f"values outside the normalized [0, 1] range in: {out_of_range}")
    if not df["day"].between(1, 7).all():
        raise ValidationError("day must be 1..7")
    if set(df["label"].unique()) - {0, 1}:
        raise ValidationError("label must be 0/1")
    share = df["label"].mean()
    if share < 0.05 or share > 0.95:
        raise ValidationError(f"label share {share:.2f} is implausible for a market week")
    # The published `date` field is not monotonic at 5 points, so order is checked on the
    # half-hour sequence instead: period rises within a day, then resets as the weekday advances.
    dp = df["period"].diff().iloc[1:]
    day_next = (df["day"].shift(1) % 7 + 1).iloc[1:]
    rollover = (df["period"].iloc[1:] == 0) & (df["day"].iloc[1:] == day_next)
    if not ((dp > 0) | rollover).all():
        raise ValidationError("records are not in half-hour order")


@dataclass
class DataVersion:
    sha256: str
    path: Path
    rows: int
    source_name: str


class DataStore:
    """Content-addressed store: identical uploads map to the same version, and every
    model can name the exact bytes it was trained on."""

    def __init__(self, root: Path):
        self.root = root
        self.objects = root / "objects"
        self.ledger = root / "versions.jsonl"
        self.objects.mkdir(parents=True, exist_ok=True)

    def put(self, path: Path) -> tuple[DataVersion, bool]:
        blob = path.read_bytes()
        sha = hashlib.sha256(blob).hexdigest()
        obj = self.objects / f"{sha}.csv"
        is_new = not obj.exists()
        if is_new:
            obj.write_bytes(blob)
            rows = blob.count(b"\n") - 1
            with self.ledger.open("a") as f:
                f.write(json.dumps({"sha256": sha, "source": path.name, "rows": rows}) + "\n")
        return DataVersion(sha, obj, blob.count(b"\n") - 1, path.name), is_new

    def versions(self) -> list[dict]:
        if not self.ledger.exists():
            return []
        return [json.loads(line) for line in self.ledger.read_text().splitlines() if line.strip()]

    def frame(self, shas: list[str]) -> pd.DataFrame:
        return pd.concat([pd.read_csv(self.objects / f"{s}.csv") for s in shas], ignore_index=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Previous period's direction and price change. The first row of a batch has no
    predecessor and gets NaN, which the gradient-boosted trees handle natively."""
    out = df.copy()
    out["prev_label"] = out["label"].shift(1)
    out["nswprice_delta"] = out["nswprice"].diff()
    return out
