"""Build a JSON label mapping from the training partition only."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("FYP_DATA_DIR", PROJECT_ROOT / "dataset")).resolve()
RESULTS_DIR = Path(os.environ.get("FYP_RESULTS_DIR", PROJECT_ROOT / "results")).resolve()


def main() -> None:
    train_path = DATA_DIR / "train_updated.csv"
    dataframe = pd.read_csv(train_path, encoding="utf-8")
    classes = sorted(dataframe["categoryName"].dropna().astype(str).unique().tolist())
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = RESULTS_DIR / "label_mapping.json"
    output.write_text(
        json.dumps({"classes": classes}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved {len(classes)} classes to {output}")


if __name__ == "__main__":
    main()
