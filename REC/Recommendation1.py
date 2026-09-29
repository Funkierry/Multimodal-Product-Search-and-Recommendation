"""Command-line demonstration of interaction-based item recommendations."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fyp.recommendation.collaborative import InteractionRecommender


def load_interactions(path: Path) -> pd.DataFrame:
    dataframe = pd.read_csv(path, on_bad_lines="skip")
    required = {"ASIN", "user"}
    missing = required - set(dataframe.columns)
    if missing:
        raise ValueError(f"Interaction dataset is missing columns: {sorted(missing)}")
    return dataframe.dropna(subset=["ASIN", "user"]).drop_duplicates(
        subset=["ASIN", "user"]
    )


def build_recommender(dataframe: pd.DataFrame) -> InteractionRecommender:
    grouped_items = {
        str(user): set(group["ASIN"].astype(str))
        for user, group in dataframe.groupby("user")
    }
    return InteractionRecommender(grouped_items)


def main() -> int:
    data_dir = Path(os.environ.get("FYP_DATA_DIR", PROJECT_ROOT / "dataset")).resolve()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("item_id", help="ASIN to recommend related items for")
    parser.add_argument(
        "--interactions",
        type=Path,
        default=data_dir / "Dataset_Rec.csv",
    )
    parser.add_argument("--top-n", type=int, default=5)
    args = parser.parse_args()

    interactions = load_interactions(args.interactions)
    recommender = build_recommender(interactions)
    recommendations = recommender.recommend(args.item_id, top_n=args.top_n)
    if not recommendations:
        print(f"No co-purchase recommendations are available for {args.item_id}")
        return 0
    for rank, (asin, score) in enumerate(recommendations, start=1):
        print(f"{rank}. {asin}  score={score:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
