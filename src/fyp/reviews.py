"""Offline review summaries and lightweight runtime loading."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from fyp.config import ProjectPaths


def load_review_summaries(path: str | Path) -> dict:
    """Load prepared summaries without importing the NLP pipeline."""
    with Path(path).open(encoding="utf-8") as source:
        data = json.load(source)
    if not isinstance(data, dict):
        raise ValueError("Review summaries must be a JSON object keyed by ASIN")
    return data


def _sentiment(review: str, analyzer) -> int:
    score = analyzer.polarity_scores(review)["compound"]
    if score <= -0.05:
        return -1
    if score >= 0.05:
        return 1
    return 0


def _adjectives(review: str, nlp) -> list[str]:
    return [
        token.lemma_
        for token in nlp(review.lower())
        if token.pos_ == "ADJ" and not token.is_stop
    ]


def build_review_summaries(
    csv_path: str | Path,
    output_path: str | Path,
    *,
    nlp=None,
    analyzer=None,
) -> dict:
    """Build the UI's per-ASIN aspect counts from the interaction CSV."""
    import pandas as pd

    csv_path = Path(csv_path)
    output_path = Path(output_path)
    data = pd.read_csv(csv_path, on_bad_lines="skip")
    original_columns = {name.lower(): name for name in data.columns}
    required = {"asin": "ASIN", "user": "user", "review": "Review"}
    missing = sorted(name for name in required if name not in original_columns)
    if missing:
        raise ValueError(f"Review CSV is missing columns: {missing}")
    data = data.rename(
        columns={original_columns[key]: canonical for key, canonical in required.items()}
    )
    data = data.dropna(subset=["ASIN", "Review"])
    data["Review"] = data["Review"].astype(str)
    data = data.drop_duplicates(subset=["ASIN", "user", "Review"])

    if nlp is None:
        import spacy

        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError as exc:
            raise RuntimeError(
                "Install en_core_web_sm before building review summaries: "
                "python -m spacy download en_core_web_sm"
            ) from exc
    if analyzer is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        analyzer = SentimentIntensityAnalyzer()

    summaries = {}
    for asin, group in data.groupby("ASIN", sort=True):
        positive = Counter()
        negative = Counter()
        for review in group["Review"]:
            adjectives = _adjectives(review, nlp)
            sentiment = _sentiment(review, analyzer)
            if sentiment > 0:
                positive.update(adjectives)
            elif sentiment < 0:
                negative.update(adjectives)

        top_positive = positive.most_common(6)
        top_negative = negative.most_common(6)
        aspects = list(dict.fromkeys(
            [word for word, _ in top_positive] + [word for word, _ in top_negative]
        ))
        if not aspects:
            continue
        summaries[str(asin)] = {
            "aspects": aspects,
            "positiveScores": [positive.get(word, 0) for word in aspects],
            "negativeScores": [negative.get(word, 0) for word in aspects],
            "positiveKeywords": [word for word, count in top_positive for _ in range(count)],
            "negativeKeywords": [word for word, count in top_negative for _ in range(count)],
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output_path)
    return summaries


def main() -> None:
    paths = ProjectPaths.from_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=paths.data_dir / "Dataset_Rec.csv")
    parser.add_argument("--output", type=Path, default=paths.root / "reviews_data.json")
    args = parser.parse_args()
    summaries = build_review_summaries(args.input, args.output)
    print(f"Saved summaries for {len(summaries)} products to {args.output}")


if __name__ == "__main__":
    main()
