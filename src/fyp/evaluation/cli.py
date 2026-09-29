from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from fyp.artifacts import ArtifactManifest, sha256_file
from fyp.config import PROJECT_ROOT, ProjectPaths

from .recommendation import evaluate_recommendations
from .search import (
    collect_openclip_rankings,
    evaluate_search,
    read_jsonl,
)


def _save_report(report: dict, output: Path | None) -> None:
    formatted = json.dumps(report, ensure_ascii=False, indent=2)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(formatted + "\n", encoding="utf-8")
        print(f"Saved report to {output}")
    else:
        print(formatted)


def recommendation_main() -> None:
    paths = ProjectPaths.from_environment()
    parser = argparse.ArgumentParser(description="Evaluate simulated user-item interactions")
    parser.add_argument("--interactions", type=Path,
                        default=paths.data_dir / "Dataset_Rec.csv")
    parser.add_argument("--metadata", type=Path,
                        default=PROJECT_ROOT / "Search" / "items_meta.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "results" / "recommendation_eval.json")
    args = parser.parse_args()

    interactions = pd.read_csv(args.interactions)
    if not {"ASIN", "user"}.issubset(interactions.columns):
        raise ValueError("Interactions must contain ASIN and user columns")
    interactions = interactions.dropna(subset=["ASIN", "user"]).drop_duplicates(
        subset=["ASIN", "user"]
    )
    interaction_items = set(interactions["ASIN"].astype(str))
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    categories = {
        item["asin"]: str(item["categoryName"])
        for item in metadata
        if item.get("asin") in interaction_items and item.get("categoryName")
    }
    report = evaluate_recommendations(
        interactions[["user", "ASIN"]].to_dict("records"), categories, seed=args.seed
    )
    report["interaction_pairs"] = len(interactions)
    report["interactions_sha256"] = sha256_file(args.interactions)
    report["catalog_sha256"] = sha256_file(args.metadata)
    _save_report(report, args.output)


def search_main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate judged product search queries")
    parser.add_argument("--judgments", required=True, type=Path)
    parser.add_argument("--rankings", type=Path,
                        help="Precomputed JSONL rankings; otherwise run OpenCLIP")
    parser.add_argument("--manifest", type=Path,
                        default=PROJECT_ROOT / "SIM" / "manifest.json")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--candidates", type=int, default=200)
    parser.add_argument("--save-rankings", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    judgments = read_jsonl(args.judgments)
    if args.rankings is not None:
        rankings = read_jsonl(args.rankings)
    else:
        validation_start = time.perf_counter()
        manifest = ArtifactManifest.load(args.manifest)
        validation_ms = (time.perf_counter() - validation_start) * 1000
        base = args.manifest.parent
        rankings, model_load_ms = collect_openclip_rankings(
            judgments,
            base / manifest.search_index_path,
            base / manifest.metadata_path,
            model_name=manifest.model_name,
            pretrained=manifest.pretrained,
            top_k=args.top_k, candidate_count=args.candidates,
        )
        cold_start_ms = validation_ms + model_load_ms
        if args.save_rankings is not None:
            args.save_rankings.parent.mkdir(parents=True, exist_ok=True)
            args.save_rankings.write_text(
                "".join(json.dumps(row) + "\n" for row in rankings), encoding="utf-8"
            )
    report = evaluate_search(judgments, rankings)
    report["judgments_sha256"] = sha256_file(args.judgments)
    if args.rankings is not None:
        report["rankings_sha256"] = sha256_file(args.rankings)
    else:
        report["manifest_sha256"] = sha256_file(args.manifest)
        report["top_k"] = args.top_k
        report["candidate_count"] = args.candidates
        report["cold_start_ms"] = cold_start_ms
    _save_report(report, args.output)
