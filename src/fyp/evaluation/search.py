from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from .metrics import mean_metrics, ranking_metrics


def read_jsonl(path: str | Path) -> list[dict]:
    records = []
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"JSONL row {line_number} must be an object")
            records.append(record)
    return records


def validate_judgments(judgments: list[dict]) -> None:
    if not judgments:
        raise ValueError("Judgments file is empty")
    query_ids = set()
    for row in judgments:
        query_id = row.get("query_id")
        query = row.get("query")
        relevant = row.get("relevant_asins")
        if not isinstance(query_id, str) or not query_id.strip() or query_id in query_ids:
            raise ValueError(f"Query ID is missing or duplicated: {query_id}")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"Query text is missing for {query_id}")
        if (not isinstance(relevant, list) or not relevant
                or any(not isinstance(asin, str) or not asin for asin in relevant)
                or len(relevant) != len(set(relevant))):
            raise ValueError(f"Relevant ASINs are missing or duplicated for {query_id}")
        if "filters" in row and not isinstance(row["filters"], dict):
            raise ValueError(f"Filters must be an object for {query_id}")
        query_ids.add(query_id)


def evaluate_search(judgments: list[dict], rankings: list[dict]) -> dict:
    validate_judgments(judgments)
    ranking_by_id = {}
    for row in rankings:
        query_id = row.get("query_id")
        if query_id in ranking_by_id:
            raise ValueError(f"Duplicate ranking for {query_id}")
        ranking_by_id[query_id] = row
    expected_ids = {row["query_id"] for row in judgments}
    if set(ranking_by_id) != expected_ids:
        raise ValueError("Judgments and rankings must have the same query IDs")

    scores = []
    by_segment: dict[str, list[dict[str, float]]] = {}
    latencies = []
    for row in judgments:
        ranking = ranking_by_id[row["query_id"]]
        asins = ranking.get("asins")
        if not isinstance(asins, list) or any(not isinstance(x, str) for x in asins):
            raise ValueError(f"Invalid ranking for {row['query_id']}")
        metrics = ranking_metrics(asins, set(row["relevant_asins"]))
        scores.append(metrics)
        default_segment = "filtered" if row.get("filters") else "unfiltered"
        by_segment.setdefault(str(row.get("segment", default_segment)), []).append(metrics)
        latency = ranking.get("latency_ms")
        if latency is not None:
            if (not isinstance(latency, (int, float))
                    or not math.isfinite(latency) or latency < 0):
                raise ValueError(f"Invalid latency for {row['query_id']}")
            latencies.append(latency)

    report = {
        "query_count": len(scores),
        "metrics": mean_metrics(scores),
        "segments": {
            segment: {"query_count": len(rows), "metrics": mean_metrics(rows)}
            for segment, rows in sorted(by_segment.items())
        },
    }
    if latencies:
        report["latency_ms"] = {
            "measured_queries": len(latencies),
            "p50": float(np.percentile(latencies, 50)),
            "p95": float(np.percentile(latencies, 95)),
        }
    return report


def matches_filters(item: dict, filters: dict) -> bool:
    if not isinstance(filters, dict):
        raise ValueError("Query filters must be an object")
    category = filters.get("category")
    if category and str(item.get("categoryName", "")).casefold() != str(category).casefold():
        return False
    if "min_price" in filters or "max_price" in filters:
        try:
            price = float(str(item.get("price", "")).replace("£", "").replace(",", ""))
        except (TypeError, ValueError):
            return False
        if not math.isfinite(price):
            return False
        if "min_price" in filters and price < float(filters["min_price"]):
            return False
        if "max_price" in filters and price > float(filters["max_price"]):
            return False
    return True


def collect_openclip_rankings(
    judgments: list[dict], index_path: Path, metadata_path: Path, *,
    model_name: str = "ViT-L-14", pretrained: str = "openai",
    top_k: int = 20, candidate_count: int = 200
) -> tuple[list[dict], float]:
    """Run the exact text-to-image baseline for a judged query set."""
    import faiss
    import open_clip
    import torch

    validate_judgments(judgments)
    if top_k < 1 or candidate_count < top_k:
        raise ValueError("top_k must be positive and no larger than candidate_count")
    load_start = time.perf_counter()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    index = faiss.read_index(str(index_path))
    if index.ntotal != len(metadata):
        raise ValueError("Search index and metadata counts differ")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, _ = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    cold_start_ms = (time.perf_counter() - load_start) * 1000

    rankings = []
    for row in judgments:
        start = time.perf_counter()
        with torch.inference_mode():
            tokens = tokenizer([row["query"]]).to(device)
            query_vector = model.encode_text(tokens)
            query_vector = torch.nn.functional.normalize(query_vector, dim=-1)
            query_vector = query_vector.cpu().numpy().astype(np.float32)
        filters = row.get("filters", {})
        requested = candidate_count if filters else top_k
        _, indices = index.search(query_vector, requested)
        asins = []
        for position in indices[0]:
            if not 0 <= position < len(metadata):
                continue
            item = metadata[int(position)]
            if matches_filters(item, filters):
                asins.append(item["asin"])
            if len(asins) >= top_k:
                break
        latency_ms = (time.perf_counter() - start) * 1000
        rankings.append({"query_id": row["query_id"], "asins": asins,
                         "latency_ms": latency_ms})
    return rankings, cold_start_ms
