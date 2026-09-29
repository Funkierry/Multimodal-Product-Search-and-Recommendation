from __future__ import annotations

import math


def ranking_metrics(
    ranked_items: list[str], relevant_items: set[str], *, cutoffs: tuple[int, ...] = (5, 10, 20)
) -> dict[str, float]:
    """Return binary relevance metrics for one query or held-out user item."""
    if not relevant_items:
        raise ValueError("At least one relevant item is required")
    if len(ranked_items) != len(set(ranked_items)):
        raise ValueError("A ranking contains duplicate items")
    if any(k < 1 for k in cutoffs):
        raise ValueError("Cutoffs must be positive")

    first_hit = next((rank for rank, item in enumerate(ranked_items, 1)
                      if item in relevant_items), None)
    result = {"mrr": 1.0 / first_hit if first_hit is not None else 0.0}
    for k in cutoffs:
        prefix = ranked_items[:k]
        hits = sum(item in relevant_items for item in prefix)
        result[f"recall@{k}"] = hits / len(relevant_items)
        dcg = sum(
            1.0 / math.log2(rank + 1)
            for rank, item in enumerate(prefix, 1)
            if item in relevant_items
        )
        ideal = sum(1.0 / math.log2(rank + 1)
                    for rank in range(1, min(k, len(relevant_items)) + 1))
        result[f"ndcg@{k}"] = dcg / ideal
    return result


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        raise ValueError("No evaluated rankings")
    return {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}
