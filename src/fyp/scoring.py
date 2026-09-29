from __future__ import annotations

from collections.abc import Iterable
from typing import Any

SearchResult = tuple[dict[str, Any], float]


def similarity_percent(similarity: float) -> float:
    """Map cosine similarity [-1, 1] to a bounded display score [0, 100]."""
    bounded = max(-1.0, min(1.0, float(similarity)))
    return (bounded + 1.0) * 50.0


def sort_by_similarity(results: Iterable[SearchResult]) -> list[SearchResult]:
    """Return highest-similarity results first."""
    return sorted(results, key=lambda result: result[1], reverse=True)


def filter_results(
    results: Iterable[SearchResult],
    *,
    min_price: float | None = None,
    max_price: float | None = None,
    category: str | None = None,
) -> list[SearchResult]:
    category_key = category.casefold() if category and category.casefold() != "all" else None
    filtered: list[SearchResult] = []
    for item, similarity in results:
        try:
            price = float(item["price"])
        except (KeyError, TypeError, ValueError):
            continue
        if min_price is not None and price < min_price:
            continue
        if max_price is not None and price > max_price:
            continue
        if category_key and str(item.get("categoryName", "")).casefold() != category_key:
            continue
        filtered.append((item, similarity))
    return sort_by_similarity(filtered)
