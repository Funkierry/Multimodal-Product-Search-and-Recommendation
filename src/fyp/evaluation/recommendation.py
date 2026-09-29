"""Deterministic offline evaluation for the simulated interaction dataset."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping

from fyp.recommendation.collaborative import InteractionRecommender

from .metrics import mean_metrics, ranking_metrics


def leave_one_out(
    records: Iterable[Mapping], *, seed: int = 42
) -> tuple[dict[str, set[str]], dict[str, str], int]:
    """Hold out one warm item per eligible user without using row order."""
    user_items: dict[str, set[str]] = defaultdict(set)
    for record in records:
        raw_user, raw_item = record.get("user"), record.get("ASIN")
        if raw_user is None or raw_item is None:
            continue
        user, item = str(raw_user).strip(), str(raw_item).strip()
        if user and item:
            user_items[user].add(item)

    remaining = Counter(item for items in user_items.values() for item in items)
    training = {user: set(items) for user, items in user_items.items()}
    held_out = {}
    for user in sorted(user_items):
        if len(user_items[user]) < 2:
            continue
        candidates = [item for item in user_items[user] if remaining[item] > 1]
        if not candidates:
            continue
        chosen = min(
            candidates,
            key=lambda item: hashlib.sha256(f"{seed}:{user}:{item}".encode()).digest(),
        )
        training[user].remove(chosen)
        remaining[chosen] -= 1
        held_out[user] = chosen
    return training, held_out, len(user_items) - len(held_out)


def _popularity_rank(popularity: Counter[str], seen: set[str]) -> list[str]:
    return sorted(
        (item for item in popularity if item not in seen),
        key=lambda item: (-popularity[item], item),
    )


def rank_for_user(
    method: str,
    seen: set[str],
    recommender: InteractionRecommender,
    categories: Mapping[str, str],
    *,
    top_k: int = 10,
) -> list[str]:
    popularity = recommender.popularity
    fallback = _popularity_rank(popularity, seen)
    if method == "popularity":
        return fallback[:top_k]
    if method == "same_category":
        seen_categories = {categories[item] for item in seen if categories.get(item)}
        preferred = [item for item in fallback if categories.get(item) in seen_categories]
        preferred_set = set(preferred)
        return (preferred + [item for item in fallback if item not in preferred_set])[:top_k]
    if method != "co_purchase":
        raise ValueError(f"Unknown recommendation method: {method}")

    scores: dict[str, float] = defaultdict(float)
    for source in sorted(seen):
        for candidate, score in recommender.recommend(source, top_n=len(popularity)):
            if candidate not in seen:
                scores[candidate] += score
    ranked = sorted(scores, key=lambda item: (-scores[item], -popularity[item], item))
    return (ranked + [item for item in fallback if item not in scores])[:top_k]


def evaluate_recommendations(
    records: Iterable[Mapping], categories: Mapping[str, str], *, seed: int = 42
) -> dict:
    training, held_out, excluded_users = leave_one_out(records, seed=seed)
    if not held_out:
        raise ValueError("No users have an eligible warm-item holdout")
    recommender = InteractionRecommender(training)
    catalog = set(recommender.popularity)
    split = {
        "training": {user: sorted(items) for user, items in sorted(training.items())},
        "held_out": dict(sorted(held_out.items())),
    }
    split_sha256 = hashlib.sha256(
        json.dumps(split, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    methods = {}
    for method in ("co_purchase", "popularity", "same_category"):
        rows = []
        recommended_items = set()
        diversities = []
        recommended_popularity = []
        for user, target in sorted(held_out.items()):
            ranked = rank_for_user(method, training[user], recommender, categories)
            rows.append(ranking_metrics(ranked, {target}, cutoffs=(5, 10)))
            recommended_items.update(ranked)
            diversities.append(len({categories[item] for item in ranked if categories.get(item)}))
            recommended_popularity.extend(recommender.popularity[item] for item in ranked)
        methods[method] = {
            "metrics": mean_metrics(rows),
            "catalog_coverage@10": len(recommended_items) / len(catalog),
            "mean_distinct_categories@10": sum(diversities) / len(diversities),
            "mean_training_item_popularity@10": (
                sum(recommended_popularity) / len(recommended_popularity)
            ),
        }

    return {
        "protocol": "deterministic per-user leave-one-out, warm items only",
        "data_note": "The project report describes these user-item interactions as simulated; no timestamps are available.",
        "seed": seed,
        "split_sha256": split_sha256,
        "input_users": len(training),
        "evaluated_users": len(held_out),
        "excluded_users": excluded_users,
        "training_catalog_items": len(catalog),
        "items_with_known_category": len(catalog & set(categories)),
        "methods": methods,
    }
