from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from typing import Any


class InteractionRecommender:
    """Deterministic item-to-item collaborative recommendation baseline."""

    def __init__(self, user_items: Mapping[str, set[str]]) -> None:
        self.user_items = {user: set(items) for user, items in user_items.items()}
        self.item_users: dict[str, set[str]] = defaultdict(set)
        self.popularity: Counter[str] = Counter()
        for user, items in self.user_items.items():
            for item in items:
                self.item_users[item].add(user)
                self.popularity[item] += 1

    @classmethod
    def from_records(
        cls,
        records: Iterable[Mapping[str, Any]],
        *,
        user_field: str = "user",
        item_field: str = "ASIN",
    ) -> "InteractionRecommender":
        user_items: dict[str, set[str]] = defaultdict(set)
        for record in records:
            user = record.get(user_field)
            item = record.get(item_field)
            if user is None or item is None:
                continue
            user_items[str(user)].add(str(item))
        return cls(user_items)

    def recommend(self, item_id: str, *, top_n: int = 3) -> list[tuple[str, float]]:
        source_users = self.item_users.get(item_id, set())
        if not source_users or top_n <= 0:
            return []

        scores: dict[str, float] = {}
        source_count = len(source_users)
        candidates = {item for user in source_users for item in self.user_items[user]}
        candidates.discard(item_id)
        for candidate in candidates:
            candidate_users = self.item_users[candidate]
            co_users = len(source_users & candidate_users)
            if co_users:
                scores[candidate] = co_users / math.sqrt(source_count * len(candidate_users))

        return sorted(
            scores.items(),
            key=lambda pair: (-pair[1], -self.popularity[pair[0]], pair[0]),
        )[:top_n]
