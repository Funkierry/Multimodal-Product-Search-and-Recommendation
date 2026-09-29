from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from typing import Any


def stable_partition(key: str, *, seed: int = 42) -> float:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def split_records(
    records: Iterable[Mapping[str, Any]],
    *,
    key_field: str = "asin",
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, list[Mapping[str, Any]]]:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if not 0 <= validation_ratio < 1 or train_ratio + validation_ratio >= 1:
        raise ValueError("validation_ratio leaves no test partition")

    result: dict[str, list[Mapping[str, Any]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for record in records:
        key = str(record[key_field])
        point = stable_partition(key, seed=seed)
        if point < train_ratio:
            partition = "train"
        elif point < train_ratio + validation_ratio:
            partition = "validation"
        else:
            partition = "test"
        result[partition].append(record)
    return result
