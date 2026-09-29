from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_path(value: str | Path, *, base: Path = PROJECT_ROOT) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    data_dir: Path
    artifact_dir: Path

    @classmethod
    def from_environment(cls) -> "ProjectPaths":
        root = _resolve_path(os.environ.get("FYP_PROJECT_ROOT", PROJECT_ROOT))
        data_dir = _resolve_path(os.environ.get("FYP_DATA_DIR", "dataset"), base=root)
        artifact_dir = _resolve_path(
            os.environ.get("FYP_ARTIFACT_DIR", "artifacts"), base=root
        )
        return cls(root=root, data_dir=data_dir, artifact_dir=artifact_dir)


def configured_device() -> str:
    """Return the configured device without importing torch at module import time."""
    return os.environ.get("FYP_DEVICE", "auto").strip().lower()


def search_candidate_count() -> int:
    raw_value = os.environ.get("FYP_SEARCH_CANDIDATES", "200")
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError("FYP_SEARCH_CANDIDATES must be an integer") from exc
    if value < 1:
        raise ValueError("FYP_SEARCH_CANDIDATES must be positive")
    return value
