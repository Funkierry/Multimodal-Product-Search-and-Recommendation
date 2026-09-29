"""Convert absolute metadata image paths to portable data-directory paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def relativize_paths(items: list[dict], data_dir: Path) -> list[dict]:
    data_dir = data_dir.resolve()
    for item in items:
        raw_path = item.get("image_path")
        if not isinstance(raw_path, str) or not raw_path:
            continue
        path = Path(raw_path)
        if not path.is_absolute():
            continue
        try:
            item["image_path"] = path.resolve().relative_to(data_dir).as_posix()
        except ValueError:
            # Keep paths outside the declared data directory unchanged and visible.
            continue
    return items


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=project_root / "Search" / "items_meta.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=project_root / "dataset")
    args = parser.parse_args()

    items = json.loads(args.input.read_text(encoding="utf-8"))
    updated = relativize_paths(items, args.data_dir)
    args.output.write_text(
        json.dumps(updated, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(f"Saved portable metadata to {args.output}")


if __name__ == "__main__":
    main()
