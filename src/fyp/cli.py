from __future__ import annotations

import argparse
from pathlib import Path

from .artifacts import ArtifactManifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate FYP generated artifacts")
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest = ArtifactManifest.load(args.manifest)
    print(
        f"Artifact set is valid: {manifest.item_count} items, "
        f"{manifest.embedding_dimension} dimensions, model={manifest.model_name}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
