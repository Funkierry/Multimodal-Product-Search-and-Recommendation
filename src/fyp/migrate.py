"""Upgrade verified legacy indexes without re-encoding product images."""

from __future__ import annotations

import argparse
import gc
import json
import os
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from .artifacts import (
    ArtifactManifest,
    resolve_image_path,
    validate_catalog_metadata,
    validate_index_rows,
    validate_normalized_vectors,
)
from .config import PROJECT_ROOT


def migrate_artifacts(data_dir: Path, search_dir: Path, content_dir: Path) -> Path:
    """Preserve legacy files, validate row order, and publish a schema 2 manifest."""
    import faiss

    data_dir = data_dir.expanduser().resolve()
    search_dir = search_dir.expanduser().resolve()
    content_dir = content_dir.expanduser().resolve()
    metadata_path = search_dir / "items_meta.json"
    backup_path = search_dir / "items_meta.legacy.json"
    search_index_path = search_dir / "faiss_index.index"
    search_embeddings_path = search_dir / "image_embeddings.npy"
    content_index_path = content_dir / "faiss_index_sim.index"
    content_embeddings_path = content_dir / "embeddings_sim.npy"
    manifest_path = content_dir / "manifest.json"
    pending_metadata_path = search_dir / "items_meta.pending.json"
    pending_embeddings_path = search_dir / "image_embeddings.pending.npy"
    pending_manifest_path = content_dir / "manifest.pending.json"

    if manifest_path.exists():
        raise FileExistsError(f"A manifest already exists: {manifest_path}")
    for path in (backup_path, search_embeddings_path, pending_metadata_path,
                 pending_embeddings_path, pending_manifest_path):
        if path.exists():
            raise FileExistsError(f"Migration output already exists: {path}")
    for path in (metadata_path, search_index_path, content_index_path, content_embeddings_path):
        if not path.is_file():
            raise FileNotFoundError(f"Legacy artifact is missing: {path}")

    original_items = json.loads(metadata_path.read_text(encoding="utf-8"))
    converted_items = []
    for position, item in enumerate(original_items):
        image_path, relative = resolve_image_path(item.get("image_path"), data_dir)
        if not image_path.is_file():
            raise FileNotFoundError(
                f"Image for ASIN {item.get('asin')} at metadata row {position} is missing: "
                f"{image_path}"
            )
        converted_items.append({**item, "image_path": relative})
    validate_catalog_metadata(converted_items)
    item_count = len(converted_items)

    search_index = faiss.read_index(str(search_index_path))
    content_index = faiss.read_index(str(content_index_path))
    content_vectors = np.load(content_embeddings_path, mmap_mode="r")
    if content_vectors.ndim != 2 or content_vectors.shape[0] != item_count:
        raise ValueError("Content embeddings and metadata have different item counts")
    dimension = content_vectors.shape[1]
    if search_index.ntotal != item_count or search_index.d != dimension:
        raise ValueError("Search index and metadata have different shapes")
    validate_normalized_vectors(content_vectors, "content embeddings")
    validate_index_rows(content_index, content_vectors, "content index")

    try:
        pending_metadata_path.write_text(
            json.dumps(converted_items, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        del original_items, converted_items, content_index, content_vectors
        gc.collect()

        search_vectors = np.lib.format.open_memmap(
            pending_embeddings_path,
            mode="w+",
            dtype=np.float32,
            shape=(item_count, dimension),
        )
        for start in range(0, item_count, 8192):
            count = min(8192, item_count - start)
            search_vectors[start : start + count] = search_index.reconstruct_n(start, count)
        search_vectors.flush()
        validate_normalized_vectors(search_vectors, "search embeddings")
        validate_index_rows(search_index, search_vectors, "search index")
        del search_vectors, search_index
        gc.collect()

        pending_manifest = ArtifactManifest.from_files(
            base_dir=content_dir,
            metadata_path=pending_metadata_path,
            search_index_path=search_index_path,
            search_embeddings_path=pending_embeddings_path,
            content_index_path=content_index_path,
            content_embeddings_path=content_embeddings_path,
            model_name="ViT-L-14",
            pretrained="openai",
            item_count=item_count,
            embedding_dimension=dimension,
        )
        pending_manifest_path.write_text(
            json.dumps(asdict(pending_manifest), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        ArtifactManifest.load(pending_manifest_path)

        final_manifest = replace(
            pending_manifest,
            metadata_path=os.path.relpath(metadata_path, content_dir),
            search_embeddings_path=os.path.relpath(search_embeddings_path, content_dir),
        )
        metadata_path.replace(backup_path)
        try:
            pending_metadata_path.replace(metadata_path)
            pending_embeddings_path.replace(search_embeddings_path)
            pending_manifest_path.write_text(
                json.dumps(asdict(final_manifest), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            pending_manifest_path.replace(manifest_path)
        except Exception:
            if metadata_path.exists():
                metadata_path.replace(pending_metadata_path)
            backup_path.replace(metadata_path)
            search_embeddings_path.unlink(missing_ok=True)
            raise
    finally:
        pending_metadata_path.unlink(missing_ok=True)
        pending_embeddings_path.unlink(missing_ok=True)
        pending_manifest_path.unlink(missing_ok=True)

    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--search-dir", type=Path, default=PROJECT_ROOT / "Search")
    parser.add_argument("--content-dir", type=Path, default=PROJECT_ROOT / "SIM")
    args = parser.parse_args()
    print(migrate_artifacts(args.data_dir, args.search_dir, args.content_dir))


if __name__ == "__main__":
    main()
