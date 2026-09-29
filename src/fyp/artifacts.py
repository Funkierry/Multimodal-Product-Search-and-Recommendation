from __future__ import annotations

import hashlib
import json
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ArtifactManifest:
    schema_version: int
    dataset_sha256: str
    model_name: str
    pretrained: str
    embedding_dimension: int
    item_count: int
    metadata_path: str
    search_index_path: str
    content_index_path: str
    content_embeddings_path: str

    @classmethod
    def load(cls, path: str | Path) -> "ArtifactManifest":
        manifest_path = Path(path)
        data: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = cls(**data)
        manifest.validate(manifest_path.parent)
        return manifest

    def validate(self, base_dir: str | Path) -> None:
        if self.schema_version != 1:
            raise ValueError(f"Unsupported manifest schema: {self.schema_version}")
        if self.embedding_dimension <= 0 or self.item_count < 0:
            raise ValueError("Manifest dimensions and counts must be valid")
        base = Path(base_dir)
        missing = [
            name
            for name in (
                self.metadata_path,
                self.search_index_path,
                self.content_index_path,
                self.content_embeddings_path,
            )
            if not (base / name).is_file()
        ]
        if missing:
            raise FileNotFoundError(f"Missing artifact files: {', '.join(missing)}")

        metadata_file = base / self.metadata_path
        if sha256_file(metadata_file) != self.dataset_sha256:
            raise ValueError("Metadata checksum does not match the manifest")
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        if len(metadata) != self.item_count:
            raise ValueError(
                f"Metadata count {len(metadata)} does not match manifest {self.item_count}"
            )

        import numpy as np

        embeddings = np.load(base / self.content_embeddings_path, mmap_mode="r")
        expected_shape = (self.item_count, self.embedding_dimension)
        if embeddings.shape != expected_shape:
            raise ValueError(
                f"Embedding shape {embeddings.shape} does not match {expected_shape}"
            )

        if importlib.util.find_spec("faiss") is not None:
            import faiss

            for index_name in (self.search_index_path, self.content_index_path):
                index = faiss.read_index(str(base / index_name))
                if index.ntotal != self.item_count or index.d != self.embedding_dimension:
                    raise ValueError(
                        f"Index {index_name} has shape ({index.ntotal}, {index.d}), "
                        f"expected ({self.item_count}, {self.embedding_dimension})"
                    )


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
