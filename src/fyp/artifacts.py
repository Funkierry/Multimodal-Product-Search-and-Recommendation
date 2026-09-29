from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any


def resolve_image_path(raw_path: str, data_dir: str | Path) -> tuple[Path, str]:
    """Resolve a catalog image and return its path relative to the data directory."""
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("Product image path is missing")

    base = Path(data_dir).expanduser().resolve()
    candidate = Path(raw_path.strip()).expanduser()
    resolved = (candidate if candidate.is_absolute() else base / candidate).resolve()
    try:
        relative = resolved.relative_to(base)
    except ValueError as exc:
        raise ValueError(f"Product image path is outside the data directory: {raw_path}") from exc
    if not relative.parts:
        raise ValueError("Product image path points to the data directory")
    return resolved, relative.as_posix()


def validate_catalog_metadata(metadata: Any) -> None:
    """Keep one portable metadata row for each stable product identifier."""
    if not isinstance(metadata, list) or not metadata:
        raise ValueError("Product metadata must be a non-empty list")

    seen_asins: set[str] = set()
    for position, item in enumerate(metadata):
        if not isinstance(item, dict):
            raise ValueError(f"Metadata row {position} must be an object")
        asin = item.get("asin")
        if not isinstance(asin, str) or not asin.strip():
            raise ValueError(f"Metadata row {position} has no ASIN")
        if asin != asin.strip():
            raise ValueError(f"Metadata row {position} has an unnormalized ASIN")
        if asin in seen_asins:
            raise ValueError(f"Duplicate ASIN in metadata: {asin}")
        seen_asins.add(asin)

        raw_path = item.get("image_path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError(f"Metadata row {position} has no image path")
        normalized = raw_path.replace("\\", "/")
        path = PurePosixPath(normalized)
        if path.is_absolute() or PureWindowsPath(raw_path).is_absolute():
            raise ValueError(f"Metadata row {position} has an absolute image path")
        if ".." in path.parts or normalized in (".", ""):
            raise ValueError(f"Metadata row {position} has an unsafe image path")


def validate_normalized_vectors(vectors: Any, name: str, batch_size: int = 8192) -> None:
    """Reject missing, non-finite, or zero vectors before they reach Faiss."""
    import numpy as np

    if vectors.ndim != 2 or vectors.shape[0] == 0 or vectors.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix")
    if vectors.dtype != np.float32:
        raise ValueError(f"{name} must use float32 vectors")
    for start in range(0, vectors.shape[0], batch_size):
        chunk = vectors[start : start + batch_size]
        norms = np.linalg.norm(chunk, axis=1)
        if not np.all(np.isfinite(chunk)) or not np.all(np.isfinite(norms)):
            raise ValueError(f"{name} contains non-finite vectors near row {start}")
        if np.any(norms < 1e-6):
            raise ValueError(f"{name} contains zero vectors near row {start}")
        if not np.allclose(norms, 1.0, atol=1e-3, rtol=0):
            raise ValueError(f"{name} contains non-normalized vectors near row {start}")


def validate_index_rows(index: Any, vectors: Any, name: str, batch_size: int = 8192) -> None:
    """Check that each exact Faiss index row matches its saved vector row."""
    import numpy as np

    if index.ntotal != vectors.shape[0] or index.d != vectors.shape[1]:
        raise ValueError(
            f"{name} has shape ({index.ntotal}, {index.d}), expected {vectors.shape}"
        )
    for start in range(0, vectors.shape[0], batch_size):
        count = min(batch_size, vectors.shape[0] - start)
        reconstructed = index.reconstruct_n(start, count)
        if not np.allclose(reconstructed, vectors[start : start + count], atol=1e-6):
            raise ValueError(f"{name} row order differs from embeddings near row {start}")


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
    search_embeddings_path: str
    content_index_path: str
    content_embeddings_path: str
    search_index_sha256: str
    search_embeddings_sha256: str
    content_index_sha256: str
    content_embeddings_sha256: str

    @classmethod
    def from_files(
        cls,
        *,
        base_dir: str | Path,
        metadata_path: str | Path,
        search_index_path: str | Path,
        search_embeddings_path: str | Path,
        content_index_path: str | Path,
        content_embeddings_path: str | Path,
        model_name: str,
        pretrained: str,
        item_count: int,
        embedding_dimension: int,
    ) -> "ArtifactManifest":
        base = Path(base_dir)
        return cls(
            schema_version=2,
            dataset_sha256=sha256_file(metadata_path),
            model_name=model_name,
            pretrained=pretrained,
            embedding_dimension=embedding_dimension,
            item_count=item_count,
            metadata_path=os.path.relpath(metadata_path, base),
            search_index_path=os.path.relpath(search_index_path, base),
            search_embeddings_path=os.path.relpath(search_embeddings_path, base),
            content_index_path=os.path.relpath(content_index_path, base),
            content_embeddings_path=os.path.relpath(content_embeddings_path, base),
            search_index_sha256=sha256_file(search_index_path),
            search_embeddings_sha256=sha256_file(search_embeddings_path),
            content_index_sha256=sha256_file(content_index_path),
            content_embeddings_sha256=sha256_file(content_embeddings_path),
        )

    @classmethod
    def load(
        cls, path: str | Path, *, verify_index_rows: bool = True
    ) -> "ArtifactManifest":
        manifest_path = Path(path)
        data: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
        if data.get("schema_version") != 2:
            raise ValueError(
                f"Unsupported manifest schema: {data.get('schema_version')}; rebuild artifacts"
            )
        manifest = cls(**data)
        manifest.validate(manifest_path.parent, verify_index_rows=verify_index_rows)
        return manifest

    def validate(self, base_dir: str | Path, *, verify_index_rows: bool = True) -> None:
        if self.schema_version != 2:
            raise ValueError(f"Unsupported manifest schema: {self.schema_version}; rebuild artifacts")
        if self.embedding_dimension <= 0 or self.item_count <= 0:
            raise ValueError("Manifest dimensions and counts must be positive")
        base = Path(base_dir)
        files = {
            "metadata": (self.metadata_path, self.dataset_sha256),
            "search index": (self.search_index_path, self.search_index_sha256),
            "search embeddings": (self.search_embeddings_path, self.search_embeddings_sha256),
            "content index": (self.content_index_path, self.content_index_sha256),
            "content embeddings": (self.content_embeddings_path, self.content_embeddings_sha256),
        }
        for name, (relative_path, expected_digest) in files.items():
            path = base / relative_path
            if not path.is_file():
                raise FileNotFoundError(f"Missing {name}: {path}")
            if sha256_file(path) != expected_digest:
                raise ValueError(f"{name} checksum does not match the manifest")

        metadata = json.loads((base / self.metadata_path).read_text(encoding="utf-8"))
        validate_catalog_metadata(metadata)
        if len(metadata) != self.item_count:
            raise ValueError(
                f"Metadata count {len(metadata)} does not match manifest {self.item_count}"
            )

        import numpy as np

        matrices = {
            "search embeddings": np.load(base / self.search_embeddings_path, mmap_mode="r"),
            "content embeddings": np.load(base / self.content_embeddings_path, mmap_mode="r"),
        }
        expected_shape = (self.item_count, self.embedding_dimension)
        for name, matrix in matrices.items():
            if matrix.shape != expected_shape:
                raise ValueError(f"{name} shape {matrix.shape} does not match {expected_shape}")
            validate_normalized_vectors(matrix, name)

        if verify_index_rows:
            if importlib.util.find_spec("faiss") is None:
                raise RuntimeError("faiss is required to verify index row order")
            import faiss

            for name, index_path, vectors in (
                ("search index", self.search_index_path, matrices["search embeddings"]),
                ("content index", self.content_index_path, matrices["content embeddings"]),
            ):
                index = faiss.read_index(str(base / index_path))
                validate_index_rows(index, vectors, name)


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
