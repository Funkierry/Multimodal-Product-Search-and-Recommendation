"""Build an image-and-title index in the exact order of the search catalog."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import faiss
import numpy as np
import open_clip
import torch
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fyp.artifacts import (
    ArtifactManifest,
    resolve_image_path,
    validate_catalog_metadata,
    validate_index_rows,
    validate_normalized_vectors,
)


class EmbeddingBuilder:
    MODEL_NAME = "ViT-L-14"
    PRETRAINED = "openai"

    def __init__(self):
        self.META_JSON = Path(
            os.environ.get("FYP_ITEMS_META", PROJECT_ROOT / "Search" / "items_meta.json")
        ).resolve()
        self.DATA_DIR = Path(os.environ.get("FYP_DATA_DIR", PROJECT_ROOT / "dataset")).resolve()
        self.SAVE_DIR = Path(
            os.environ.get("PRODUCT_CONTENT_DIR", PROJECT_ROOT / "SIM")
        ).resolve()
        self.INDEX_PATH = self.SAVE_DIR / "faiss_index_sim.index"
        self.EMB_PATH = self.SAVE_DIR / "embeddings_sim.npy"
        self.SEARCH_INDEX_PATH = self.META_JSON.parent / "faiss_index.index"
        self.SEARCH_EMB_PATH = self.META_JSON.parent / "image_embeddings.npy"
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.items_meta: list[dict] = []
        self.embeddings: list[np.ndarray | None] = []

        self.load_data()
        self.init_clip_model()

    def load_data(self):
        if not self.META_JSON.is_file():
            raise FileNotFoundError(f"Search catalog is missing: {self.META_JSON}")
        self.items_meta = json.loads(self.META_JSON.read_text(encoding="utf-8"))
        validate_catalog_metadata(self.items_meta)

        for path in (self.SEARCH_INDEX_PATH, self.SEARCH_EMB_PATH):
            if not path.is_file():
                raise FileNotFoundError(f"Search artifact is missing: {path}; rebuild search index")
        search_vectors = np.load(self.SEARCH_EMB_PATH, mmap_mode="r")
        if search_vectors.ndim != 2:
            raise ValueError("Search embeddings must be a two-dimensional matrix")
        expected_shape = (len(self.items_meta), search_vectors.shape[1])
        if search_vectors.shape != expected_shape:
            raise ValueError("Search embeddings and metadata have different item counts")
        validate_normalized_vectors(search_vectors, "search embeddings")

        search_index = faiss.read_index(str(self.SEARCH_INDEX_PATH))
        validate_index_rows(search_index, search_vectors, "search index")
        print(f"Loaded and checked {len(self.items_meta)} ordered search items")

    def init_clip_model(self):
        self.model, _, self.img_transform = open_clip.create_model_and_transforms(
            self.MODEL_NAME, pretrained=self.PRETRAINED, device=self.DEVICE
        )
        self.tokenizer = open_clip.get_tokenizer(self.MODEL_NAME)
        self.model.eval()

    def compute_embeddings(self):
        batch_size = int(os.environ.get("FYP_EMBEDDING_BATCH_SIZE", "64"))
        if batch_size < 1:
            raise ValueError("FYP_EMBEDDING_BATCH_SIZE must be positive")
        self.embeddings = [None] * len(self.items_meta)
        batch_indices: list[int] = []
        image_tensors = []
        titles: list[str] = []

        def flush_batch():
            if not batch_indices:
                return
            image_batch = torch.stack(image_tensors).to(self.DEVICE)
            text_tokens = self.tokenizer(titles).to(self.DEVICE)
            with torch.inference_mode():
                image_features = self.model.encode_image(image_batch)
                text_features = self.model.encode_text(text_tokens)
                fused = 0.5 * image_features + 0.5 * text_features
                fused = torch.nn.functional.normalize(fused, dim=-1)
            for item_index, embedding in zip(batch_indices, fused.cpu().numpy()):
                self.embeddings[item_index] = embedding.astype(np.float32)
            batch_indices.clear()
            image_tensors.clear()
            titles.clear()

        for idx, item in enumerate(tqdm(self.items_meta, desc="Computing content embeddings")):
            try:
                image_path, _ = resolve_image_path(item["image_path"], self.DATA_DIR)
                with Image.open(image_path) as source:
                    image_tensors.append(self.img_transform(source.convert("RGB")))
            except (OSError, ValueError) as exc:
                raise RuntimeError(
                    f"Cannot build aligned content index: image for {item['asin']} "
                    f"at metadata row {idx} is unavailable"
                ) from exc
            batch_indices.append(idx)
            titles.append(str(item.get("title") or ""))
            if len(batch_indices) >= batch_size:
                flush_batch()
        flush_batch()
        if any(embedding is None for embedding in self.embeddings):
            raise RuntimeError("Some products have no content embedding")

    def build_faiss_index(self):
        if len(self.embeddings) != len(self.items_meta) or not self.embeddings:
            raise ValueError("Content embeddings must match the ordered catalog")
        matrix = np.asarray(self.embeddings, dtype=np.float32)
        validate_normalized_vectors(matrix, "content embeddings")
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)

        self.SAVE_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(self.INDEX_PATH))
        np.save(self.EMB_PATH, matrix)

        manifest = ArtifactManifest.from_files(
            base_dir=self.SAVE_DIR,
            metadata_path=self.META_JSON,
            search_index_path=self.SEARCH_INDEX_PATH,
            search_embeddings_path=self.SEARCH_EMB_PATH,
            content_index_path=self.INDEX_PATH,
            content_embeddings_path=self.EMB_PATH,
            model_name=self.MODEL_NAME,
            pretrained=self.PRETRAINED,
            embedding_dimension=matrix.shape[1],
            item_count=len(self.items_meta),
        )
        manifest_path = self.SAVE_DIR / "manifest.json"
        pending_path = self.SAVE_DIR / "manifest.pending.json"
        try:
            pending_path.write_text(
                json.dumps(asdict(manifest), ensure_ascii=False, indent=2), encoding="utf-8"
            )
            ArtifactManifest.load(pending_path)
            pending_path.replace(manifest_path)
        finally:
            pending_path.unlink(missing_ok=True)
        print(f"Saved validated artifact manifest to {manifest_path}")

    def run(self):
        self.compute_embeddings()
        self.build_faiss_index()


if __name__ == "__main__":
    EmbeddingBuilder().run()
