import importlib.util
import json
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np

from fyp.artifacts import (
    ArtifactManifest,
    resolve_image_path,
    sha256_file,
    validate_catalog_metadata,
    validate_normalized_vectors,
)


def make_artifacts(root: Path) -> Path:
    metadata_path = root / "items.json"
    metadata_path.write_text(
        json.dumps(
            [
                {"asin": "a", "image_path": "images/a.jpg"},
                {"asin": "b", "image_path": "images/b.jpg"},
            ]
        ),
        encoding="utf-8",
    )
    vectors = np.eye(2, 3, dtype=np.float32)
    np.save(root / "search.npy", vectors)
    np.save(root / "content.npy", vectors)
    (root / "search.index").write_bytes(b"search index placeholder")
    (root / "content.index").write_bytes(b"content index placeholder")
    manifest = ArtifactManifest(
        schema_version=2,
        dataset_sha256=sha256_file(metadata_path),
        model_name="test",
        pretrained="test",
        embedding_dimension=3,
        item_count=2,
        metadata_path="items.json",
        search_index_path="search.index",
        search_embeddings_path="search.npy",
        content_index_path="content.index",
        content_embeddings_path="content.npy",
        search_index_sha256=sha256_file(root / "search.index"),
        search_embeddings_sha256=sha256_file(root / "search.npy"),
        content_index_sha256=sha256_file(root / "content.index"),
        content_embeddings_sha256=sha256_file(root / "content.npy"),
    )
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(asdict(manifest)), encoding="utf-8")
    return manifest_path


class CatalogTests(unittest.TestCase):
    def test_image_path_stays_inside_data_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, relative = resolve_image_path("images/a.jpg", root)
            self.assertEqual(path, root / "images" / "a.jpg")
            self.assertEqual(relative, "images/a.jpg")
            with self.assertRaisesRegex(ValueError, "outside the data directory"):
                resolve_image_path("../elsewhere/a.jpg", root)

    def test_catalog_rejects_duplicate_ids_and_nonportable_paths(self):
        with self.assertRaisesRegex(ValueError, "Duplicate ASIN"):
            validate_catalog_metadata(
                [
                    {"asin": "a", "image_path": "images/a.jpg"},
                    {"asin": "a", "image_path": "images/b.jpg"},
                ]
            )
        with self.assertRaisesRegex(ValueError, "absolute image path"):
            validate_catalog_metadata([{"asin": "a", "image_path": "C:\\images\\a.jpg"}])
        with self.assertRaisesRegex(ValueError, "unsafe image path"):
            validate_catalog_metadata([{"asin": "a", "image_path": "../a.jpg"}])

    def test_vectors_reject_zero_and_nonfinite_rows(self):
        for bad_row in ([0.0, 0.0], [np.nan, 0.0]):
            with self.subTest(bad_row=bad_row):
                with self.assertRaises(ValueError):
                    validate_normalized_vectors(
                        np.asarray([[1.0, 0.0], bad_row], dtype=np.float32), "test"
                    )

    def test_content_builder_stops_on_missing_image(self):
        script = Path(__file__).resolve().parents[1] / "SIM" / "compu.py"
        fake_torch = ModuleType("torch")
        fake_torch.cuda = SimpleNamespace(is_available=lambda: False)
        with patch.dict(
            sys.modules,
            {"faiss": ModuleType("faiss"), "open_clip": ModuleType("open_clip"), "torch": fake_torch},
        ):
            spec = importlib.util.spec_from_file_location("content_builder_test", script)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            module.tqdm = lambda items, **kwargs: items
            with tempfile.TemporaryDirectory() as directory:
                builder = module.EmbeddingBuilder.__new__(module.EmbeddingBuilder)
                builder.DATA_DIR = Path(directory)
                builder.items_meta = [{"asin": "a", "image_path": "images/missing.jpg"}]
                with self.assertRaisesRegex(RuntimeError, "image for a"):
                    builder.compute_embeddings()


class ArtifactManifestTests(unittest.TestCase):
    def test_full_audit_requires_faiss(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = make_artifacts(Path(directory))
            with patch("fyp.artifacts.importlib.util.find_spec", return_value=None):
                with self.assertRaisesRegex(RuntimeError, "faiss is required"):
                    ArtifactManifest.load(manifest_path)

    def test_old_manifest_reports_rebuild_instruction(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = make_artifacts(Path(directory))
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload["schema_version"] = 1
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "rebuild artifacts"):
                ArtifactManifest.load(manifest_path)

    def test_manifest_validates_counts_shape_and_checksums(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = make_artifacts(root)
            with patch("fyp.artifacts.importlib.util.find_spec", return_value=None):
                manifest = ArtifactManifest.load(manifest_path, verify_index_rows=False)
            self.assertEqual(manifest.item_count, 2)

            (root / "search.index").write_bytes(b"different index")
            with patch("fyp.artifacts.importlib.util.find_spec", return_value=None):
                with self.assertRaisesRegex(ValueError, "search index checksum"):
                    ArtifactManifest.load(manifest_path, verify_index_rows=False)

    def test_manifest_rejects_zero_content_vector_even_with_matching_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = make_artifacts(root)
            np.save(root / "content.npy", np.zeros((2, 3), dtype=np.float32))
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload["content_embeddings_sha256"] = sha256_file(root / "content.npy")
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")
            with patch("fyp.artifacts.importlib.util.find_spec", return_value=None):
                with self.assertRaisesRegex(ValueError, "zero vectors"):
                    ArtifactManifest.load(manifest_path, verify_index_rows=False)

    def test_manifest_rejects_same_size_index_with_wrong_row_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = make_artifacts(root)
            swapped = np.eye(2, 3, dtype=np.float32)[::-1].copy()
            fake_index = SimpleNamespace(
                ntotal=2,
                d=3,
                reconstruct_n=lambda start, count: swapped[start : start + count],
            )
            fake_faiss = SimpleNamespace(read_index=lambda path: fake_index)
            with patch("fyp.artifacts.importlib.util.find_spec", return_value=True):
                with patch.dict("sys.modules", {"faiss": fake_faiss}):
                    with self.assertRaisesRegex(ValueError, "row order differs"):
                        ArtifactManifest.load(manifest_path)


if __name__ == "__main__":
    unittest.main()
