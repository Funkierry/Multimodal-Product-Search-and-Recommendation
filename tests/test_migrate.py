import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from fyp.migrate import migrate_artifacts


class MigrationTests(unittest.TestCase):
    def make_legacy_files(self, root: Path):
        data_dir = root / "dataset"
        search_dir = root / "Search"
        content_dir = root / "SIM"
        (data_dir / "images").mkdir(parents=True)
        search_dir.mkdir()
        content_dir.mkdir()
        items = []
        for asin in ("a", "b"):
            image_path = data_dir / "images" / f"{asin}.jpg"
            Image.new("RGB", (2, 2)).save(image_path)
            items.append({"asin": asin, "image_path": str(image_path)})
        (search_dir / "items_meta.json").write_text(json.dumps(items), encoding="utf-8")
        (search_dir / "faiss_index.index").write_bytes(b"search")
        (content_dir / "faiss_index_sim.index").write_bytes(b"content")
        vectors = np.eye(2, 3, dtype=np.float32)
        np.save(content_dir / "embeddings_sim.npy", vectors)
        return data_dir, search_dir, content_dir, vectors

    def test_migration_preserves_backup_and_publishes_verified_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            data_dir, search_dir, content_dir, vectors = self.make_legacy_files(Path(directory))
            fake_index = SimpleNamespace(
                ntotal=2,
                d=3,
                reconstruct_n=lambda start, count: vectors[start : start + count],
            )
            fake_faiss = ModuleType("faiss")
            fake_faiss.read_index = lambda path: fake_index
            with patch.dict(sys.modules, {"faiss": fake_faiss}):
                with patch("fyp.artifacts.importlib.util.find_spec", return_value=True):
                    manifest_path = migrate_artifacts(data_dir, search_dir, content_dir)

            self.assertTrue(manifest_path.is_file())
            self.assertTrue((search_dir / "items_meta.legacy.json").is_file())
            updated = json.loads((search_dir / "items_meta.json").read_text(encoding="utf-8"))
            self.assertEqual([item["image_path"] for item in updated], ["images/a.jpg", "images/b.jpg"])
            np.testing.assert_array_equal(np.load(search_dir / "image_embeddings.npy"), vectors)

    def test_migration_rejects_misaligned_content_before_changing_catalog(self):
        with tempfile.TemporaryDirectory() as directory:
            data_dir, search_dir, content_dir, vectors = self.make_legacy_files(Path(directory))
            fake_index = SimpleNamespace(
                ntotal=2,
                d=3,
                reconstruct_n=lambda start, count: vectors[::-1][start : start + count],
            )
            fake_faiss = ModuleType("faiss")
            fake_faiss.read_index = lambda path: fake_index
            with patch.dict(sys.modules, {"faiss": fake_faiss}):
                with self.assertRaisesRegex(ValueError, "row order differs"):
                    migrate_artifacts(data_dir, search_dir, content_dir)

            self.assertFalse((search_dir / "items_meta.legacy.json").exists())
            original = json.loads((search_dir / "items_meta.json").read_text(encoding="utf-8"))
            self.assertTrue(Path(original[0]["image_path"]).is_absolute())


if __name__ == "__main__":
    unittest.main()
