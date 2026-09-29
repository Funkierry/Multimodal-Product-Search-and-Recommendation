import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from fyp.artifacts import ArtifactManifest, sha256_file


class ArtifactManifestTests(unittest.TestCase):
    def test_manifest_validates_counts_shape_and_checksum(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata_path = root / "items.json"
            metadata_path.write_text(json.dumps([{"asin": "a"}, {"asin": "b"}]))
            np.save(root / "embeddings.npy", np.zeros((2, 3), dtype=np.float32))
            (root / "search.index").write_bytes(b"placeholder")
            (root / "content.index").write_bytes(b"placeholder")
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "dataset_sha256": sha256_file(metadata_path),
                        "model_name": "test",
                        "pretrained": "test",
                        "embedding_dimension": 3,
                        "item_count": 2,
                        "metadata_path": "items.json",
                        "search_index_path": "search.index",
                        "content_index_path": "content.index",
                        "content_embeddings_path": "embeddings.npy",
                    }
                )
            )
            with patch("fyp.artifacts.importlib.util.find_spec", return_value=None):
                manifest = ArtifactManifest.load(manifest_path)
            self.assertEqual(manifest.item_count, 2)


if __name__ == "__main__":
    unittest.main()
