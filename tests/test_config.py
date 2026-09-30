import os
import unittest
from unittest.mock import patch

from fyp.config import faiss_search_threads


class FaissThreadConfigTests(unittest.TestCase):
    def test_defaults_to_measured_laptop_setting_and_accepts_override(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(faiss_search_threads(), 4)
        with patch.dict(os.environ, {"FYP_FAISS_THREADS": "2"}):
            self.assertEqual(faiss_search_threads(), 2)

    def test_rejects_invalid_thread_count(self):
        for value in ("0", "many"):
            with self.subTest(value=value), patch.dict(os.environ, {"FYP_FAISS_THREADS": value}):
                with self.assertRaises(ValueError):
                    faiss_search_threads()


if __name__ == "__main__":
    unittest.main()
