import unittest

import pandas as pd

from SIM.classification_audit import audit_frames


def row(asin, category):
    return {"asin": asin, "title": asin, "imgUrl": f"{asin}.jpg", "categoryName": category}


class ClassificationAuditTests(unittest.TestCase):
    def test_unseen_classes_are_counted_without_discarding_test_rows(self):
        frames = {
            "train": pd.DataFrame([row("a", "known"), row("b", None)]),
            "val": pd.DataFrame([row("c", "known"), row("d", "new")]),
            "test": pd.DataFrame([row("e", "new"), row("f", "known")]),
        }
        report = audit_frames(frames)
        self.assertEqual(report["splits"]["train"]["missing_category_rows"], 1)
        self.assertEqual(report["splits"]["test"]["rows"], 2)
        self.assertEqual(report["splits"]["test"]["unseen_class_rows"], 1)
        self.assertEqual(report["splits"]["test"]["known_class_coverage"], 0.5)

    def test_rejects_product_leakage_across_splits(self):
        frames = {
            "train": pd.DataFrame([row("a", "known")]),
            "val": pd.DataFrame([row("b", "known")]),
            "test": pd.DataFrame([row("a", "known")]),
        }
        with self.assertRaisesRegex(ValueError, "train and test share 1 ASIN"):
            audit_frames(frames)


if __name__ == "__main__":
    unittest.main()
