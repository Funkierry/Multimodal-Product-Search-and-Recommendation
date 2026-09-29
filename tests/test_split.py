import unittest

from fyp.data.split import split_records


class SplitTests(unittest.TestCase):
    def test_split_is_stable_and_disjoint_by_key(self):
        records = [{"asin": f"item-{index}"} for index in range(1000)]
        first = split_records(records, seed=7)
        second = split_records(reversed(records), seed=7)
        first_sets = {name: {row["asin"] for row in rows} for name, rows in first.items()}
        second_sets = {name: {row["asin"] for row in rows} for name, rows in second.items()}
        self.assertEqual(first_sets, second_sets)
        self.assertFalse(first_sets["train"] & first_sets["validation"])
        self.assertFalse(first_sets["train"] & first_sets["test"])
        self.assertFalse(first_sets["validation"] & first_sets["test"])


if __name__ == "__main__":
    unittest.main()
