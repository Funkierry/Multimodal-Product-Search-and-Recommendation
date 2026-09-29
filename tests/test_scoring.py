import unittest

from fyp.scoring import filter_results, similarity_percent, sort_by_similarity


class ScoringTests(unittest.TestCase):
    def test_highest_similarity_is_first(self):
        results = [({"asin": "low"}, 0.1), ({"asin": "high"}, 0.8)]
        self.assertEqual(sort_by_similarity(results)[0][0]["asin"], "high")

    def test_display_score_increases_with_similarity(self):
        self.assertLess(similarity_percent(0.2), similarity_percent(0.8))
        self.assertEqual(similarity_percent(-2), 0)
        self.assertEqual(similarity_percent(2), 100)

    def test_filter_keeps_similarity_order(self):
        results = [
            ({"asin": "a", "price": 20, "categoryName": "Shirt"}, 0.3),
            ({"asin": "b", "price": 25, "categoryName": "Shirt"}, 0.7),
            ({"asin": "c", "price": 25, "categoryName": "Boots"}, 0.9),
        ]
        filtered = filter_results(results, min_price=10, max_price=30, category="shirt")
        self.assertEqual([item["asin"] for item, _ in filtered], ["b", "a"])


if __name__ == "__main__":
    unittest.main()
