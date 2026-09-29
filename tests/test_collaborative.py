import unittest

from fyp.recommendation.collaborative import InteractionRecommender


class CollaborativeTests(unittest.TestCase):
    def test_recommendations_use_normalized_co_purchase(self):
        records = [
            {"user": "u1", "ASIN": "target"},
            {"user": "u1", "ASIN": "a"},
            {"user": "u2", "ASIN": "target"},
            {"user": "u2", "ASIN": "a"},
            {"user": "u2", "ASIN": "b"},
            {"user": "u3", "ASIN": "b"},
        ]
        recommender = InteractionRecommender.from_records(records)
        recommendations = recommender.recommend("target", top_n=2)
        self.assertEqual([item for item, _ in recommendations], ["a", "b"])
        self.assertGreater(recommendations[0][1], recommendations[1][1])


if __name__ == "__main__":
    unittest.main()
