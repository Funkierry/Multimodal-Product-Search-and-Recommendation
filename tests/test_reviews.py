import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from fyp.reviews import build_review_summaries, load_review_summaries


class FakeAnalyzer:
    def polarity_scores(self, review):
        return {"compound": 1 if "good" in review else -1}


def fake_nlp(review):
    return [
        SimpleNamespace(lemma_=word, pos_="ADJ", is_stop=False)
        for word in review.split()
        if word in {"soft", "scratchy"}
    ]


class ReviewSummaryTests(unittest.TestCase):
    def test_offline_builder_deduplicates_reviews_and_keeps_runtime_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "reviews.csv"
            output_path = Path(directory) / "reviews.json"
            csv_path.write_text(
                "asin,USER,review\n"
                "item-1,u1,good soft\n"
                "item-1,u1,good soft\n"
                "item-1,u2,bad scratchy\n",
                encoding="utf-8",
            )
            summaries = build_review_summaries(
                csv_path, output_path, nlp=fake_nlp, analyzer=FakeAnalyzer()
            )
            self.assertEqual(load_review_summaries(output_path), summaries)
            self.assertEqual(summaries["item-1"]["aspects"], ["soft", "scratchy"])
            self.assertEqual(summaries["item-1"]["positiveScores"], [1, 0])
            self.assertEqual(summaries["item-1"]["negativeScores"], [0, 1])

    def test_invalid_source_fails_before_writing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "reviews.csv"
            output_path = Path(directory) / "reviews.json"
            csv_path.write_text("ASIN,Review\nitem-1,good soft\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing columns"):
                build_review_summaries(
                    csv_path, output_path, nlp=fake_nlp, analyzer=FakeAnalyzer()
                )
            self.assertFalse(output_path.exists())


if __name__ == "__main__":
    unittest.main()
