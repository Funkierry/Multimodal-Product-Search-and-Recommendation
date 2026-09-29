import contextlib
import io
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fyp.evaluation.cli import search_main
from fyp.evaluation.metrics import ranking_metrics
from fyp.evaluation.recommendation import evaluate_recommendations, leave_one_out
from fyp.evaluation.search import evaluate_search, matches_filters


class RankingMetricTests(unittest.TestCase):
    def test_recall_mrr_and_ndcg_use_judged_relevance(self):
        result = ranking_metrics(["b", "a", "c"], {"a", "c"}, cutoffs=(1, 2))
        self.assertEqual(result["recall@1"], 0.0)
        self.assertEqual(result["recall@2"], 0.5)
        self.assertEqual(result["mrr"], 0.5)
        ideal = 1 + 1 / math.log2(3)
        self.assertAlmostEqual(result["ndcg@2"], (1 / math.log2(3)) / ideal)

    def test_search_matches_query_ids_and_reports_latency_by_segment(self):
        judgments = [
            {"query_id": "q1", "query": "red shoes", "relevant_asins": ["a"]},
            {"query_id": "q2", "query": "blue shoes", "relevant_asins": ["b"],
             "segment": "filtered"},
        ]
        rankings = [
            {"query_id": "q2", "asins": ["b"], "latency_ms": 30},
            {"query_id": "q1", "asins": ["x", "a"], "latency_ms": 10},
        ]
        result = evaluate_search(judgments, rankings)
        self.assertEqual(result["query_count"], 2)
        self.assertEqual(result["metrics"]["mrr"], 0.75)
        self.assertEqual(result["latency_ms"]["p50"], 20)
        self.assertEqual(result["segments"]["filtered"]["query_count"], 1)
        with self.assertRaisesRegex(ValueError, "same query IDs"):
            evaluate_search(judgments, rankings[:1])

    def test_filtered_search_respects_category_and_price(self):
        item = {"categoryName": "Shoes", "price": "£20.00"}
        self.assertTrue(matches_filters(item, {"category": "shoes", "max_price": 25}))
        self.assertFalse(matches_filters(item, {"min_price": 30}))
        self.assertFalse(matches_filters({"categoryName": "Shoes", "price": ""},
                                        {"max_price": 25}))

    def test_search_cli_scores_precomputed_rankings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            judgments = root / "judgments.jsonl"
            rankings = root / "rankings.jsonl"
            output = root / "report.json"
            judgments.write_text(json.dumps({
                "query_id": "q1", "query": "red shoes", "relevant_asins": ["a"]
            }) + "\n", encoding="utf-8")
            rankings.write_text(json.dumps({
                "query_id": "q1", "asins": ["b", "a"], "latency_ms": 12
            }) + "\n", encoding="utf-8")
            argv = ["fyp-evaluate-search", "--judgments", str(judgments),
                    "--rankings", str(rankings), "--output", str(output)]
            with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                search_main()
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["metrics"]["mrr"], 0.5)
            self.assertEqual(report["segments"]["unfiltered"]["query_count"], 1)


class RecommendationEvaluationTests(unittest.TestCase):
    records = [
        {"user": "u1", "ASIN": "a"}, {"user": "u1", "ASIN": "b"},
        {"user": "u1", "ASIN": "c"}, {"user": "u2", "ASIN": "a"},
        {"user": "u2", "ASIN": "b"}, {"user": "u3", "ASIN": "a"},
        {"user": "u3", "ASIN": "c"}, {"user": "u4", "ASIN": "b"},
        {"user": "u4", "ASIN": "c"},
    ]

    def test_holdout_is_stable_disjoint_and_warm(self):
        training, held_out, _ = leave_one_out(self.records)
        reverse_training, reverse_held_out, _ = leave_one_out(reversed(self.records))
        self.assertEqual(training, reverse_training)
        self.assertEqual(held_out, reverse_held_out)
        self.assertTrue(held_out)
        for user, target in held_out.items():
            self.assertNotIn(target, training[user])
            self.assertTrue(any(target in items for items in training.values()))

    def test_report_contains_baselines_and_split_identity(self):
        categories = {"a": "Shoes", "b": "Shoes", "c": "Clothes"}
        report = evaluate_recommendations(self.records, categories)
        self.assertGreater(report["evaluated_users"], 0)
        self.assertEqual(
            set(report["methods"]), {"co_purchase", "popularity", "same_category"}
        )
        self.assertEqual(len(report["split_sha256"]), 64)
        for method in report["methods"].values():
            self.assertGreaterEqual(method["metrics"]["recall@5"], 0.0)
            self.assertLessEqual(method["metrics"]["recall@5"], 1.0)


if __name__ == "__main__":
    unittest.main()
