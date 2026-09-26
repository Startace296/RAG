import unittest

from src.benchmarks.metrics import evaluate_results, is_relevant


def _result(page: int, text: str = "") -> dict:
    return {"page": page, "text": text}


class RelevanceTest(unittest.TestCase):
    def test_page_and_keyword_both_required_when_both_labeled(self) -> None:
        question = {"expected_pages": [3], "expected_keywords": ["lạm phát"]}
        self.assertTrue(is_relevant(_result(3, "Lạm phát tăng"), question))
        self.assertFalse(is_relevant(_result(3, "Xuất khẩu tăng"), question))
        self.assertFalse(is_relevant(_result(9, "Lạm phát tăng"), question))

    def test_only_pages_labeled(self) -> None:
        question = {"expected_pages": [3]}
        self.assertTrue(is_relevant(_result(3), question))
        self.assertFalse(is_relevant(_result(4), question))

    def test_only_keywords_labeled(self) -> None:
        question = {"expected_keywords": ["FDI"]}
        self.assertTrue(is_relevant(_result(1, "vốn fdi"), question))
        self.assertFalse(is_relevant(_result(1, "vốn ODA"), question))

    def test_keyword_in_parent_text_counts(self) -> None:
        question = {"expected_pages": [2], "expected_keywords": ["GDP"]}
        result = {"page": 2, "text": "child", "parent_text": "GDP tăng 6%"}
        self.assertTrue(is_relevant(result, question))


class EvaluateResultsTest(unittest.TestCase):
    def test_ndcg_penalizes_missing_labeled_pages(self) -> None:
        question = {"expected_pages": [1, 2, 3]}
        results = [_result(1), _result(9), _result(9)]
        metrics = evaluate_results(results, question, top_k=3)

        self.assertEqual(metrics["hit"], 1.0)
        self.assertAlmostEqual(metrics["recall"], 1 / 3)
        self.assertEqual(metrics["mrr"], 1.0)
        # One relevant result at rank 1 out of three labeled pages.
        self.assertLess(metrics["ndcg"], 1.0)

    def test_ndcg_never_exceeds_one(self) -> None:
        question = {"expected_pages": [1]}
        results = [_result(1), _result(1), _result(1)]
        metrics = evaluate_results(results, question, top_k=3)
        self.assertAlmostEqual(metrics["ndcg"], 1.0)

    def test_recall_ignores_page_matches_without_keyword(self) -> None:
        question = {"expected_pages": [1, 2], "expected_keywords": ["nợ công"]}
        results = [_result(1, "nợ công"), _result(2, "chủ đề khác")]
        metrics = evaluate_results(results, question, top_k=2)
        self.assertAlmostEqual(metrics["recall"], 0.5)
        self.assertAlmostEqual(metrics["precision"], 0.5)

    def test_no_relevant_results(self) -> None:
        metrics = evaluate_results([_result(5)], {"expected_pages": [1]}, top_k=5)
        self.assertEqual(metrics, {"hit": 0.0, "precision": 0.0, "recall": 0.0, "mrr": 0.0, "ndcg": 0.0})


if __name__ == "__main__":
    unittest.main()
