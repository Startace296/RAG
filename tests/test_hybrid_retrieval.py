import unittest
from unittest import mock

import numpy as np

from src.app.rag_service import RagService
from src.documents.pdf_loader import (
    _continues_heading,
    _find_repeated_lines,
    _is_heading_candidate,
    _strip_boilerplate,
)
from src.retrieval import BM25Index, CrossEncoderReranker, VectorStore

HEADER = "Cập nhật Tình hình Kinh tế Việt Nam • Tháng 5/2026"


class BM25IndexTest(unittest.TestCase):
    def test_exact_terms_rank_first(self) -> None:
        index = BM25Index(
            [
                "Tăng trưởng GDP đạt 8% trong năm 2025.",
                "Giá năng lượng toàn cầu tăng thêm hơn 50% tính đến cuối tháng 4.",
                "Tín dụng ngân hàng tăng trưởng ổn định.",
            ]
        )
        ranks = index.ranks("Giá năng lượng toàn cầu tăng bao nhiêu đến tháng 4?")
        self.assertEqual(int(ranks[1]), 1)

    def test_unknown_terms_score_zero(self) -> None:
        index = BM25Index(["một hai ba", "bốn năm sáu"])
        self.assertTrue(np.all(index.scores("xyz") == 0))


class BoilerplateTest(unittest.TestCase):
    def test_running_header_and_page_numbers_are_removed(self) -> None:
        pages = [f"{number}\n{HEADER}\nNội dung trang {number}" for number in range(1, 6)]
        repeated = _find_repeated_lines(pages)
        self.assertEqual(repeated, {HEADER})

        text, removed = _strip_boilerplate(pages[2], repeated)
        self.assertEqual(text, "Nội dung trang 3")
        self.assertEqual(removed, 2)

    def test_repeated_line_inside_page_is_kept(self) -> None:
        source = "Nguồn: NSO và tính toán của NHTG."
        page = "\n".join(["Hình 1", source, "Hình 2", source, "Hình 3", source, "Kết luận"])
        text, removed = _strip_boilerplate(page, set())
        self.assertEqual(text.count(source), 3)
        self.assertEqual(removed, 0)

    def test_stamp_at_page_edge_is_removed(self) -> None:
        stamp = "Public Disclosure Authorized"
        text, _ = _strip_boilerplate("\n".join(["Tiêu đề", stamp, stamp, stamp]), set())
        self.assertEqual(text, "Tiêu đề")

    def test_header_with_glued_page_number_is_removed(self) -> None:
        pages = [f"{number} {HEADER}\nNội dung trang {number}" for number in range(1, 6)]
        repeated = _find_repeated_lines(pages)
        self.assertEqual(repeated, {HEADER})
        text, _ = _strip_boilerplate(pages[3], repeated)
        self.assertEqual(text, "Nội dung trang 4")

    def test_paragraphs_are_kept_and_line_wraps_joined(self) -> None:
        wrapped = "Dòng thứ nhất của đoạn văn dài bị PDF ngắt ở lề"
        page = f"Tiêu đề\n\n{wrapped}\ngiữa câu.\n\nĐoạn thứ hai."
        text, _ = _strip_boilerplate(page, set())
        self.assertEqual(text, f"Tiêu đề\n\n{wrapped} giữa câu.\n\nĐoạn thứ hai.")

    def test_short_title_lines_keep_their_breaks(self) -> None:
        page = "Cập nhật Tình hình Kinh tế\nViệt Nam\nKiên định Cải cách,\nỨng phó với Biến động"
        text, _ = _strip_boilerplate(page, set())
        self.assertEqual(text, page)


class HeadingTest(unittest.TestCase):
    def test_chart_labels_are_not_headings(self) -> None:
        self.assertFalse(_is_heading_candidate("2024 2025e 2026f 2027f 2028f"))
        self.assertFalse(_is_heading_candidate("Nguồn: NSO và tính toán của NHTG."))
        self.assertFalse(_is_heading_candidate("HÌNH"))
        self.assertTrue(_is_heading_candidate("Tài chính công"))

    def test_wrapped_heading_is_continued(self) -> None:
        self.assertTrue(_continues_heading("B. Diễn biến kinh tế", "gần đây"))
        self.assertTrue(_continues_heading("C. Triển vọng, Rủi ro và", "Thách thức"))
        self.assertFalse(_continues_heading("A. Tình hình thế giới", "Bối cảnh toàn cầu"))

class HybridFusionTest(unittest.TestCase):
    def _service(self) -> RagService:
        service = RagService.__new__(RagService)
        service.hybrid_rrf_k = 60
        service.store = mock.Mock(spec=VectorStore)
        return service

    def test_keyword_match_moves_dense_candidate_up(self) -> None:
        service = self._service()
        # Candidate at position 7 is third by dense score but first by BM25.
        service.store.keyword_ranks.return_value = np.array([40, 50, 30, 20, 10, 9, 8, 1])
        results = [
            {"position": 0, "similarity_score": 0.90},
            {"position": 1, "similarity_score": 0.89},
            {"position": 7, "similarity_score": 0.88},
        ]
        fused = service._fuse_keyword_ranks("câu hỏi", results)
        self.assertEqual([result["position"] for result in fused], [7, 0, 1])
        self.assertEqual(fused[0]["keyword_rank"], 1)


class CrossEncoderRerankTest(unittest.TestCase):
    def _service(self, scores: list[float]) -> RagService:
        service = RagService.__new__(RagService)
        service.reranker = mock.Mock(spec=CrossEncoderReranker)
        service.reranker.scores.return_value = np.array(scores, dtype="float32")
        return service

    def test_cross_encoder_score_sets_final_order(self) -> None:
        service = self._service([0.1, 0.9, 0.5])
        results = [
            {"position": 0, "text": "a"},
            {"position": 1, "text": "b"},
            {"position": 2, "text": "c"},
        ]
        reranked = service._rerank("câu hỏi", results)
        self.assertEqual([result["position"] for result in reranked], [1, 2, 0])
        self.assertAlmostEqual(reranked[0]["rerank_score"], 0.9, places=5)

    def test_parent_text_is_scored(self) -> None:
        service = self._service([0.0])
        service._rerank("câu hỏi", [{"position": 0, "text": "con", "parent_text": "cha"}])
        service.reranker.scores.assert_called_once_with("câu hỏi", ["cha"])

    def test_ties_keep_previous_order(self) -> None:
        service = self._service([0.3, 0.3])
        results = [{"position": 4, "text": "a"}, {"position": 2, "text": "b"}]
        reranked = service._rerank("câu hỏi", results)
        self.assertEqual([result["position"] for result in reranked], [4, 2])


if __name__ == "__main__":
    unittest.main()
