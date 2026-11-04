import os
import unittest
from pathlib import Path
from unittest import mock

from src.chunking import split_documents
from src.chunking.metadata import context_header, time_periods, value_types
from src.retrieval import matches_structure, parse_query, soft_score

PDF_PATH = Path(__file__).resolve().parents[1] / "data" / "KinhteVN.pdf"


def _page(segments: list[dict] | None, images: list[dict] | None = None) -> dict:
    record = {
        "document_id": "doc",
        "source": "doc.pdf",
        "page": 3,
        "text": "page text that the structured model must not chunk",
        "images": images or [],
        "metadata": {},
    }
    if segments is not None:
        record["segments"] = segments
    return record


class ChunkMetadataTest(unittest.TestCase):
    def test_time_periods_are_normalized(self) -> None:
        text = "Lạm phát tháng 4/2026 tăng; quý I/2026 thâm hụt; năm 2025 tăng 8%; trong tháng 3."
        self.assertEqual(time_periods(text), ["2025", "2026", "2026-04", "2026-Q1", "M03"])

    def test_forecast_and_estimate_columns_are_value_types(self) -> None:
        self.assertEqual(value_types("2024 2025e 2026f 2027f"), ["forecast", "estimate"])
        self.assertEqual(value_types("Xuất khẩu tăng 16% trong năm 2025."), [])

    def test_header_keeps_chapter_and_section_only(self) -> None:
        header = context_header(
            {
                "heading_path": ["B. Diễn biến", "Giá cả", "Một câu tiêu đề rất dài"],
                "box_id": "2",
                "box_title": "Tín dụng",
                "figure_no": "B2.1",
            }
        )
        self.assertEqual(header, "Mục: B. Diễn biến > Giá cả | Hộp 2: Tín dụng | Hình B2.1")


class QueryMetadataTest(unittest.TestCase):
    def test_structure_references_are_parsed(self) -> None:
        query = parse_query("Hình b2.1 trong Hộp 2 và Bảng 3 nói gì?")
        self.assertEqual((query.figure_no, query.box_id, query.table_no), ("B2.1", "2", "3"))

    def test_tinh_hinh_is_not_a_figure_reference(self) -> None:
        query = parse_query("Tình hình kinh tế Việt Nam năm 2026 thế nào?")
        self.assertFalse(query.has_structure)
        self.assertEqual(query.time_periods, ("2026",))

    def test_body_text_citing_a_figure_matches(self) -> None:
        query = parse_query("Hình 16 thể hiện gì?")
        self.assertTrue(matches_structure({"figure_no": "16"}, query))
        self.assertTrue(matches_structure({"figure_refs": ["16", "17"]}, query))
        self.assertFalse(matches_structure({"figure_no": "160", "figure_refs": ["1"]}, query))

    def test_month_without_year_matches_month_with_year(self) -> None:
        query = parse_query("Lạm phát tháng 4/2026 là bao nhiêu?")
        self.assertGreater(soft_score({"time_periods": ["M04"]}, query), 0)
        self.assertEqual(soft_score({"time_periods": ["2025-04"]}, query), 0)

    def test_source_request_prefers_chunks_with_a_source(self) -> None:
        query = parse_query("Số liệu PMI lấy từ nguồn nào?")
        self.assertGreater(soft_score({"data_source": "Haver Analytics"}, query), soft_score({}, query))


class SegmentChunkingTest(unittest.TestCase):
    def test_figure_is_one_chunk_with_its_own_image(self) -> None:
        figure = {
            "element_type": "figure",
            "text": "Hình 16: Chỉ số PMI\nNguồn: Haver Analytics.",
            "heading_path": ["B", "Sản xuất"],
            "figure_no": "16",
            "data_source": "Haver Analytics",
            "image_indexes": [1],
        }
        images = [
            {"document_id": "doc", "page_number": 3, "image_index": 0, "image_path": "a.png"},
            {"document_id": "doc", "page_number": 3, "image_index": 1, "image_path": "b.png"},
        ]
        chunks = split_documents([_page([figure], images)], chunk_size=5, chunk_overlap=0, min_chunk_size=1)

        self.assertEqual(len(chunks), 1)
        chunk = chunks[0]
        self.assertEqual(chunk["figure_no"], "16")
        self.assertEqual(chunk["data_source"], "Haver Analytics")
        self.assertEqual([ref["image_path"] for ref in chunk["image_refs"]], ["b.png"])
        self.assertEqual(chunk["context_header"], "Mục: B > Sản xuất | Hình 16")

    def test_empty_segment_list_gives_no_chunks(self) -> None:
        self.assertEqual(split_documents([_page([])]), [])

    def test_page_model_still_chunks_page_text(self) -> None:
        chunks = split_documents([_page(None)], chunk_size=50, chunk_overlap=0, min_chunk_size=1)
        self.assertEqual(len(chunks), 1)
        self.assertNotIn("element_type", chunks[0])

    def test_chunk_indexes_continue_across_segments_of_a_page(self) -> None:
        body = {"element_type": "body", "text": "Đoạn một.", "heading_path": ["A"]}
        table = {"element_type": "table", "text": "Bảng 3: GDP 6.8", "table_no": "3"}
        chunks = split_documents([_page([body, table])], chunk_size=50, chunk_overlap=0, min_chunk_size=1)
        self.assertEqual([chunk["page_chunk_index"] for chunk in chunks], [0, 1])
        self.assertEqual(len({chunk["chunk_id"] for chunk in chunks}), 2)


@unittest.skipUnless(PDF_PATH.is_file(), "data/KinhteVN.pdf is not available")
class KinhteVNStructureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from src.documents import load_pdf_document

        with mock.patch.dict(os.environ, {"DOCUMENT_IMAGE_EXPORT": "false", "DOCUMENT_IMAGE_OCR": "false"}):
            cls.document = load_pdf_document(PDF_PATH, image_output_dir=None, document_model="structured")
        cls.segments = {page.page_number: page.segments for page in cls.document.pages}

    def find(self, page: int, **fields: str) -> dict:
        for segment in self.segments[page]:
            if all(segment.get(key) == value for key, value in fields.items()):
                return segment
        self.fail(f"No segment {fields} on page {page}")

    def test_cover_gives_document_info(self) -> None:
        self.assertEqual(self.document.metadata["cover_subtitle"], "Kiên định Cải cách, Ứng phó với Biến động")
        info = self.find(1, element_type="document_info")
        self.assertIn("Tháng 5/2026", info["text"])

    def test_figure_has_caption_source_and_image(self) -> None:
        figure = self.find(16, element_type="figure", figure_no="16")
        self.assertIn("PMI", figure["caption"])
        self.assertEqual(figure["data_source"], "Haver Analytics")
        self.assertEqual(len(figure["image_indexes"]), 1)

    def test_box_starts_at_its_label_not_at_the_page_top(self) -> None:
        box_text = " ".join(s["text"] for s in self.segments[30] if s.get("box_id") == "3")
        other_text = " ".join(s["text"] for s in self.segments[30] if not s.get("box_id"))
        self.assertIn("hai phần ba", box_text)
        self.assertIn("Rủi ro suy giảm trong ngắn hạn", other_text)

    def test_table_keeps_forecast_columns(self) -> None:
        table = self.find(30, element_type="table", table_no="3")
        self.assertIn("2024 2025e 2026f 2027f 2028f", table["text"])
        self.assertIn("Tăng trưởng GDP (%) 7.0 8.0 6.8 7.1 7.4", table["text"])

    def test_table_of_contents_is_not_indexed(self) -> None:
        self.assertEqual(self.segments[6], [])
        self.assertEqual(self.segments[7], [])

    def test_body_text_has_no_chart_captions(self) -> None:
        body = " ".join(s["text"] for s in self.segments[16] if s["element_type"] == "body")
        self.assertNotIn("Nguồn: Haver Analytics", body)
