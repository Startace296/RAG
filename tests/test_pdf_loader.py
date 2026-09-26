import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pymupdf

from src.documents import load_pdf_document


def _png(side: int) -> bytes:
    pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, side, side), False)
    pixmap.clear_with(200)
    return pixmap.tobytes("png")


class PdfImageFilterTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.pdf_path = self.root / "sample.pdf"

        pdf = pymupdf.open()
        page = pdf.new_page()
        page.insert_text((72, 72), "Trang có ảnh")
        page.insert_image(pymupdf.Rect(72, 100, 172, 200), stream=_png(64))
        page.insert_image(pymupdf.Rect(200, 100, 202, 102), stream=_png(2))
        pdf.save(self.pdf_path)
        pdf.close()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_tiny_images_are_skipped_and_stale_files_removed(self) -> None:
        image_dir = self.root / "images"
        env = {"DOCUMENT_IMAGE_EXPORT": "true", "DOCUMENT_IMAGE_OCR": "false", "IMAGE_MIN_SIDE": "32"}
        with mock.patch.dict(os.environ, env):
            first = load_pdf_document(self.pdf_path, image_output_dir=image_dir)
            document_dir = image_dir / first.document_id
            stale = document_dir / "page_0009_image_0000_xref_999.png"
            stale.write_bytes(b"old")

            document = load_pdf_document(self.pdf_path, image_output_dir=image_dir)

        page = document.pages[0]
        self.assertEqual(len(page.images), 1)
        self.assertEqual(page.metadata["skipped_image_count"], 1)
        self.assertEqual(page.images[0].metadata["width"], 64)
        self.assertFalse(stale.exists())
        self.assertEqual(len(list(document_dir.iterdir())), 1)


if __name__ == "__main__":
    unittest.main()
