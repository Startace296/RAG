import os
import re
from hashlib import sha256
from pathlib import Path
from typing import Any

import pymupdf

from .schemas import (
    Document,
    DocumentImage,
    DocumentPage,
    PageLayoutBlock,
    PageRecord,
    PageTableMetadata,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGE_OUTPUT_DIR = PROJECT_ROOT / "storage" / "images"
DOCUMENT_HASH_CHUNK_SIZE = 1024 * 1024
LAYOUT_TEXT_PREVIEW_CHARS = 160


def _build_document_id(path: Path) -> str:
    """Create a stable ID from the file content."""
    return _build_document_id_from_hash(_hash_file(path))


def _build_document_id_from_hash(content_hash: str) -> str:
    return content_hash[:16]


def _hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(DOCUMENT_HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clean_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in metadata.items()
        if value not in (None, "")
    }


def _rect_to_list(rect: Any) -> list[float]:
    try:
        values = list(rect)
    except TypeError:
        values = [
            getattr(rect, "x0", 0.0),
            getattr(rect, "y0", 0.0),
            getattr(rect, "x1", 0.0),
            getattr(rect, "y1", 0.0),
        ]
    return [round(float(value), 2) for value in values[:4]]


def _block_type_name(block_type: int | None) -> str:
    if block_type == 0:
        return "text"
    if block_type == 1:
        return "image"
    return "unknown"


def _normalize_preview(text: str, max_chars: int = LAYOUT_TEXT_PREVIEW_CHARS) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    if len(normalized) <= max_chars:
        return normalized
    return normalized[:max_chars].rsplit(" ", 1)[0].strip()


def _extract_block_text(block: dict[str, Any]) -> str:
    lines: list[str] = []
    for line in block.get("lines", []):
        spans = line.get("spans", []) if isinstance(line, dict) else []
        line_text = "".join(str(span.get("text", "")) for span in spans)
        if line_text.strip():
            lines.append(line_text.strip())
    return "\n".join(lines)


def _extract_page_layout(page: pymupdf.Page) -> tuple[list[PageLayoutBlock], dict[str, int]]:
    try:
        text_dict = page.get_text("dict")
    except Exception:
        return [], {"block_count": 0, "text_block_count": 0, "image_block_count": 0}

    layout_blocks: list[PageLayoutBlock] = []
    text_block_count = 0
    image_block_count = 0

    for block_index, block in enumerate(text_dict.get("blocks", [])):
        if not isinstance(block, dict):
            continue

        block_type = int(block.get("type", -1))
        block_type_name = _block_type_name(block_type)
        if block_type_name == "text":
            text_block_count += 1
        elif block_type_name == "image":
            image_block_count += 1

        lines = block.get("lines", [])
        line_count = len(lines) if isinstance(lines, list) else 0
        span_count = 0
        if isinstance(lines, list):
            span_count = sum(
                len(line.get("spans", []))
                for line in lines
                if isinstance(line, dict) and isinstance(line.get("spans", []), list)
            )

        layout_block: PageLayoutBlock = {
            "block_index": block_index,
            "block_type": block_type_name,
            "bbox": _rect_to_list(block.get("bbox", [0.0, 0.0, 0.0, 0.0])),
            "line_count": line_count,
            "span_count": span_count,
        }

        if block_type_name == "text":
            preview = _normalize_preview(_extract_block_text(block))
            if preview:
                layout_block["text_preview"] = preview

        layout_blocks.append(layout_block)

    return layout_blocks, {
        "block_count": len(layout_blocks),
        "text_block_count": text_block_count,
        "image_block_count": image_block_count,
    }


def _extract_page_heading(text: str, layout_blocks: list[PageLayoutBlock]) -> str:
    for block in layout_blocks:
        if block.get("block_type") != "text":
            continue
        candidate = str(block.get("text_preview", "")).strip()
        if _is_heading_candidate(candidate):
            return candidate

    for line in text.splitlines():
        candidate = line.strip()
        if _is_heading_candidate(candidate):
            return candidate

    return ""


def _is_heading_candidate(text: str) -> bool:
    if not text:
        return False
    words = text.split()
    return len(words) <= 14 and len(text) <= 140


def _extract_page_tables(page: pymupdf.Page) -> list[PageTableMetadata]:
    find_tables = getattr(page, "find_tables", None)
    if find_tables is None:
        return []

    try:
        table_finder = find_tables()
    except Exception:
        return []

    tables = getattr(table_finder, "tables", [])
    if not isinstance(tables, list):
        return []

    table_metadata: list[PageTableMetadata] = []
    for table_index, table in enumerate(tables):
        row_count = _int_value(getattr(table, "row_count", 0))
        column_count = _int_value(getattr(table, "col_count", 0))

        if row_count == 0 or column_count == 0:
            extracted = _safe_extract_table(table)
            if extracted:
                row_count = len(extracted)
                column_count = max((len(row) for row in extracted), default=0)

        table_metadata.append(
            {
                "table_index": table_index,
                "bbox": _rect_to_list(getattr(table, "bbox", [0.0, 0.0, 0.0, 0.0])),
                "row_count": row_count,
                "column_count": column_count,
            }
        )

    return table_metadata


def _safe_extract_table(table: Any) -> list[list[Any]]:
    extract = getattr(table, "extract", None)
    if extract is None:
        return []
    try:
        rows = extract()
    except Exception:
        return []
    return rows if isinstance(rows, list) else []


def _int_value(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _should_export_images() -> bool:
    return os.getenv("DOCUMENT_IMAGE_EXPORT", "true").lower() == "true"


def _should_ocr_images() -> bool:
    return os.getenv("DOCUMENT_IMAGE_OCR", "false").lower() == "true"


def _resolve_image_output_dir(image_output_dir: str | Path | None) -> Path | None:
    if image_output_dir is None:
        if not _should_export_images() and not _should_ocr_images():
            return None
        configured_dir = os.getenv("IMAGE_OUTPUT_DIR")
        image_output_dir = configured_dir if configured_dir else DEFAULT_IMAGE_OUTPUT_DIR

    output_dir = Path(image_output_dir)
    return output_dir if output_dir.is_absolute() else PROJECT_ROOT / output_dir


def _safe_extension(extension: str | None) -> str:
    if not extension:
        return "bin"
    cleaned = "".join(char for char in extension.lower() if char.isalnum())
    return cleaned or "bin"


def _extract_image_ocr_text(image_path: str | None) -> tuple[str | None, dict[str, Any]]:
    if not _should_ocr_images():
        return None, {"ocr_status": "disabled"}
    if not image_path:
        return None, {"ocr_status": "skipped_no_image_file"}

    try:
        import pytesseract  # type: ignore[import-not-found]

        text = pytesseract.image_to_string(image_path).strip()
    except ImportError:
        return None, {
            "ocr_status": "missing_dependency",
            "ocr_error": "Install pytesseract and the Tesseract OCR binary to enable OCR.",
        }
    except Exception as exc:
        return None, {
            "ocr_status": "failed",
            "ocr_error": str(exc),
        }

    if not text:
        return None, {"ocr_status": "empty"}
    return text, {
        "ocr_status": "extracted",
        "ocr_char_count": len(text),
        "ocr_word_count": len(text.split()),
    }


def _save_pdf_image(
    pdf: pymupdf.Document,
    xref: int,
    output_dir: Path,
    page_number: int,
    image_index: int,
) -> tuple[str | None, dict[str, Any]]:
    try:
        image_info = pdf.extract_image(xref)
        image_bytes = image_info.get("image")
        if not image_bytes:
            return None, {"extraction_status": "empty_image"}

        extension = _safe_extension(image_info.get("ext"))
        output_dir.mkdir(parents=True, exist_ok=True)
        image_path = output_dir / (
            f"page_{page_number:04d}_image_{image_index:04d}_xref_{xref}.{extension}"
        )
        image_path.write_bytes(image_bytes)
        return str(image_path), {
            "extraction_status": "exported",
            "extension": extension,
            "file_size_bytes": len(image_bytes),
        }
    except Exception as exc:
        return None, {
            "extraction_status": "failed",
            "extraction_error": str(exc),
        }


def _extract_images(
    pdf: pymupdf.Document,
    page: pymupdf.Page,
    document_id: str,
    page_number: int,
    output_dir: Path | None,
) -> list[DocumentImage]:
    images: list[DocumentImage] = []
    for image_index, image in enumerate(page.get_images(full=True)):
        xref = int(image[0])
        metadata = {
            "xref": xref,
            "width": image[2],
            "height": image[3],
            "bits_per_component": image[4],
            "colorspace": image[5],
            "name": image[7],
            "filter": image[8],
        }
        image_path = None
        if output_dir is not None:
            image_path, export_metadata = _save_pdf_image(
                pdf=pdf,
                xref=xref,
                output_dir=output_dir,
                page_number=page_number,
                image_index=image_index,
            )
            metadata.update(export_metadata)

        ocr_text, ocr_metadata = _extract_image_ocr_text(image_path)
        metadata.update(ocr_metadata)

        images.append(
            DocumentImage(
                document_id=document_id,
                page_number=page_number,
                image_index=image_index,
                image_path=image_path,
                ocr_text=ocr_text,
                metadata=metadata,
            )
        )
    return images


def load_pdf_document(
    pdf_path: str | Path,
    image_output_dir: str | Path | None = None,
) -> Document:
    """Load a PDF into a structured Document object."""
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    if not path.is_file():
        raise ValueError(f"PDF path is not a file: {path}")

    try:
        content_hash = _hash_file(path)
        document_id = _build_document_id_from_hash(content_hash)
        resolved_image_output_dir = _resolve_image_output_dir(image_output_dir)
        document_image_output_dir = (
            resolved_image_output_dir / document_id
            if resolved_image_output_dir is not None
            else None
        )

        with pymupdf.open(path) as pdf:
            pdf_metadata = _clean_metadata(pdf.metadata or {})
            pages: list[DocumentPage] = []

            for page_number, page in enumerate(pdf, start=1):
                text = page.get_text("text").strip()
                layout_blocks, layout_counts = _extract_page_layout(page)
                tables = _extract_page_tables(page)
                section_title = _extract_page_heading(text, layout_blocks)
                page_rect = page.rect
                images = _extract_images(
                    pdf=pdf,
                    page=page,
                    document_id=document_id,
                    page_number=page_number,
                    output_dir=document_image_output_dir,
                )
                image_text = "\n\n".join(
                    image_text_value
                    for image in images
                    if (image_text_value := image.to_chunk_text())
                )
                searchable_text = "\n\n".join(part for part in [text, image_text] if part)
                if text or images:
                    pages.append(
                        DocumentPage(
                            document_id=document_id,
                            page_number=page_number,
                            text=text,
                            source_name=path.name,
                            images=images,
                            metadata={
                                "char_count": len(text),
                                "word_count": len(text.split()),
                                "searchable_char_count": len(searchable_text),
                                "searchable_word_count": len(searchable_text.split()),
                                "image_count": len(images),
                                "image_text_count": sum(
                                    1 for image in images if image.to_chunk_text()
                                ),
                                "page_width": round(float(page_rect.width), 2),
                                "page_height": round(float(page_rect.height), 2),
                                "page_rotation": int(page.rotation),
                                "section_title": section_title,
                                "block_count": layout_counts["block_count"],
                                "text_block_count": layout_counts["text_block_count"],
                                "image_block_count": layout_counts["image_block_count"],
                                "table_count": len(tables),
                                "layout_blocks": layout_blocks,
                                "tables": tables,
                            },
                        )
                    )
    except Exception as exc:
        raise ValueError(f"Could not read PDF file: {path}") from exc

    return Document(
        document_id=document_id,
        source_path=str(path),
        source_name=path.name,
        document_type="pdf",
        title=pdf_metadata.get("title"),
        author=pdf_metadata.get("author"),
        created_at=pdf_metadata.get("creationDate"),
        metadata={
            **pdf_metadata,
            "page_count": len(pages),
            "image_count": sum(len(page.images) for page in pages),
            "document_id_strategy": "sha256_file_content_16",
            "content_hash": content_hash,
            "image_output_dir": (
                str(document_image_output_dir)
                if document_image_output_dir is not None
                else None
            ),
            "file_size_bytes": path.stat().st_size,
        },
        pages=pages,
    )


def load_pdf(pdf_path: str | Path) -> list[PageRecord]:
    """Extract page records from a PDF file for chunking."""
    return load_pdf_document(pdf_path).to_chunk_inputs()
