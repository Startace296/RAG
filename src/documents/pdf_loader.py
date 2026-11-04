import os
import re
from hashlib import sha256
from pathlib import Path
from typing import Any

import pymupdf

from ..core import DEFAULT_STORAGE_DIR, resolve_path
from .schemas import (
    Document,
    DocumentImage,
    DocumentPage,
    PageLayoutBlock,
    PageRecord,
    PageTableMetadata,
)


DEFAULT_IMAGE_OUTPUT_DIR = DEFAULT_STORAGE_DIR / "images"
DOCUMENT_HASH_CHUNK_SIZE = 1024 * 1024
LAYOUT_TEXT_PREVIEW_CHARS = 160
DEFAULT_IMAGE_MIN_SIDE = 32
# Bump when text extraction or chunk text changes so saved indexes are rebuilt.
DOCUMENT_PARSER_VERSION = 4
# A line is a running header/footer when it appears on at least this share of pages.
BOILERPLATE_PAGE_RATIO = 0.3
BOILERPLATE_MIN_PAGES = 3
BOILERPLATE_MIN_CHARS = 15
# The same line repeated this many times on one page is a stamp or watermark.
BOILERPLATE_REPEATS_PER_PAGE = 3
PAGE_EDGE_LINES = 4
# Lines at least this long that end without a break were wrapped at the page margin.
WRAPPED_LINE_MIN_CHARS = 40
# Section headings use a font at least this much larger than the body text.
HEADING_FONT_RATIO = 1.4
# A heading block ending with one of these words continues in the next block.
HEADING_CONTINUATION_WORDS = (" và", " của", " với", " cho", " and", " of")
# Figure captions and source notes are short but are not section headings.
HEADING_EXCLUDED_PREFIXES = ("nguồn:", "source:", "hình", "figure", "bảng", "table")
PAGE_NUMBER_PATTERN = re.compile(r"^\d{1,4}\s+|\s+\d{1,4}$")
# "page": one text blob per page; "structured": body/figure/table segments with headings and boxes.
DOCUMENT_MODELS = {"page", "structured"}
# Page numbers are printed alone in the small footer font.
PAGE_NUMBER_FONT_SIZES = (8.5, 9.5)


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


def _extract_page_heading(text: str) -> str:
    """Return the first short paragraph of the (cleaned) page text."""
    for paragraph in re.split(r"\n\s*\n", text):
        candidate = _normalize_line(paragraph)
        if _is_heading_candidate(candidate):
            return candidate
    return ""


def _text_lines(page: pymupdf.Page) -> list[list[list[dict[str, Any]]]]:
    """Return the text blocks of the page, top to bottom, as lines of spans."""
    try:
        blocks = page.get_text("dict", sort=True).get("blocks", [])
    except Exception:
        return []
    return [
        [line.get("spans", []) for line in block.get("lines", [])]
        for block in blocks
        if isinstance(block, dict) and block.get("type") == 0
    ]


def _body_font_size(pdf: pymupdf.Document) -> float:
    """Return the font size that covers the most characters in the document."""
    char_counts: dict[float, int] = {}
    for page in pdf:
        for lines in _text_lines(page):
            for spans in lines:
                for span in spans:
                    size = round(float(span.get("size", 0.0)), 1)
                    char_counts[size] = char_counts.get(size, 0) + len(str(span.get("text", "")).strip())
    return max(char_counts, key=char_counts.__getitem__) if char_counts else 0.0


def _extract_font_headings(
    page: pymupdf.Page, body_size: float, repeated_lines: set[str]
) -> list[str]:
    """Return headings on the page, recognised by a font clearly larger than body text.

    Only the large spans of a block are kept, so a heading printed in the same
    block as its subtitle does not swallow the subtitle.
    """
    if body_size <= 0:
        return []
    min_size = body_size * HEADING_FONT_RATIO
    fragments: list[str] = []
    previous_was_heading = False
    for lines in _text_lines(page):
        # Spans of one line can split a word, so they are joined without spaces.
        heading = _normalize_line(
            " ".join(
                "".join(
                    str(span.get("text", ""))
                    for span in spans
                    if float(span.get("size", 0.0)) >= min_size
                )
                for spans in lines
            )
        )
        if not heading or _line_key(heading) in repeated_lines:
            previous_was_heading = False
            continue
        # A long heading wrapped over two blocks ("B. Diễn biến kinh tế" + "gần đây").
        if previous_was_heading and _continues_heading(fragments[-1], heading):
            fragments[-1] = f"{fragments[-1]} {heading}"
        else:
            fragments.append(heading)
        previous_was_heading = True
    return [heading for heading in fragments if _is_heading_candidate(heading)]


def _continues_heading(previous: str, current: str) -> bool:
    return (
        current[0].islower()
        or previous.endswith((",", "-", "–"))
        or previous.lower().endswith(HEADING_CONTINUATION_WORDS)
    )


def _is_heading_candidate(text: str) -> bool:
    if not text or not any(char.isalpha() for char in text):
        return False
    lowered = text.lower()
    if lowered.startswith(HEADING_EXCLUDED_PREFIXES) or "nguồn:" in lowered:
        return False
    words = text.split()
    # Chart axes such as "2024 2025e 2026f" are mostly numbers.
    if sum(1 for word in words if word[0].isalpha()) * 2 < len(words):
        return False
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


def _normalize_line(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()


def _line_key(line: str) -> str:
    """Compare lines without a page number glued to either end.

    Running headers often carry the page number on the same line
    ("12 Cập nhật ..."), so the raw line differs on every page.
    """
    return PAGE_NUMBER_PATTERN.sub("", _normalize_line(line))


def _extract_page_text(page: pymupdf.Page) -> str:
    """Return the page text with one text block per paragraph.

    Blocks are separated by a blank line so chunking can split on real
    paragraphs instead of falling back to sentences.
    """
    blocks = [
        str(block[4]).strip()
        for block in page.get_text("blocks")
        if len(block) > 6 and block[6] == 0 and str(block[4]).strip()
    ]
    return "\n\n".join(blocks)


def _find_repeated_lines(page_texts: list[str]) -> set[str]:
    """Return line keys that repeat across many pages, such as running headers."""
    page_counts: dict[str, int] = {}
    for text in page_texts:
        for line in {_line_key(line) for line in text.splitlines()}:
            if len(line) >= BOILERPLATE_MIN_CHARS:
                page_counts[line] = page_counts.get(line, 0) + 1

    min_pages = max(BOILERPLATE_MIN_PAGES, round(len(page_texts) * BOILERPLATE_PAGE_RATIO))
    return {line for line, count in page_counts.items() if count >= min_pages}


def _strip_boilerplate(text: str, repeated_lines: set[str]) -> tuple[str, int]:
    """Remove running headers/footers, page numbers at the page edges and stamps.

    These lines carry no content but appear in many chunks, which pulls every
    chunk towards queries that mention the document title. Paragraphs
    (separated by blank lines) are kept, and the PDF line wraps inside each
    paragraph are joined with spaces.
    """
    paragraph_ids: list[int] = []
    lines: list[str] = []
    for paragraph_id, paragraph in enumerate(re.split(r"\n\s*\n", text)):
        for line in map(_normalize_line, paragraph.splitlines()):
            if line:
                paragraph_ids.append(paragraph_id)
                lines.append(line)
    line_counts: dict[str, int] = {}
    for line in lines:
        line_counts[line] = line_counts.get(line, 0) + 1

    def is_stamp(line: str) -> bool:
        return (
            len(line) >= BOILERPLATE_MIN_CHARS
            and line_counts[line] >= BOILERPLATE_REPEATS_PER_PAGE
        )

    def is_edge_noise(line: str, distance_from_edge: int) -> bool:
        # Stamps only count at the page edges: inside the page a repeated line is
        # usually real content, such as the same figure source under several charts.
        if _line_key(line) in repeated_lines or is_stamp(line):
            return True
        return line.isdigit() and distance_from_edge < PAGE_EDGE_LINES

    start = 0
    while start < len(lines) and is_edge_noise(lines[start], start):
        start += 1
    end = len(lines)
    while end > start and is_edge_noise(lines[end - 1], len(lines) - end):
        end -= 1

    paragraphs: dict[int, list[str]] = {}
    for index in range(start, end):
        if _line_key(lines[index]) not in repeated_lines:
            paragraphs.setdefault(paragraph_ids[index], []).append(lines[index])
    removed = len(lines) - sum(len(paragraph) for paragraph in paragraphs.values())
    return "\n\n".join(_join_wrapped_lines(paragraph) for paragraph in paragraphs.values()), removed


def _join_wrapped_lines(lines: list[str]) -> str:
    """Join lines the PDF wrapped at the margin; keep breaks after short lines.

    A long line that stops mid-sentence was wrapped, while short lines are
    titles or labels ("Cập nhật Tình hình Kinh tế" / "Kiên định Cải cách,")
    whose line breaks carry structure the LLM needs.
    """
    text = lines[0] if lines else ""
    for previous, line in zip(lines, lines[1:]):
        wrapped = len(previous) >= WRAPPED_LINE_MIN_CHARS
        text += (" " if wrapped else "\n") + line
    return text


def _should_strip_boilerplate() -> bool:
    return os.getenv("DOCUMENT_STRIP_BOILERPLATE", "true").lower() == "true"


def _document_model(document_model: str | None) -> str:
    model = (document_model or os.getenv("DOCUMENT_MODEL", "structured")).strip().lower()
    if model not in DOCUMENT_MODELS:
        raise ValueError(f"Unsupported document model: {model}. Use one of: {', '.join(sorted(DOCUMENT_MODELS))}")
    return model


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

    return resolve_path(image_output_dir)


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


def _image_min_side() -> int:
    try:
        return max(0, int(os.getenv("IMAGE_MIN_SIDE", str(DEFAULT_IMAGE_MIN_SIDE))))
    except ValueError:
        return DEFAULT_IMAGE_MIN_SIDE


def _remove_stale_images(output_dir: Path, keep: set[Path]) -> None:
    """Remove images an earlier run exported for this document that this run did not."""
    if not output_dir.is_dir():
        return
    for path in output_dir.glob("page_*_image_*_xref_*.*"):
        if path.is_file() and path.resolve() not in keep:
            path.unlink()


def _extract_images(
    pdf: pymupdf.Document,
    page: pymupdf.Page,
    document_id: str,
    page_number: int,
    output_dir: Path | None,
) -> tuple[list[DocumentImage], int]:
    """Return useful images on the page and the number of skipped ones.

    Images smaller than IMAGE_MIN_SIDE pixels on either side (masks, spacers,
    1x1 or 2x2 pixels) and repeated references to the same image on the same
    page are skipped.
    """
    images: list[DocumentImage] = []
    skipped = 0
    seen_xrefs: set[int] = set()
    min_side = _image_min_side()

    for image in page.get_images(full=True):
        xref = int(image[0])
        width = _int_value(image[2])
        height = _int_value(image[3])
        if xref in seen_xrefs or min(width, height) < min_side:
            skipped += 1
            continue
        seen_xrefs.add(xref)

        image_index = len(images)
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
    return images, skipped


def load_pdf_document(
    pdf_path: str | Path,
    image_output_dir: str | Path | None = None,
    strip_boilerplate: bool | None = None,
    document_model: str | None = None,
) -> Document:
    """Load a PDF into a structured Document object.

    ``strip_boilerplate`` removes running headers/footers and page numbers;
    ``None`` reads DOCUMENT_STRIP_BOILERPLATE (default true).
    ``document_model="structured"`` also splits every page into body, figure
    and table segments; ``None`` reads DOCUMENT_MODEL (default "structured").

    Raises:
        ValueError: If PyMuPDF cannot open the file. Errors while parsing an
            opened PDF are raised as they are, so bugs are not reported as a
            broken file.
    """
    from .structure import (
        StructureState,
        attach_figure_references,
        document_info_segment,
        extract_page_segments,
    )

    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    if not path.is_file():
        raise ValueError(f"PDF path is not a file: {path}")

    content_hash = _hash_file(path)
    document_id = _build_document_id_from_hash(content_hash)
    resolved_image_output_dir = _resolve_image_output_dir(image_output_dir)
    document_image_output_dir = (
        resolved_image_output_dir / document_id
        if resolved_image_output_dir is not None
        else None
    )

    try:
        opened_pdf = pymupdf.open(path)
    except Exception as exc:
        raise ValueError(f"Could not read PDF file: {path}") from exc

    with opened_pdf as pdf:
        pdf_metadata = _clean_metadata(pdf.metadata or {})
        pages: list[DocumentPage] = []
        skipped_image_count = 0
        raw_page_texts = [_extract_page_text(page) for page in pdf]
        if strip_boilerplate is None:
            strip_boilerplate = _should_strip_boilerplate()
        repeated_lines = _find_repeated_lines(raw_page_texts) if strip_boilerplate else set()
        body_font_size = _body_font_size(pdf)
        current_section = ""
        structured = _document_model(document_model) == "structured"
        structure_state = StructureState()

        def is_noise_line(line: str, size: float) -> bool:
            if _line_key(line) in repeated_lines:
                return True
            low, high = PAGE_NUMBER_FONT_SIZES
            return line.isdigit() and len(line) <= 3 and low <= size <= high

        info_segment, cover_info = (
            document_info_segment(pdf[0], path.name, is_noise_line)
            if structured and len(pdf)
            else (None, {})
        )

        for page_number, page in enumerate(pdf, start=1):
            text = raw_page_texts[page_number - 1].strip()
            removed_line_count = 0
            if strip_boilerplate:
                text, removed_line_count = _strip_boilerplate(text, repeated_lines)
            layout_blocks, layout_counts = _extract_page_layout(page)
            tables = _extract_page_tables(page)
            # A page without a heading continues the section of the previous page.
            font_headings = _extract_font_headings(page, body_font_size, repeated_lines)
            # The lowest heading on the page is the most specific one, e.g. the
            # section "Bối cảnh toàn cầu" below the chapter "A. Tình hình thế giới".
            section_title = (
                font_headings[-1] if font_headings else current_section
            ) or _extract_page_heading(text)
            if font_headings:
                current_section = font_headings[-1]
            page_rect = page.rect
            images, skipped_on_page = _extract_images(
                pdf=pdf,
                page=page,
                document_id=document_id,
                page_number=page_number,
                output_dir=document_image_output_dir,
            )
            skipped_image_count += skipped_on_page
            image_text = "\n\n".join(
                image_text_value
                for image in images
                if (image_text_value := image.to_chunk_text())
            )
            searchable_text = "\n\n".join(part for part in [text, image_text] if part)
            segments = (
                extract_page_segments(
                    page,
                    images,
                    structure_state,
                    is_noise_line=is_noise_line,
                    join_lines=_join_wrapped_lines,
                    continues_heading=_continues_heading,
                    is_heading_candidate=_is_heading_candidate,
                )
                if structured
                else None
            )
            if page_number == 1 and info_segment is not None and segments is not None:
                segments.insert(0, info_segment)
            if text or images:
                pages.append(
                    DocumentPage(
                        document_id=document_id,
                        page_number=page_number,
                        text=text,
                        source_name=path.name,
                        images=images,
                        segments=segments,
                        metadata={
                            "char_count": len(text),
                            "word_count": len(text.split()),
                            "boilerplate_line_count": removed_line_count,
                            "searchable_char_count": len(searchable_text),
                            "searchable_word_count": len(searchable_text.split()),
                            "image_count": len(images),
                            "skipped_image_count": skipped_on_page,
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
        if structured:
            attach_figure_references([(page.page_number, page.segments or []) for page in pages])

    # Old images are removed only after a successful parse, so a failed run keeps them.
    if document_image_output_dir is not None:
        _remove_stale_images(
            document_image_output_dir,
            keep={
                Path(image.image_path).resolve()
                for page in pages
                for image in page.images
                if image.image_path
            },
        )

    return Document(
        document_id=document_id,
        source_path=str(path),
        source_name=path.name,
        document_type="pdf",
        title=pdf_metadata.get("title") or cover_info.get("title"),
        author=pdf_metadata.get("author"),
        created_at=pdf_metadata.get("creationDate"),
        metadata={
            **pdf_metadata,
            **{f"cover_{key}": value for key, value in cover_info.items()},
            "page_count": len(pages),
            "image_count": sum(len(page.images) for page in pages),
            "skipped_image_count": skipped_image_count,
            "boilerplate_lines": sorted(repeated_lines),
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


def load_pdf(
    pdf_path: str | Path,
    strip_boilerplate: bool | None = None,
    document_model: str | None = None,
) -> list[PageRecord]:
    """Extract page records from a PDF file for chunking."""
    return load_pdf_document(
        pdf_path, strip_boilerplate=strip_boilerplate, document_model=document_model
    ).to_chunk_inputs()
