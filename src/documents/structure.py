"""Structured document model: split each PDF page into body text, figures and tables.

The page model keeps one text blob per page, so a chunk mixes body paragraphs
with chart captions and source lines, and it cannot tell which box, figure or
table a passage belongs to. Questions about this report refer to exactly that
structure ("Hộp 2", "Hình 16", "Bảng 3", "nguồn số liệu"), so this module reads
the page layout and returns ``DocumentSegment`` records that keep it.

Layout rules (fonts and sizes) were derived from KinhteVN.pdf:

- chapter headings (>= 30pt), section headings (>= 15.5pt) and bold-italic
  sub-headings (about 11-14pt) form a three-level heading path;
- "HÌNH 16" labels, chart titles in OpenSans and "Nguồn: ..." lines in a small
  font describe a figure;
- "Bảng 3." captions and small Calibri cells form a table;
- "HỘP 1" opens a box that runs to the end of the page.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Callable

import pymupdf

from .schemas import DocumentImage, DocumentSegment


CHAPTER_MIN_SIZE = 30.0
SECTION_MIN_SIZE = 15.5
SUBHEADING_MIN_SIZE = 10.9
SUBHEADING_MAX_CHARS = 220
SOURCE_MAX_SIZE = 8.5
# A caption or source line belongs to the closest figure label above it in the same column.
CAPTION_MAX_GAP = 140.0
IMAGE_MAX_GAP = 320.0
# Smaller images next to a label are the label decoration, not the chart.
FIGURE_IMAGE_MIN_HEIGHT = 30.0
TABLE_ROW_TOLERANCE = 4.0
MAX_FIGURE_REFERENCES = 2
MIN_CITING_SENTENCE_CHARS = 40
TOC_LEADER = re.compile(r"\.{4,}|…{2,}|\.…|…\.")
TOC_HEADINGS = {"MỤC LỤC", "HÌNH", "BẢNG"}
COVER_DATE = re.compile(r"Tháng\s+\d{1,2}/\d{4}", re.I)
STAMP_REPEATS = 3
STAMP_MIN_CHARS = 15

FIGURE_LABEL = re.compile(r"^HÌNH\s+(B?\d+(?:\.\d+)?)\s*[:.]?\s*(.*)$", re.S)
BOX_LABEL = re.compile(r"^HỘP\s+(\d+)\s*(.*)$", re.S)
TABLE_LABEL = re.compile(r"^Bảng\s+(\d+)\.\s*(.*)$", re.S)
SOURCE_PREFIXES = ("nguồn", "ghi chú")
SENTENCE_BREAK = re.compile(r"(?<=[.!?…])\s+")


@dataclass
class _Line:
    text: str
    size: float
    fonts: list[str]

    @property
    def bold(self) -> bool:
        return all("Bold" in font for font in self.fonts)

    @property
    def bold_italic(self) -> bool:
        return all("BoldIt" in font for font in self.fonts)


@dataclass
class _Block:
    bbox: tuple[float, float, float, float]
    lines: list[_Line]

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)

    @property
    def max_size(self) -> float:
        return max(line.size for line in self.lines)

    @property
    def fonts(self) -> set[str]:
        return {font for line in self.lines for font in line.fonts}


@dataclass
class _Owner:
    """A figure or table that collects captions, source lines, cells and images."""

    kind: str
    number: str
    bbox: tuple[float, float, float, float]
    heading_path: list[str]
    box_id: str
    box_title: str
    captions: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    cells: list[_Block] = field(default_factory=list)
    image_indexes: list[int] = field(default_factory=list)


@dataclass
class StructureState:
    """Heading path carried from page to page while a document is read."""

    heading_path: list[str] = field(default_factory=list)
    last_heading_level: int = 0


def extract_page_segments(
    page: pymupdf.Page,
    images: list[DocumentImage],
    state: StructureState,
    is_noise_line: Callable[[str, float], bool],
    join_lines: Callable[[list[str]], str],
    continues_heading: Callable[[str, str], bool],
    is_heading_candidate: Callable[[str], bool],
) -> list[DocumentSegment]:
    """Return the body, figure and table segments of one page, in reading order."""
    blocks = _read_blocks(page, is_noise_line)
    mid_x = float(page.rect.width) / 2
    box = _find_box(blocks)

    def box_of(block: _Block) -> tuple[str, str]:
        if box is not None and block.bbox[1] >= box["y0"] - 2:
            return box["id"], box["title"]
        return "", ""

    owners: list[_Owner] = []
    body_blocks: list[_Block] = []
    # First pass: follow the headings so each figure gets the path of its position.
    scan = StructureState(list(state.heading_path), state.last_heading_level)
    for block in _reading_order(blocks, mid_x):
        first = block.lines[0].text
        if box is not None and id(block) in box["blocks"]:
            continue
        figure = FIGURE_LABEL.match(first)
        table = TABLE_LABEL.match(first)
        if figure:
            box_id, box_title = box_of(block)
            owner = _Owner("figure", figure.group(1), block.bbox, _visible_path(scan.heading_path), box_id, box_title)
            rest = " ".join([figure.group(2).strip(), *(line.text for line in block.lines[1:])]).strip()
            if rest:
                owner.captions.append(rest)
            owners.append(owner)
        elif table and block.lines[0].bold:
            box_id, box_title = box_of(block)
            owner = _Owner("table", table.group(1), block.bbox, _visible_path(scan.heading_path), box_id, box_title)
            caption = " ".join([table.group(2).strip(), *(line.text for line in block.lines[1:])])
            owner.captions.append(caption.strip())
            owners.append(owner)
        else:
            body_blocks.append(block)
            _split_heading_lines(block, scan, box_of(block)[0], continues_heading, is_heading_candidate)

    figures = [owner for owner in owners if owner.kind == "figure"]
    tables = [owner for owner in owners if owner.kind == "table"]

    # A table spans from its caption down to the first source line below it; the
    # cells use the body font in some tables, so position is what identifies them.
    by_position = sorted(body_blocks, key=lambda block: (block.bbox[1], block.bbox[0]))
    table_regions: list[tuple[_Owner, float, _Block | None]] = []
    for table in tables:
        source = next(
            (block for block in by_position if _is_source_block(block) and block.bbox[1] > table.bbox[1]),
            None,
        )
        table_regions.append((table, source.bbox[1] if source else float(page.rect.height), source))

    # Route captions, sources and table cells to their owner, top to bottom.
    remaining: list[_Block] = []
    for block in by_position:
        region = next(
            (
                (table, source)
                for table, bottom, source in table_regions
                if table.bbox[1] - 5 <= block.bbox[1] and block.bbox[3] <= bottom + 1
            ),
            None,
        )
        if _is_source_block(block):
            owner = next((table for table, _, source in table_regions if source is block), None)
            if owner is None:
                owner = _closest_owner(block, figures, mid_x, CAPTION_MAX_GAP)
            if owner is not None:
                owner.sources.append(_normalize(block.text))
                continue
        if any("OpenSans" in font for font in block.fonts) and figures:
            owner = _closest_owner(block, figures, mid_x, CAPTION_MAX_GAP)
            if owner is not None:
                caption, _, source = _normalize(block.text).partition("Nguồn:")
                if caption.strip():
                    owner.captions.append(caption.strip())
                if source.strip():
                    owner.sources.append(f"Nguồn: {source.strip()}")
                continue
        if region is not None:
            region[0].cells.append(block)
            continue
        remaining.append(block)

    _assign_images(page, images, figures, mid_x)

    # Body segments: consecutive blocks with the same heading path and box.
    # Second pass: heading tracking restarts from the path the page started with.
    segments: list[DocumentSegment] = []
    body_state = StructureState(list(state.heading_path), state.last_heading_level)
    current: DocumentSegment | None = None
    paragraphs: list[str] = []
    has_text = False

    def flush(end_of_page: bool = False) -> None:
        nonlocal current, paragraphs, has_text
        # A page of headings only (the cover) still keeps its text at the end of the page.
        if current is not None and (has_text or (end_of_page and paragraphs)):
            current["text"] = "\n\n".join(paragraphs)
            segments.append(current)
            paragraphs = []
        # A heading with no text of its own ("Cú sốc dầu mỏ Trung Đông" right above
        # a sub-heading) is carried into the next segment instead of standing alone.
        current, has_text = None, False

    for block in _reading_order(remaining, mid_x):
        box_id, box_title = box_of(block)
        heading_lines, text_lines = _split_heading_lines(
            block, body_state, box_id, continues_heading, is_heading_candidate
        )
        if _is_toc_block(block, body_state.heading_path):
            continue
        if current is None or (tuple(current["heading_path"]), current["box_id"]) != (
            tuple(_visible_path(body_state.heading_path)),
            box_id,
        ):
            flush()
            current = {
                "element_type": "body",
                "heading_path": _visible_path(body_state.heading_path),
                "box_id": box_id,
                "box_title": box_title,
                "bbox": list(block.bbox),
            }
        paragraphs.extend(heading_lines)
        if text_lines:
            paragraphs.append(join_lines(text_lines))
            has_text = True
    page_has_text = any(segment["element_type"] == "body" for segment in segments) or has_text
    flush(end_of_page=True)
    # A page of headings only is a cover: its large title lines do not open a chapter.
    if page_has_text:
        state.heading_path = body_state.heading_path
        state.last_heading_level = body_state.last_heading_level

    for owner in owners:
        segments.append(_owner_segment(owner))
    return segments


def document_info_segment(
    page: pymupdf.Page, source_name: str, is_noise_line: Callable[[str, float], bool]
) -> tuple[DocumentSegment | None, dict[str, str]]:
    """Read the report's title, subtitle and date from its cover page.

    Questions such as "báo cáo có tiêu đề phụ là gì?" ask about the document
    itself, not about a passage, so these values are stored as document
    metadata and as one searchable "document info" segment.
    """
    lines = [
        line
        for block in sorted(_read_blocks(page, is_noise_line), key=lambda block: block.bbox[1])
        for line in block.lines
    ]
    if not lines:
        return None, {}
    largest = max(line.size for line in lines)
    title = " ".join(line.text for line in lines if line.size >= largest * 0.9)
    subtitle = " ".join(line.text for line in lines if line.bold_italic and line.size < largest * 0.9)
    date = next((match.group(0) for line in lines if (match := COVER_DATE.search(line.text))), "")
    info = {key: value for key, value in (("title", title), ("subtitle", subtitle), ("date", date)) if value}
    if not title:
        return None, info
    text_lines = ["Thông tin tài liệu", f"Tên báo cáo: {title}"]
    if subtitle:
        text_lines.append(f"Tiêu đề phụ: {subtitle}")
    if date:
        text_lines.append(f"Thời gian phát hành: {date}")
    text_lines.append(f"Tệp nguồn: {source_name}")
    segment: DocumentSegment = {
        "element_type": "document_info",
        "text": "\n".join(text_lines),
        "heading_path": [],
        "box_id": "",
        "box_title": "",
    }
    return segment, info


def attach_figure_references(pages: list[tuple[int, list[DocumentSegment]]]) -> None:
    """Add the body sentences that cite each figure ("... (Hình 16).") to its segment.

    A chart title alone ("Chỉ số PMI") rarely says what the chart shows; the
    report text that cites it does. Sentences on the same page come first.
    """
    sentences: list[tuple[int, str]] = []
    for page_number, segments in pages:
        for segment in segments:
            if segment.get("element_type") == "body":
                for paragraph in segment.get("text", "").split("\n\n"):
                    sentences.extend(
                        (page_number, sentence.strip())
                        for sentence in SENTENCE_BREAK.split(paragraph)
                        if _is_citing_sentence(sentence.strip())
                    )

    for page_number, segments in pages:
        for segment in segments:
            number = segment.get("figure_no", "")
            if segment.get("element_type") != "figure" or not number:
                continue
            pattern = re.compile(rf"\bHình\s+{re.escape(number)}(?![\d.]\d)")
            citing = sorted(
                (abs(cited_page - page_number), sentence)
                for cited_page, sentence in sentences
                if pattern.search(sentence)
            )
            references = [sentence for _, sentence in citing[:MAX_FIGURE_REFERENCES]]
            if references:
                segment["text"] = segment["text"] + "\nNội dung liên quan: " + " ".join(references)


def _is_citing_sentence(sentence: str) -> bool:
    # Table-of-contents entries ("Hình 18. Chi tiêu hộ gia đình ......8") are not citations.
    return len(sentence) >= MIN_CITING_SENTENCE_CHARS and not TOC_LEADER.search(sentence)


def _reading_order(blocks: list[_Block], mid_x: float) -> list[_Block]:
    """Order blocks as a reader would: left column, then right column, band by band.

    A full-width block (a box introduction, a wide chart) closes the band above it,
    so two-column text below it is read after it.
    """
    ordered: list[_Block] = []
    band: list[_Block] = []

    def close_band() -> None:
        order = {"left": 0, "right": 1}
        left = [block for block in band if _column(block.bbox, mid_x) == "left"]
        right = [block for block in band if _column(block.bbox, mid_x) == "right"]
        # Read by column only when the columns sit side by side; a left heading below a
        # right-hand list ("LỜI CẢM ƠN" under "TỪ VIẾT TẮT") is read top to bottom.
        side_by_side = (
            left
            and right
            and max(min(b.bbox[1] for b in left), min(b.bbox[1] for b in right))
            < min(max(b.bbox[3] for b in left), max(b.bbox[3] for b in right))
        )
        if side_by_side:
            band.sort(key=lambda block: (order[_column(block.bbox, mid_x)], block.bbox[1]))
        ordered.extend(band)
        band.clear()

    for block in sorted(blocks, key=lambda block: (block.bbox[1], block.bbox[0])):
        if _column(block.bbox, mid_x) == "full":
            close_band()
            ordered.append(block)
        else:
            band.append(block)
    close_band()
    return ordered


def _read_blocks(page: pymupdf.Page, is_noise_line: Callable[[str, float], bool]) -> list[_Block]:
    try:
        raw_blocks = page.get_text("dict").get("blocks", [])
    except Exception:
        return []
    blocks: list[_Block] = []
    for raw in raw_blocks:
        if not isinstance(raw, dict) or raw.get("type") != 0:
            continue
        lines: list[_Line] = []
        for raw_line in raw.get("lines", []):
            spans = [span for span in raw_line.get("spans", []) if str(span.get("text", "")).strip()]
            if not spans:
                continue
            # Spans of one line can split a word, so they are joined without spaces.
            text = _normalize("".join(str(span.get("text", "")) for span in raw_line.get("spans", [])))
            size = max(float(span.get("size", 0.0)) for span in spans)
            if text and not is_noise_line(text, size):
                # Punctuation such as "…" is often set in another font; only letters decide the style.
                styled = [span for span in spans if any(char.isalnum() for char in str(span.get("text", "")))]
                fonts = [str(span.get("font", "")) for span in styled or spans]
                lines.append(_Line(text, size, fonts))
        if lines:
            blocks.append(_Block(tuple(float(v) for v in raw.get("bbox", (0, 0, 0, 0))), lines))

    # A line printed several times on one page is a stamp ("Public Disclosure Authorized").
    counts: dict[str, int] = {}
    for block in blocks:
        for line in block.lines:
            counts[line.text] = counts.get(line.text, 0) + 1
    stamps = {text for text, count in counts.items() if count >= STAMP_REPEATS and len(text) >= STAMP_MIN_CHARS}
    for block in blocks:
        block.lines = [line for line in block.lines if line.text not in stamps]
    return [block for block in blocks if block.lines]


def _is_toc_block(block: _Block, heading_path: list[str]) -> bool:
    """Table-of-contents entries repeat headings and captions; nobody asks for them."""
    if any(heading.upper() in TOC_HEADINGS for heading in heading_path):
        return True
    # "HÌNH" / "BẢNG" list titles are not headings (captions start with these words).
    if all(line.text.upper() in TOC_HEADINGS for line in block.lines):
        return True
    leader_lines = sum(1 for line in block.lines if TOC_LEADER.search(line.text))
    return leader_lines * 2 >= len(block.lines)


def _find_box(blocks: list[_Block]) -> dict[str, Any] | None:
    """Return the box opened on this page: its number, title, top edge and title blocks."""
    for index, block in enumerate(blocks):
        match = BOX_LABEL.match(block.lines[0].text)
        if not match or not block.lines[0].bold:
            continue
        title = " ".join([match.group(2).strip(), *(line.text for line in block.lines[1:])]).strip()
        title_blocks = {id(block)}
        if not title:
            # "HỘP 2" and its title can be two blocks: take the next bold block below it.
            for candidate in blocks[index + 1:]:
                if candidate.bbox[1] >= block.bbox[1] and candidate.lines[0].bold and candidate.max_size >= 11:
                    title = _normalize(candidate.text)
                    title_blocks.add(id(candidate))
                    break
        return {"id": match.group(1), "title": title, "y0": block.bbox[1], "blocks": title_blocks}
    return None


def _heading_level(line: _Line, is_heading_candidate: Callable[[str], bool]) -> int:
    """Return 1 for a chapter, 2 for a section, 3 for a sub-heading, 0 for body text."""
    if line.size >= CHAPTER_MIN_SIZE and is_heading_candidate(line.text):
        return 1
    if line.size >= SECTION_MIN_SIZE and is_heading_candidate(line.text):
        return 2
    if line.bold_italic and line.size >= SUBHEADING_MIN_SIZE and len(line.text) <= SUBHEADING_MAX_CHARS:
        return 3
    return 0


def _split_heading_lines(
    block: _Block,
    state: StructureState,
    box_id: str,
    continues_heading: Callable[[str, str], bool],
    is_heading_candidate: Callable[[str], bool],
) -> tuple[list[str], list[str]]:
    """Update the heading path from the block's heading lines; return (headings, text lines)."""
    headings: list[str] = []
    text_lines: list[str] = []
    for line in block.lines:
        # Inside a box the bold lines are the box's own layout, not report headings.
        level = 0 if box_id else _heading_level(line, is_heading_candidate)
        if not level:
            text_lines.append(line.text)
            state.last_heading_level = 0
            continue
        path = state.heading_path
        if (
            state.last_heading_level == level
            and len(path) >= level
            and continues_heading(path[level - 1], line.text)
        ):
            path[level - 1] = f"{path[level - 1]} {line.text}"
            if headings:
                headings[-1] = f"{headings[-1]} {line.text}"
        else:
            del path[level - 1:]
            path.extend([""] * (level - 1 - len(path)))
            path.append(line.text)
            headings.append(line.text)
        state.last_heading_level = level
    return headings, text_lines


def _visible_path(heading_path: list[str]) -> list[str]:
    # The state keeps "" for a missing level (a section before any chapter) so levels stay aligned.
    return [heading for heading in heading_path if heading]


def _is_source_block(block: _Block) -> bool:
    return block.text.lower().startswith(SOURCE_PREFIXES) and block.max_size <= SOURCE_MAX_SIZE


def _column(bbox: tuple[float, float, float, float], mid_x: float) -> str:
    if bbox[2] - bbox[0] > mid_x * 1.1:
        return "full"
    return "left" if (bbox[0] + bbox[2]) / 2 < mid_x else "right"


def _closest_owner(
    block: _Block,
    owners: list[_Owner],
    mid_x: float,
    max_gap: float,
    same_column: bool = True,
    slack: float = 4.0,
) -> _Owner | None:
    """Return the owner whose label is closest above the block (in the same column)."""
    block_column = _column(block.bbox, mid_x)
    best: tuple[float, _Owner] | None = None
    for owner in owners:
        gap = block.bbox[1] - owner.bbox[1]
        if gap < -slack or gap > max_gap:
            continue
        owner_column = _column(owner.bbox, mid_x)
        if same_column and "full" not in (owner_column, block_column) and owner_column != block_column:
            continue
        if best is None or abs(gap) < best[0]:
            best = (abs(gap), owner)
    return best[1] if best else None


def _assign_images(
    page: pymupdf.Page, images: list[DocumentImage], figures: list[_Owner], mid_x: float
) -> None:
    """Link each chart image to the figure label right above it."""
    for image in images:
        xref = image.metadata.get("xref")
        try:
            rects = page.get_image_rects(int(xref)) if xref is not None else []
        except Exception:
            rects = []
        for rect in rects:
            if rect.height < FIGURE_IMAGE_MIN_HEIGHT:
                continue
            fake_block = _Block((rect.x0, rect.y0, rect.x1, rect.y1), [])
            owner = _closest_owner(fake_block, figures, mid_x, IMAGE_MAX_GAP)
            if owner is not None and image.image_index not in owner.image_indexes:
                owner.image_indexes.append(image.image_index)


def _table_rows(cells: list[_Block]) -> list[str]:
    """Group cell blocks into rows by their vertical position."""
    rows: list[list[_Block]] = []
    for cell in sorted(cells, key=lambda block: ((block.bbox[1] + block.bbox[3]) / 2, block.bbox[0])):
        center = (cell.bbox[1] + cell.bbox[3]) / 2
        if rows and abs(center - (rows[-1][0].bbox[1] + rows[-1][0].bbox[3]) / 2) <= TABLE_ROW_TOLERANCE:
            rows[-1].append(cell)
        else:
            rows.append([cell])
    return [
        " | ".join(_normalize(cell.text) for cell in sorted(row, key=lambda block: block.bbox[0]))
        for row in rows
    ]


def _owner_segment(owner: _Owner) -> DocumentSegment:
    label = "Hình" if owner.kind == "figure" else "Bảng"
    caption = " ".join(owner.captions).strip()
    source = " ".join(owner.sources).strip()
    parts = [f"{label} {owner.number}: {caption}".rstrip(": ")]
    if owner.kind == "table":
        parts.extend(_table_rows(owner.cells))
    if source:
        parts.append(source)
    segment: DocumentSegment = {
        "element_type": owner.kind,
        "text": "\n".join(parts),
        "heading_path": owner.heading_path,
        "box_id": owner.box_id,
        "box_title": owner.box_title,
        "caption": caption,
        "data_source": _source_name(source),
        "image_indexes": owner.image_indexes,
        "bbox": list(owner.bbox),
    }
    segment["figure_no" if owner.kind == "figure" else "table_no"] = owner.number
    return segment


def _source_name(source: str) -> str:
    """Return the "Nguồn: ..." part without notes ("Ghi chú: ...")."""
    match = re.search(r"Nguồn\s*:\s*(.*?)(?:Ghi chú\s*:|$)", source, re.S)
    return match.group(1).strip().rstrip(".") if match else ""


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()
