from dataclasses import dataclass, field
from typing import Any, TypedDict


class DocumentMetadata(TypedDict, total=False):
    """Metadata extracted from the source document."""

    page_count: int
    image_count: int
    skipped_image_count: int
    image_output_dir: str | None
    file_size_bytes: int
    document_id_strategy: str
    content_hash: str


class DocumentImageMetadata(TypedDict, total=False):
    """Technical metadata for an image found in a document page."""

    xref: int
    width: int
    height: int
    bits_per_component: int
    colorspace: str
    name: str
    filter: str
    extraction_status: str
    extraction_error: str
    extension: str
    file_size_bytes: int
    ocr_status: str
    ocr_error: str
    ocr_char_count: int
    ocr_word_count: int


class DocumentImageRecord(TypedDict, total=False):
    """JSON-friendly image record carried into chunking."""

    document_id: str
    page_number: int
    image_index: int
    image_path: str | None
    ocr_text: str | None
    caption: str | None
    metadata: DocumentImageMetadata


class PageLayoutBlock(TypedDict, total=False):
    """Compact layout block metadata extracted from a PDF page."""

    block_index: int
    block_type: str
    bbox: list[float]
    line_count: int
    span_count: int
    text_preview: str


class PageTableMetadata(TypedDict, total=False):
    """Compact table metadata extracted from a PDF page."""

    table_index: int
    bbox: list[float]
    row_count: int
    column_count: int


class PageMetadata(TypedDict, total=False):
    """Page-level metadata used by chunking and diagnostics."""

    char_count: int
    word_count: int
    searchable_char_count: int
    searchable_word_count: int
    image_count: int
    skipped_image_count: int
    image_text_count: int
    page_width: float
    page_height: float
    page_rotation: int
    section_title: str
    block_count: int
    text_block_count: int
    image_block_count: int
    table_count: int
    layout_blocks: list[PageLayoutBlock]
    tables: list[PageTableMetadata]


@dataclass(frozen=True)
class DocumentImage:
    """Image metadata found on a document page."""

    document_id: str
    page_number: int
    image_index: int
    image_path: str | None = None
    ocr_text: str | None = None
    caption: str | None = None
    metadata: DocumentImageMetadata = field(default_factory=dict)

    def to_chunk_text(self) -> str:
        """Return searchable text extracted from the image."""
        parts: list[str] = []
        if self.ocr_text:
            parts.append(f"[Image {self.image_index + 1} OCR]\n{self.ocr_text.strip()}")
        if self.caption:
            parts.append(f"[Image {self.image_index + 1} Caption]\n{self.caption.strip()}")
        return "\n\n".join(parts)

    def to_record(self) -> DocumentImageRecord:
        """Return a JSON-friendly image metadata record."""
        return {
            "document_id": self.document_id,
            "page_number": self.page_number,
            "image_index": self.image_index,
            "image_path": self.image_path,
            "ocr_text": self.ocr_text,
            "caption": self.caption,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class DocumentPage:
    """Text, images, and metadata extracted from one source document page."""

    document_id: str
    page_number: int
    text: str
    source_name: str
    images: list[DocumentImage] = field(default_factory=list)
    metadata: PageMetadata = field(default_factory=dict)

    def to_chunk_input(self) -> "PageRecord":
        """Return the dict shape used by the chunking pipeline."""
        return {
            "document_id": self.document_id,
            "text": self.chunk_text(),
            "page": self.page_number,
            "source": self.source_name,
            "metadata": self.metadata,
            "images": [image.to_record() for image in self.images],
        }

    def chunk_text(self) -> str:
        """Return page text plus searchable image text."""
        image_text = self.image_chunk_text()
        return "\n\n".join(part for part in [self.text, image_text] if part)

    def image_chunk_text(self) -> str:
        """Return searchable OCR/caption text for all images on the page."""
        return "\n\n".join(
            image_text
            for image in self.images
            if (image_text := image.to_chunk_text())
        )


@dataclass(frozen=True)
class Document:
    """A source document before it is split into chunks."""

    document_id: str
    source_path: str
    source_name: str
    document_type: str
    title: str | None = None
    author: str | None = None
    created_at: str | None = None
    metadata: DocumentMetadata | dict[str, Any] = field(default_factory=dict)
    pages: list[DocumentPage] = field(default_factory=list)

    def to_chunk_inputs(self) -> list["PageRecord"]:
        """Return page-level records compatible with the chunker."""
        return [page.to_chunk_input() for page in self.pages if page.chunk_text()]


class PageRecord(TypedDict, total=False):
    """Backward-compatible page record consumed by the chunking pipeline."""

    text: str
    page: int
    source: str
    document_id: str
    metadata: PageMetadata
    images: list[DocumentImageRecord]
