from dataclasses import dataclass, field
from typing import Any, TypedDict


@dataclass(frozen=True)
class DocumentImage:
    """Image metadata found on a document page."""

    document_id: str
    page_number: int
    image_index: int
    image_path: str | None = None
    ocr_text: str | None = None
    caption: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DocumentPage:
    """Text, images, and metadata extracted from one source document page."""

    document_id: str
    page_number: int
    text: str
    source_name: str
    images: list[DocumentImage] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_chunk_input(self) -> "PageRecord":
        """Return the dict shape used by the chunking pipeline."""
        return {
            "document_id": self.document_id,
            "text": self.text,
            "page": self.page_number,
            "source": self.source_name,
            "metadata": self.metadata,
        }


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
    metadata: dict[str, Any] = field(default_factory=dict)
    pages: list[DocumentPage] = field(default_factory=list)

    def to_chunk_inputs(self) -> list["PageRecord"]:
        """Return page-level records compatible with the chunker."""
        return [page.to_chunk_input() for page in self.pages if page.text]


class PageRecord(TypedDict, total=False):
    """Backward-compatible page record consumed by the chunking pipeline."""

    text: str
    page: int
    source: str
    document_id: str
    metadata: dict[str, Any]
