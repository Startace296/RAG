import os
from hashlib import sha256
from pathlib import Path
from typing import Any

import pymupdf

from .schemas import Document, DocumentImage, DocumentPage, PageRecord


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGE_OUTPUT_DIR = PROJECT_ROOT / "storage" / "images"


def _build_document_id(path: Path) -> str:
    """Create a stable ID from file identity information."""
    stat = path.stat()
    raw_id = f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
    return sha256(raw_id.encode("utf-8")).hexdigest()[:16]


def _clean_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in metadata.items()
        if value not in (None, "")
    }


def _should_export_images() -> bool:
    return os.getenv("DOCUMENT_IMAGE_EXPORT", "true").lower() == "true"


def _resolve_image_output_dir(image_output_dir: str | Path | None) -> Path | None:
    if image_output_dir is None:
        if not _should_export_images():
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

        images.append(
            DocumentImage(
                document_id=document_id,
                page_number=page_number,
                image_index=image_index,
                image_path=image_path,
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
        document_id = _build_document_id(path)
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
                images = _extract_images(
                    pdf=pdf,
                    page=page,
                    document_id=document_id,
                    page_number=page_number,
                    output_dir=document_image_output_dir,
                )
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
                                "image_count": len(images),
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
