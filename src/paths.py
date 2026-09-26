import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_STORAGE_DIR = PROJECT_ROOT / "storage"
DEFAULT_EVALUATION_DIR = PROJECT_ROOT / "evaluation"
DEFAULT_QUESTIONS_FILE = DEFAULT_EVALUATION_DIR / "kinhtevn_questions.jsonl"


def resolve_path(path_value: str | Path) -> Path:
    """Resolve a relative path against the project root."""
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _first_pdf(directory: Path) -> Path | None:
    pdf_files = sorted(directory.glob("*.pdf")) if directory.is_dir() else []
    return pdf_files[0] if pdf_files else None


def default_pdf_path() -> Path:
    """Return the PDF configured by PDF_PATH, or the first PDF in data/.

    PDF_PATH may point to a file or to a directory. When it is a directory, the
    first PDF in alphabetical order is used.
    """
    configured_path = os.getenv("PDF_PATH")
    if configured_path:
        path = resolve_path(configured_path)
        if path.is_dir():
            return _first_pdf(path) or path / "document.pdf"
        return path

    fallback = DEFAULT_DATA_DIR / "document.pdf"
    if fallback.exists():
        return fallback
    return _first_pdf(DEFAULT_DATA_DIR) or fallback


def default_questions_path() -> Path:
    """Return the evaluation question file configured by EVALUATION_QUESTIONS."""
    configured_path = os.getenv("EVALUATION_QUESTIONS")
    if configured_path:
        return resolve_path(configured_path)
    return DEFAULT_QUESTIONS_FILE
