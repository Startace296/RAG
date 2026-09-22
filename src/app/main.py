import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from ..embeddings import DEFAULT_EMBEDDING_MODEL
from .rag_service import DEFAULT_LOCAL_LLM_MODEL, RagResponse, RagService


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STORAGE_DIR = PROJECT_ROOT / "storage"


def _resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _find_pdf_from_directory(directory: Path) -> Path | None:
    pdf_files = sorted(directory.glob("*.pdf"))
    return pdf_files[0] if pdf_files else None


def _default_pdf_path() -> Path:
    configured_path = os.getenv("PDF_PATH")
    if configured_path:
        path = _resolve_path(configured_path)
        if path.is_dir():
            pdf_file = _find_pdf_from_directory(path)
            if pdf_file:
                return pdf_file
        return path

    default_path = PROJECT_ROOT / "data" / "document.pdf"
    if default_path.exists():
        return default_path

    pdf_file = _find_pdf_from_directory(PROJECT_ROOT / "data")
    if pdf_file:
        return pdf_file

    return default_path


def _index_exists(storage_dir: Path) -> bool:
    return (storage_dir / "index.faiss").exists() and (storage_dir / "metadata.json").exists()


def _is_verbose() -> bool:
    return os.getenv("RAG_VERBOSE", "false").lower() == "true"


def _print_response(response: RagResponse) -> None:
    print()
    print(response["answer"])

    if response["sources"]:
        print("\nSources:")
        for source in response["sources"]:
            print(
                "- "
                f"{source['source']}, page {source['page']}, "
                f"chunk {source['chunk_index']}, "
                f"score {source['similarity_score']:.4f}"
            )
    print()


def _ensure_index(service: RagService, storage_dir: Path, rebuild: bool) -> bool:
    if rebuild:
        print("Preparing document index...")
    elif _index_exists(storage_dir):
        if _is_verbose():
            print(f"Loaded existing FAISS index from: {storage_dir}")
        return True
    else:
        print("Preparing document index...")

    try:
        service.rebuild_index()
    except Exception as exc:
        print(f"Error while building FAISS index: {exc}")
        return False

    if _is_verbose():
        print("FAISS index is ready.")
    return True


def _warm_up_llm(service: RagService) -> bool:
    print("Preparing local model...")
    try:
        service.warm_up()
    except Exception as exc:
        print(f"Error while loading local LLM: {exc}")
        return False

    if _is_verbose():
        print("Local LLM is ready.")
    return True


def _chat_loop(service: RagService) -> None:
    print("\nRAG console is ready.")
    print('Type a question and press Enter. Type "exit" to stop.\n')

    while True:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return

        if question.lower() == "exit":
            print("Goodbye.")
            return
        if not question:
            continue

        try:
            response = service.answer(question)
        except Exception as exc:
            print(f"Error while answering: {exc}\n")
            continue

        _print_response(response)


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")

    parser = argparse.ArgumentParser(description="Run a terminal RAG app for a PDF")
    parser.add_argument(
        "--pdf",
        type=Path,
        default=_default_pdf_path(),
        help="Path to a PDF file, or use PDF_PATH in .env",
    )
    parser.add_argument(
        "--storage-dir",
        type=Path,
        default=DEFAULT_STORAGE_DIR,
        help="Directory containing FAISS index and metadata",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force ingestion even if a FAISS index already exists",
    )
    args = parser.parse_args()

    pdf_path = _resolve_path(args.pdf)
    storage_dir = _resolve_path(args.storage_dir)

    if not pdf_path.exists():
        print(f"Error: PDF file not found: {pdf_path}")
        return
    if not pdf_path.is_file():
        print(f"Error: PDF path is not a file: {pdf_path}")
        return

    try:
        service = RagService(
            pdf_path=pdf_path,
            storage_dir=storage_dir,
            embedding_model=os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
            embedding_document_prefix=os.getenv("EMBEDDING_DOCUMENT_PREFIX", ""),
            embedding_query_prefix=os.getenv("EMBEDDING_QUERY_PREFIX", ""),
            top_k=int(os.getenv("TOP_K", "5")),
            chunk_size=int(os.getenv("CHUNK_SIZE", "300")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "50")),
            chunking_strategy=os.getenv("CHUNKING_STRATEGY", "recursive"),
            min_chunk_size=int(os.getenv("MIN_CHUNK_SIZE", "50")),
            vector_index_type=os.getenv("VECTOR_INDEX_TYPE", "flat"),
            faiss_hnsw_m=int(os.getenv("FAISS_HNSW_M", "32")),
            faiss_ivf_nlist=int(os.getenv("FAISS_IVF_NLIST", "64")),
            faiss_ivf_nprobe=int(os.getenv("FAISS_IVF_NPROBE", "8")),
            similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.25")),
            max_context_chars=int(os.getenv("MAX_CONTEXT_CHARS", "12000")),
            llm_model=os.getenv("LOCAL_LLM_MODEL", DEFAULT_LOCAL_LLM_MODEL),
        )
    except Exception as exc:
        print(f"Error while initializing RAG service: {exc}")
        return

    if _is_verbose():
        print(f"PDF: {pdf_path}")
        print(f"Storage: {storage_dir}")

    if not _ensure_index(service, storage_dir, args.rebuild):
        return

    if not _warm_up_llm(service):
        return

    _chat_loop(service)


if __name__ == "__main__":
    main()
