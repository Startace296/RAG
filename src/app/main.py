import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from ..paths import DEFAULT_STORAGE_DIR, PROJECT_ROOT, default_pdf_path, resolve_path
from .rag_service import RagResponse, RagService


# Kept for callers that imported the old private helper.
_resolve_path = resolve_path


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
    elif service.store.exists():
        mismatches = service.index_mismatches()
        if not mismatches:
            if _is_verbose():
                print(f"Loaded existing FAISS index from: {storage_dir}")
            return True
        print("Saved index was built with different settings; rebuilding it:")
        for key, (saved, current) in mismatches.items():
            print(f"  {key}: saved={saved!r}, current={current!r}")
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
    # Vietnamese answers fail to print on Windows when output is redirected
    # and the console code page is not UTF-8.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    load_dotenv(PROJECT_ROOT / ".env")

    parser = argparse.ArgumentParser(description="Run a terminal RAG app for a PDF")
    parser.add_argument(
        "--pdf",
        type=Path,
        default=default_pdf_path(),
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

    pdf_path = resolve_path(args.pdf)
    storage_dir = resolve_path(args.storage_dir)

    if not pdf_path.exists():
        print(f"Error: PDF file not found: {pdf_path}")
        return
    if not pdf_path.is_file():
        print(f"Error: PDF path is not a file: {pdf_path}")
        return

    try:
        service = RagService.from_env(pdf_path=pdf_path, storage_dir=storage_dir)
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
