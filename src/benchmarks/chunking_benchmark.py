import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from ..chunking import SUPPORTED_CHUNKING_STRATEGIES, split_documents
from ..documents import load_pdf
from ..embeddings import DEFAULT_EMBEDDING_MODEL, EmbeddingService
from ..retrieval import VectorStore
from .metrics import evaluate_results, mean


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVALUATION_FILE = PROJECT_ROOT / "evaluation" / "chunking_questions.jsonl"


def _default_pdf_path() -> Path:
    configured_path = os.getenv("PDF_PATH")
    if configured_path:
        path = PROJECT_ROOT / configured_path
        if path.is_dir():
            pdf_files = sorted(path.glob("*.pdf"))
            if pdf_files:
                return pdf_files[0]
        return path

    pdf_files = sorted((PROJECT_ROOT / "data").glob("*.pdf"))
    if pdf_files:
        return pdf_files[0]
    return PROJECT_ROOT / "data" / "document.pdf"


def _load_questions(path: Path) -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        item = json.loads(line)
        if not item.get("question"):
            raise ValueError(f"Missing question at {path}:{line_number}")
        questions.append(item)
    return questions


def _run_strategy(
    strategy: str,
    pages: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    embedder: EmbeddingService,
    args: argparse.Namespace,
) -> dict[str, Any]:
    build_started = time.perf_counter()
    chunks = split_documents(
        pages,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        chunking_strategy=strategy,
        min_chunk_size=args.min_chunk_size,
    )
    document_embeddings = embedder.embed_documents(chunks)
    store = VectorStore(PROJECT_ROOT / "storage" / "chunking_benchmark" / strategy)
    store.build(document_embeddings, chunks)
    build_seconds = time.perf_counter() - build_started

    hits: list[float] = []
    precisions: list[float] = []
    recalls: list[float] = []
    mrrs: list[float] = []
    ndcgs: list[float] = []
    retrieval_seconds: list[float] = []

    for question in questions:
        query_embedding = embedder.embed_query(str(question["question"]))
        retrieval_started = time.perf_counter()
        results = store.search(query_embedding, top_k=args.top_k)
        retrieval_seconds.append(time.perf_counter() - retrieval_started)

        metrics = evaluate_results(results, question, args.top_k)
        hits.append(metrics["hit"])
        precisions.append(metrics["precision"])
        recalls.append(metrics["recall"])
        mrrs.append(metrics["mrr"])
        ndcgs.append(metrics["ndcg"])

    return {
        "strategy": strategy,
        "chunk_size": args.chunk_size,
        "overlap": args.chunk_overlap,
        "chunks": len(chunks),
        "queries": len(questions),
        f"hit@{args.top_k}": mean(hits),
        f"precision@{args.top_k}": mean(precisions),
        f"recall@{args.top_k}": mean(recalls),
        "mrr": mean(mrrs),
        f"ndcg@{args.top_k}": mean(ndcgs),
        "build_s": build_seconds,
        "retrieval_ms": mean(retrieval_seconds) * 1000,
    }


def _print_table(rows: list[dict[str, Any]], top_k: int) -> None:
    headers = [
        "strategy",
        "chunk_size",
        "overlap",
        "chunks",
        "queries",
        f"hit@{top_k}",
        f"precision@{top_k}",
        f"recall@{top_k}",
        "mrr",
        f"ndcg@{top_k}",
        "build_s",
        "retrieval_ms",
    ]

    formatted_rows: list[dict[str, str]] = []
    for row in rows:
        formatted_row: dict[str, str] = {}
        for header in headers:
            value = row[header]
            if isinstance(value, float):
                formatted_row[header] = f"{value:.4f}"
            else:
                formatted_row[header] = str(value)
        formatted_rows.append(formatted_row)

    widths = {
        header: max(
            len(header),
            *(len(row[header]) for row in formatted_rows),
        )
        for header in headers
    }

    def format_row(row: dict[str, str]) -> str:
        cells = []
        for header in headers:
            value = row[header]
            if header == "strategy":
                cells.append(value.ljust(widths[header]))
            else:
                cells.append(value.rjust(widths[header]))
        return "  ".join(cells)

    print(format_row({header: header for header in headers}))
    print("  ".join("-" * widths[header] for header in headers))
    for row in formatted_rows:
        print(format_row(row))


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")

    parser = argparse.ArgumentParser(description="Benchmark chunking strategies")
    parser.add_argument("--pdf", type=Path, default=_default_pdf_path())
    parser.add_argument("--questions", type=Path, default=DEFAULT_EVALUATION_FILE)
    parser.add_argument("--top-k", type=int, default=int(os.getenv("TOP_K", "5")))
    parser.add_argument("--chunk-size", type=int, default=int(os.getenv("CHUNK_SIZE", "300")))
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=int(os.getenv("CHUNK_OVERLAP", "50")),
    )
    parser.add_argument(
        "--min-chunk-size",
        type=int,
        default=int(os.getenv("MIN_CHUNK_SIZE", "50")),
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=[
            "fixed_word",
            "sentence",
            "paragraph",
            "recursive",
            "semantic",
            "parent_child",
            "sliding_window",
        ],
        choices=sorted(SUPPORTED_CHUNKING_STRATEGIES),
    )
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
    )
    args = parser.parse_args()

    if not args.pdf.exists():
        raise SystemExit(f"PDF not found: {args.pdf}")
    if not args.questions.exists():
        raise SystemExit(f"Question file not found: {args.questions}")

    questions = _load_questions(args.questions)
    if not questions:
        raise SystemExit(f"No benchmark questions found: {args.questions}")

    pages = load_pdf(args.pdf)
    embedder = EmbeddingService(
        args.embedding_model,
        document_prefix=os.getenv("EMBEDDING_DOCUMENT_PREFIX", ""),
        query_prefix=os.getenv("EMBEDDING_QUERY_PREFIX", ""),
    )

    rows = [
        _run_strategy(strategy, pages, questions, embedder, args)
        for strategy in args.strategies
    ]
    _print_table(rows, args.top_k)


if __name__ == "__main__":
    main()
