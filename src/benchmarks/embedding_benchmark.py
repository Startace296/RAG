import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from ..chunking import split_documents
from ..documents import load_pdf
from ..embeddings import (
    DEFAULT_EMBEDDING_BENCHMARK_ALIASES,
    resolve_embedding_model,
)
from ..embeddings import EmbeddingService
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


def _configured_models() -> list[str]:
    configured = os.getenv("EMBEDDING_BENCHMARK_MODELS", "")
    if not configured.strip():
        return DEFAULT_EMBEDDING_BENCHMARK_ALIASES
    return [item.strip() for item in configured.split(",") if item.strip()]


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


def _safe_model_dir_name(alias: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in alias)


def _run_model(
    model_value: str,
    chunks: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    spec = resolve_embedding_model(model_value)

    load_started = time.perf_counter()
    embedder = EmbeddingService(
        spec.model_name,
        document_prefix=spec.document_prefix,
        query_prefix=spec.query_prefix,
    )
    load_seconds = time.perf_counter() - load_started

    build_started = time.perf_counter()
    document_embeddings = embedder.embed_documents(chunks)
    store = VectorStore(
        PROJECT_ROOT / "storage" / "embedding_benchmark" / _safe_model_dir_name(spec.alias)
    )
    store.build(document_embeddings, chunks)
    build_seconds = time.perf_counter() - build_started

    hits: list[float] = []
    precisions: list[float] = []
    recalls: list[float] = []
    mrrs: list[float] = []
    ndcgs: list[float] = []
    query_encode_seconds: list[float] = []
    retrieval_seconds: list[float] = []

    for question in questions:
        encode_started = time.perf_counter()
        query_embedding = embedder.embed_query(str(question["question"]))
        query_encode_seconds.append(time.perf_counter() - encode_started)

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
        "model": spec.alias,
        "dimension": spec.dimension or int(document_embeddings.shape[1]),
        "chunks": len(chunks),
        "queries": len(questions),
        f"hit@{args.top_k}": mean(hits),
        f"precision@{args.top_k}": mean(precisions),
        f"recall@{args.top_k}": mean(recalls),
        "mrr": mean(mrrs),
        f"ndcg@{args.top_k}": mean(ndcgs),
        "load_s": load_seconds,
        "build_s": build_seconds,
        "query_ms": mean(query_encode_seconds) * 1000,
        "retrieval_ms": mean(retrieval_seconds) * 1000,
    }


def _print_table(rows: list[dict[str, Any]], top_k: int) -> None:
    headers = [
        "model",
        "dimension",
        "chunks",
        "queries",
        f"hit@{top_k}",
        f"precision@{top_k}",
        f"recall@{top_k}",
        "mrr",
        f"ndcg@{top_k}",
        "load_s",
        "build_s",
        "query_ms",
        "retrieval_ms",
    ]

    formatted_rows: list[dict[str, str]] = []
    for row in rows:
        formatted_row: dict[str, str] = {}
        for header in headers:
            value = row[header]
            formatted_row[header] = f"{value:.4f}" if isinstance(value, float) else str(value)
        formatted_rows.append(formatted_row)

    widths = {
        header: max(len(header), *(len(row[header]) for row in formatted_rows))
        for header in headers
    }

    def format_row(row: dict[str, str]) -> str:
        cells = []
        for header in headers:
            value = row[header]
            if header == "model":
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

    parser = argparse.ArgumentParser(description="Benchmark embedding models")
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
        "--chunking-strategy",
        default=os.getenv("CHUNKING_STRATEGY", "recursive"),
    )
    parser.add_argument(
        "--min-chunk-size",
        type=int,
        default=int(os.getenv("MIN_CHUNK_SIZE", "50")),
    )
    parser.add_argument("--models", nargs="+", default=_configured_models())
    args = parser.parse_args()

    if not args.pdf.exists():
        raise SystemExit(f"PDF not found: {args.pdf}")
    if not args.questions.exists():
        raise SystemExit(f"Question file not found: {args.questions}")

    questions = _load_questions(args.questions)
    if not questions:
        raise SystemExit(f"No benchmark questions found: {args.questions}")

    pages = load_pdf(args.pdf)
    chunks = split_documents(
        pages,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        chunking_strategy=args.chunking_strategy,
        min_chunk_size=args.min_chunk_size,
    )

    rows = [_run_model(model, chunks, questions, args) for model in args.models]
    _print_table(rows, args.top_k)


if __name__ == "__main__":
    main()
