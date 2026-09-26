import argparse
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from ..chunking import split_documents
from ..documents import load_pdf
from ..embeddings import (
    DEFAULT_EMBEDDING_BENCHMARK_ALIASES,
    EmbeddingService,
    resolve_embedding_model,
)
from ..paths import PROJECT_ROOT, default_pdf_path, default_questions_path, resolve_path
from ..retrieval import VectorStore
from .common import mean_metrics, metric_headers, print_table, require_inputs
from .metrics import evaluate_results, mean


def _configured_models() -> list[str]:
    configured = os.getenv("EMBEDDING_BENCHMARK_MODELS", "")
    if not configured.strip():
        return DEFAULT_EMBEDDING_BENCHMARK_ALIASES
    return [item.strip() for item in configured.split(",") if item.strip()]


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

    per_question: list[dict[str, float]] = []
    query_encode_seconds: list[float] = []
    retrieval_seconds: list[float] = []
    for question in questions:
        encode_started = time.perf_counter()
        query_embedding = embedder.embed_query(str(question["question"]))
        query_encode_seconds.append(time.perf_counter() - encode_started)

        retrieval_started = time.perf_counter()
        results = store.search(query_embedding, top_k=args.top_k)
        retrieval_seconds.append(time.perf_counter() - retrieval_started)
        per_question.append(evaluate_results(results, question, args.top_k))

    return {
        "model": spec.alias,
        "dimension": spec.dimension or int(document_embeddings.shape[1]),
        "chunks": len(chunks),
        "queries": len(questions),
        **mean_metrics(per_question, args.top_k),
        "load_s": load_seconds,
        "build_s": build_seconds,
        "query_ms": mean(query_encode_seconds) * 1000,
        "retrieval_ms": mean(retrieval_seconds) * 1000,
    }


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")

    parser = argparse.ArgumentParser(description="Benchmark embedding models")
    parser.add_argument("--pdf", type=Path, default=default_pdf_path())
    parser.add_argument("--questions", type=Path, default=default_questions_path())
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

    pdf_path = resolve_path(args.pdf)
    questions_path = resolve_path(args.questions)
    questions = require_inputs(pdf_path, questions_path)
    print(
        f"PDF: {pdf_path.name} | questions: {questions_path.name} ({len(questions)}) "
        f"| chunking: {args.chunking_strategy}"
    )

    pages = load_pdf(pdf_path)
    chunks = split_documents(
        pages,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        chunking_strategy=args.chunking_strategy,
        min_chunk_size=args.min_chunk_size,
    )

    rows = [_run_model(model, chunks, questions, args) for model in args.models]
    print_table(
        rows,
        ["model", "dimension", "chunks", "queries"]
        + metric_headers(args.top_k)
        + ["load_s", "build_s", "query_ms", "retrieval_ms"],
    )


if __name__ == "__main__":
    main()
