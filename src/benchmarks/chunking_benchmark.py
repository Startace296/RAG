import argparse
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from ..chunking import SUPPORTED_CHUNKING_STRATEGIES, split_documents
from ..documents import load_pdf
from ..embeddings import DEFAULT_EMBEDDING_MODEL, EmbeddingService
from ..paths import PROJECT_ROOT, default_pdf_path, default_questions_path, resolve_path
from ..retrieval import VectorStore
from .common import mean_metrics, metric_headers, print_table, require_inputs
from .metrics import evaluate_results, mean


DEFAULT_STRATEGIES = [
    "fixed_word",
    "sentence",
    "paragraph",
    "recursive",
    "semantic",
    "parent_child",
    "sliding_window",
]


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

    per_question: list[dict[str, float]] = []
    retrieval_seconds: list[float] = []
    for question in questions:
        query_embedding = embedder.embed_query(str(question["question"]))
        retrieval_started = time.perf_counter()
        results = store.search(query_embedding, top_k=args.top_k)
        retrieval_seconds.append(time.perf_counter() - retrieval_started)
        per_question.append(evaluate_results(results, question, args.top_k))

    return {
        "strategy": strategy,
        "chunk_size": args.chunk_size,
        "overlap": args.chunk_overlap,
        "chunks": len(chunks),
        "queries": len(questions),
        **mean_metrics(per_question, args.top_k),
        "build_s": build_seconds,
        "retrieval_ms": mean(retrieval_seconds) * 1000,
    }


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")

    parser = argparse.ArgumentParser(description="Benchmark chunking strategies")
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
        "--min-chunk-size",
        type=int,
        default=int(os.getenv("MIN_CHUNK_SIZE", "50")),
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=DEFAULT_STRATEGIES,
        choices=sorted(SUPPORTED_CHUNKING_STRATEGIES),
    )
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
    )
    args = parser.parse_args()

    pdf_path = resolve_path(args.pdf)
    questions_path = resolve_path(args.questions)
    questions = require_inputs(pdf_path, questions_path)
    print(f"PDF: {pdf_path.name} | questions: {questions_path.name} ({len(questions)})")

    pages = load_pdf(pdf_path)
    embedder = EmbeddingService(
        args.embedding_model,
        document_prefix=os.getenv("EMBEDDING_DOCUMENT_PREFIX", ""),
        query_prefix=os.getenv("EMBEDDING_QUERY_PREFIX", ""),
    )

    rows = [
        _run_strategy(strategy, pages, questions, embedder, args)
        for strategy in args.strategies
    ]
    print_table(
        rows,
        ["strategy", "chunk_size", "overlap", "chunks", "queries"]
        + metric_headers(args.top_k)
        + ["build_s", "retrieval_ms"],
    )


if __name__ == "__main__":
    main()
