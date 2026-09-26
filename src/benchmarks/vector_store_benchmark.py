import argparse
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from ..chunking import split_documents
from ..documents import load_pdf
from ..embeddings import DEFAULT_EMBEDDING_MODEL, EmbeddingService
from ..paths import PROJECT_ROOT, default_pdf_path, default_questions_path, resolve_path
from ..retrieval import VectorStore
from .common import mean_metrics, metric_headers, print_table, require_inputs
from .metrics import evaluate_results, mean


DEFAULT_VECTOR_INDEXES = ["flat", "hnsw", "ivf"]


def _configured_indexes() -> list[str]:
    configured = os.getenv("VECTOR_BENCHMARK_INDEXES", "")
    if not configured.strip():
        return DEFAULT_VECTOR_INDEXES
    return [item.strip() for item in configured.split(",") if item.strip()]


def _run_index(
    index_type: str,
    document_embeddings: Any,
    chunks: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    query_embeddings: list[Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    build_started = time.perf_counter()
    store = VectorStore(
        PROJECT_ROOT / "storage" / "vector_store_benchmark" / index_type,
        index_type=index_type,
        hnsw_m=args.faiss_hnsw_m,
        ivf_nlist=args.faiss_ivf_nlist,
        ivf_nprobe=args.faiss_ivf_nprobe,
    )
    store.build(document_embeddings, chunks)
    build_seconds = time.perf_counter() - build_started

    per_question: list[dict[str, float]] = []
    retrieval_seconds: list[float] = []
    for question, query_embedding in zip(questions, query_embeddings):
        retrieval_started = time.perf_counter()
        results = store.search(query_embedding, top_k=args.top_k)
        retrieval_seconds.append(time.perf_counter() - retrieval_started)
        per_question.append(evaluate_results(results, question, args.top_k))

    return {
        "index_type": index_type,
        "chunks": len(chunks),
        "dimension": int(document_embeddings.shape[1]),
        "queries": len(questions),
        **mean_metrics(per_question, args.top_k),
        "build_s": build_seconds,
        "retrieval_ms": mean(retrieval_seconds) * 1000,
    }


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")

    parser = argparse.ArgumentParser(description="Benchmark FAISS vector store indexes")
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
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
    )
    parser.add_argument(
        "--embedding-document-prefix",
        default=os.getenv("EMBEDDING_DOCUMENT_PREFIX", ""),
    )
    parser.add_argument(
        "--embedding-query-prefix",
        default=os.getenv("EMBEDDING_QUERY_PREFIX", ""),
    )
    parser.add_argument(
        "--indexes",
        nargs="+",
        default=_configured_indexes(),
        choices=sorted(VectorStore.SUPPORTED_INDEX_TYPES),
    )
    parser.add_argument(
        "--faiss-hnsw-m",
        type=int,
        default=int(os.getenv("FAISS_HNSW_M", "32")),
    )
    parser.add_argument(
        "--faiss-ivf-nlist",
        type=int,
        default=int(os.getenv("FAISS_IVF_NLIST", "64")),
    )
    parser.add_argument(
        "--faiss-ivf-nprobe",
        type=int,
        default=int(os.getenv("FAISS_IVF_NPROBE", "8")),
    )
    args = parser.parse_args()

    pdf_path = resolve_path(args.pdf)
    questions_path = resolve_path(args.questions)
    questions = require_inputs(pdf_path, questions_path)
    print(
        f"PDF: {pdf_path.name} | questions: {questions_path.name} ({len(questions)}) "
        f"| chunking: {args.chunking_strategy} | model: {args.embedding_model}"
    )

    pages = load_pdf(pdf_path)
    chunks = split_documents(
        pages,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        chunking_strategy=args.chunking_strategy,
        min_chunk_size=args.min_chunk_size,
    )
    embedder = EmbeddingService(
        args.embedding_model,
        document_prefix=args.embedding_document_prefix,
        query_prefix=args.embedding_query_prefix,
    )
    document_embeddings = embedder.embed_documents(chunks)
    query_embeddings = [
        embedder.embed_query(str(question["question"]))
        for question in questions
    ]

    rows = [
        _run_index(index_type, document_embeddings, chunks, questions, query_embeddings, args)
        for index_type in args.indexes
    ]
    print_table(
        rows,
        ["index_type", "chunks", "dimension", "queries"]
        + metric_headers(args.top_k)
        + ["build_s", "retrieval_ms"],
    )


if __name__ == "__main__":
    main()
