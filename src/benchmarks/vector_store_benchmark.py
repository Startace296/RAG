import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from ..chunking import split_documents
from ..documents import load_pdf
from ..embeddings import DEFAULT_EMBEDDING_MODEL, EmbeddingService
from ..retrieval import VectorStore
from .metrics import evaluate_results, mean


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVALUATION_FILE = PROJECT_ROOT / "evaluation" / "chunking_questions.jsonl"
DEFAULT_VECTOR_INDEXES = ["flat", "hnsw", "ivf"]


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


def _configured_indexes() -> list[str]:
    configured = os.getenv("VECTOR_BENCHMARK_INDEXES", "")
    if not configured.strip():
        return DEFAULT_VECTOR_INDEXES
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

    hits: list[float] = []
    precisions: list[float] = []
    recalls: list[float] = []
    mrrs: list[float] = []
    ndcgs: list[float] = []
    retrieval_seconds: list[float] = []

    for question, query_embedding in zip(questions, query_embeddings):
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
        "index_type": index_type,
        "chunks": len(chunks),
        "dimension": int(document_embeddings.shape[1]),
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
        "index_type",
        "chunks",
        "dimension",
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
            if header == "index_type":
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

    parser = argparse.ArgumentParser(description="Benchmark FAISS vector store indexes")
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
    _print_table(rows, args.top_k)


if __name__ == "__main__":
    main()
