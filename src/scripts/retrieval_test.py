import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from ..chunking import split_documents
from ..documents import load_pdf
from ..embeddings import DEFAULT_EMBEDDING_MODEL, EmbeddingService
from ..retrieval import VectorStore


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _default_pdf_path() -> Path:
    default_path = PROJECT_ROOT / "data" / "document.pdf"
    if default_path.exists():
        return default_path

    pdf_files = sorted((PROJECT_ROOT / "data").glob("*.pdf"))
    if pdf_files:
        return pdf_files[0]

    return default_path


def _print_result(rank: int, result: dict[str, object]) -> None:
    print("=" * 80)
    print(f"Rank: {rank}")
    print(f"Similarity score: {float(result['similarity_score']):.4f}")
    print(f"Source: {result['source']}")
    print(f"Page: {result['page']}")
    print(f"Chunk: {result.get('chunk_index')} ({result.get('chunk_id', '')})")
    print(f"Strategy: {result.get('chunking_strategy', '')}")
    print(f"Token count: {result.get('token_count', 0)}")
    if result.get("parent_id"):
        print(f"Parent: {result.get('parent_id')}")
    print("-" * 80)
    print(result["text"])
    print()


def main() -> None:
    """Run a terminal retrieval test without calling an LLM."""
    load_dotenv(PROJECT_ROOT / ".env")

    parser = argparse.ArgumentParser(description="Test PDF retrieval without calling an LLM")
    parser.add_argument(
        "--pdf",
        type=Path,
        default=_default_pdf_path(),
        help="Path to the PDF file",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=int(os.getenv("TOP_K", "5")),
        help="Number of chunks to return",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=int(os.getenv("CHUNK_SIZE", "300")),
        help="Number of words per chunk",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=int(os.getenv("CHUNK_OVERLAP", "50")),
        help="Number of overlapping words between chunks",
    )
    parser.add_argument(
        "--chunking-strategy",
        default=os.getenv("CHUNKING_STRATEGY", "recursive"),
        choices=[
            "fixed_word",
            "sentence",
            "paragraph",
            "recursive",
            "semantic",
            "parent_child",
            "sliding_window",
        ],
        help="Chunking strategy to use",
    )
    parser.add_argument(
        "--min-chunk-size",
        type=int,
        default=int(os.getenv("MIN_CHUNK_SIZE", "50")),
        help="Minimum preferred chunk size for boundary-aware strategies",
    )
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        help="sentence-transformers model name",
    )
    parser.add_argument(
        "--embedding-document-prefix",
        default=os.getenv("EMBEDDING_DOCUMENT_PREFIX", ""),
        help="Prefix added to document chunks before embedding",
    )
    parser.add_argument(
        "--embedding-query-prefix",
        default=os.getenv("EMBEDDING_QUERY_PREFIX", ""),
        help="Prefix added to queries before embedding",
    )
    parser.add_argument(
        "--vector-index-type",
        default=os.getenv("VECTOR_INDEX_TYPE", "flat"),
        choices=["flat", "hnsw", "ivf"],
        help="FAISS index type to use",
    )
    parser.add_argument(
        "--faiss-hnsw-m",
        type=int,
        default=int(os.getenv("FAISS_HNSW_M", "32")),
        help="HNSW graph degree",
    )
    parser.add_argument(
        "--faiss-ivf-nlist",
        type=int,
        default=int(os.getenv("FAISS_IVF_NLIST", "64")),
        help="Number of IVF inverted lists",
    )
    parser.add_argument(
        "--faiss-ivf-nprobe",
        type=int,
        default=int(os.getenv("FAISS_IVF_NPROBE", "8")),
        help="Number of IVF lists searched per query",
    )
    args = parser.parse_args()

    print(f"Loading PDF: {args.pdf}")
    pages = load_pdf(args.pdf)
    print(f"Loaded {len(pages)} pages with text.")

    chunks = split_documents(
        pages,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        chunking_strategy=args.chunking_strategy,
        min_chunk_size=args.min_chunk_size,
    )
    print(f"Created {len(chunks)} chunks with {args.chunking_strategy} strategy.")

    print(f"Loading embedding model: {args.embedding_model}")
    embedder = EmbeddingService(
        args.embedding_model,
        document_prefix=args.embedding_document_prefix,
        query_prefix=args.embedding_query_prefix,
    )

    print("Creating document embeddings...")
    document_embeddings = embedder.embed_documents(chunks)

    print(f"Building FAISS {args.vector_index_type} index...")
    store = VectorStore(
        PROJECT_ROOT / "storage" / "retrieval_test",
        index_type=args.vector_index_type,
        hnsw_m=args.faiss_hnsw_m,
        ivf_nlist=args.faiss_ivf_nlist,
        ivf_nprobe=args.faiss_ivf_nprobe,
    )
    store.build(document_embeddings, chunks)

    question = input("\nEnter your question: ").strip()
    if not question:
        raise SystemExit("Question is empty.")

    query_embedding = embedder.embed_query(question)
    results = store.search(query_embedding, top_k=args.top_k)

    print(f"\nTop {len(results)} related chunks:\n")
    for rank, result in enumerate(results, start=1):
        _print_result(rank, result)


if __name__ == "__main__":
    main()
