# RAG Study Assistant

Minimal PDF question-answering application using sentence-transformers, FAISS, and a local Qwen3 model.

## Design Notes

- [Document structure and ingestion flow](docs/document_structure.md)
- [Source package structure](docs/source_structure.md)
- [Image handling in documents](docs/image_handling.md)
- [Chunking strategies and issues](docs/chunking.md)
- [Embedding models and benchmark](docs/embedding.md)
- [Vector Store indexes and benchmark](docs/vector_store.md)

## Setup

```powershell
cd "rag"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Put the source PDF at `data/document.pdf`. The first run will download the local models configured in `.env`.

Important settings in `.env`:

```env
CHUNKING_STRATEGY=recursive
CHUNK_SIZE=300
CHUNK_OVERLAP=50
VECTOR_INDEX_TYPE=flat
TOP_K=5
SIMILARITY_THRESHOLD=0.25
MAX_CONTEXT_CHARS=12000
QWEN_DO_SAMPLE=false
```

After changing the embedding model, chunking settings, or vector index settings, rebuild the index with `--rebuild`.

## Run

Build or rebuild the local FAISS index:

```powershell
python -m src.main --rebuild
```

Ask a question:

```powershell
python -m src.main "Tóm tắt nội dung chính của tài liệu"
```

The generated FAISS index and metadata are saved under `storage/`.

## Test retrieval only

Run the retriever without calling an LLM:

```powershell
python -m src.retrieval_test
```

Optional arguments:

```powershell
python -m src.retrieval_test --pdf "data/document.pdf" --top-k 5
```

Test a specific chunking strategy:

```powershell
python -m src.retrieval_test --chunking-strategy recursive
```

Run chunking benchmark:

```powershell
python -m src.chunking_benchmark
```

Run embedding benchmark:

```powershell
python -m src.embedding_benchmark
```

Run vector store benchmark:

```powershell
python -m src.vector_store_benchmark
```

Recommended benchmark order:

```powershell
python -m src.chunking_benchmark
python -m src.embedding_benchmark
python -m src.vector_store_benchmark
```

The benchmark tables report `hit@k`, `precision@k`, `recall@k`, `mrr`, and `ndcg@k`. `hit@k` checks whether at least one relevant result appears in the top-k, while `recall@k` uses the expected pages when they are available.
