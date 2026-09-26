# Source Structure

This project organizes `src` by the main layers of the RAG pipeline.

```text
src/
  app/          Application orchestration and CLI flow
  documents/    Document schemas and PDF ingestion
  chunking/     Text chunking strategies
  embeddings/   Embedding model configuration and encoding service
  retrieval/    FAISS vector store and search result metadata
  benchmarks/   Offline benchmark runners
  scripts/      Manual utility scripts
  paths.py      Shared project paths and default PDF / question file lookup
tests/          Unit tests (python -m unittest discover -s tests -t .)
```

## Layer Mapping

| Pipeline layer | Source package | Responsibility |
| --- | --- | --- |
| Document ingestion | `src.documents` | Read PDF files, extract page text, collect image metadata, export images when configured. |
| Document schema | `src.documents.schemas` | Define `Document`, `DocumentPage`, `DocumentImage`, and `PageRecord`. |
| Chunking | `src.chunking` | Split page records into traceable `TextChunk` objects. |
| Embedding | `src.embeddings` | Load sentence-transformers models and encode chunks/queries. |
| Retrieval | `src.retrieval` | Build/load/search the local FAISS index and store chunk metadata. |
| RAG orchestration | `src.app` | Connect ingestion, chunking, embedding, retrieval, and the local LLM. |
| Evaluation | `src.benchmarks` | Run chunking, embedding, and vector-store benchmark experiments. |
| Manual tools | `src.scripts` | Run utility workflows such as retrieval-only testing. |

## Compatibility Entry Points

The old top-level modules are kept as thin wrappers so existing commands still work:

```powershell
python -m src.main
python -m src.retrieval_test
python -m src.chunking_benchmark
python -m src.embedding_benchmark
python -m src.vector_store_benchmark
```

New code should import from the package that owns the layer, for example:

```python
from src.documents import load_pdf_document
from src.chunking import split_documents
from src.embeddings import EmbeddingService
from src.retrieval import VectorStore
```
