# Source Structure

`src` is organized by the layers of the RAG pipeline. Each layer is a package
whose `__init__.py` exposes its public API; other layers import only from that
package, never from a sibling layer's internal module.

```text
src/
  core/         Shared infrastructure: project paths, .env loading, verbose/quiet helpers
  documents/    Layer 1 - Document schemas and PDF ingestion
  chunking/     Layer 2 - Text chunking strategies
  embeddings/   Layer 3 - Embedding model registry and encoding service
  retrieval/    Layer 4 - FAISS vector store, BM25 keyword index, cross-encoder reranker
  generation/   Layer 5 - Answer prompt and local LLM (Qwen)
  app/          Layer 6 - RAG orchestration (RagService) and terminal CLI
  benchmarks/   Offline benchmark runners for chunking, embedding, vector store
  scripts/      Manual utilities (retrieval-only test)
tests/          Unit tests (python -m unittest discover -s tests -t .)
evaluation/     Question sets, answer checks, RAG evaluation runner, results
```

## Layer Mapping

| Pipeline layer | Package | Main modules | Responsibility |
| --- | --- | --- | --- |
| Shared | `src.core` | `paths.py`, `runtime.py` | Project root, default data/storage/question paths, `load_env()`, `is_verbose()`, `quiet_output()`. |
| Document ingestion | `src.documents` | `pdf_loader.py`, `schemas.py` | Read PDFs, extract page text/layout, collect and export image metadata. |
| Chunking | `src.chunking` | `strategies.py` | Split page records into traceable `TextChunk` objects. |
| Embedding | `src.embeddings` | `models.py`, `service.py` | Model aliases/prefixes and sentence-transformers encoding. |
| Retrieval | `src.retrieval` | `vector_store.py`, `bm25.py`, `reranker.py` | Build, save, load, and search the FAISS index with chunk metadata; BM25 keyword ranks; lazy-loaded cross-encoder reranker. |
| Generation | `src.generation` | `prompts.py`, `local_llm.py` | Vietnamese answer prompt, `NO_CONTEXT_ANSWER`, lazy-loaded `LocalLLM`. |
| Orchestration | `src.app` | `rag_service.py`, `main.py` | `RagService` connects layers 1-5; `main.py` is the interactive CLI. |
| Evaluation | `src.benchmarks` | `*_benchmark.py`, `metrics.py`, `common.py` | Retrieval benchmark experiments and metrics. |
| Manual tools | `src.scripts` | `retrieval_test.py` | Retrieval-only testing without an LLM. |

Dependency rules:

- `documents`, `chunking`, `embeddings`, `retrieval`, `generation` depend at most on `core` and never on each other.
- Only `app`, `benchmarks` and `scripts` combine several layers.
- `core` depends on no other package in `src`.

## Entry Points

```powershell
python -m src.app.main
python -m src.scripts.retrieval_test
python -m src.benchmarks.chunking_benchmark
python -m src.benchmarks.embedding_benchmark
python -m src.benchmarks.vector_store_benchmark
python evaluation/run_kinhtevn_rag_eval.py
python evaluation/run_retrieval_eval.py
```

The old top-level wrapper modules (`src.main`, `src.retrieval_test`,
`src.rag_service`, `src.vector_store`, ...) were removed. Import from the
package that owns the layer:

```python
from src.core import DEFAULT_STORAGE_DIR, default_pdf_path
from src.documents import load_pdf_document
from src.chunking import split_documents
from src.embeddings import EmbeddingService
from src.retrieval import VectorStore
from src.generation import LocalLLM, build_answer_prompt
from src.app import RagService
```
