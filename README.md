# RAG Study Assistant

Minimal PDF question-answering application using sentence-transformers, FAISS, and a local Qwen3 model.

## Design Notes

- [Báo cáo tiến độ (BAO_CAO.md)](docs/BAO_CAO.md)
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

Put the source PDF in `data/`. With `PDF_PATH=data`, the first PDF in that folder is used. The first run will download the local models configured in `.env`.

Important settings in `.env`:

```env
EMBEDDING_MODEL=intfloat/multilingual-e5-small
EMBEDDING_DOCUMENT_PREFIX="passage: "
EMBEDDING_QUERY_PREFIX="query: "
CHUNKING_STRATEGY=parent_child
CHUNK_SIZE=300
CHUNK_OVERLAP=50
VECTOR_INDEX_TYPE=flat
TOP_K=5
SIMILARITY_THRESHOLD=0.795
MAX_CONTEXT_CHARS=12000
QWEN_DO_SAMPLE=false
IMAGE_MIN_SIDE=32
EVALUATION_QUESTIONS=evaluation/kinhtevn_questions.jsonl
```

`multilingual-e5-small` reads up to 512 tokens, so 300-word chunks are embedded in full. The MiniLM model reads only 128 tokens and cuts most chunks short. E5 models need the `passage: ` and `query: ` prefixes; keep the quotes so the trailing space is preserved.

`SIMILARITY_THRESHOLD` depends on the embedding model. E5 scores sit between about 0.77 and 0.92 even for off-topic questions, so 0.795 only removes clearly unrelated questions. With MiniLM use about 0.25.

`QWEN_TEMPERATURE`, `QWEN_TOP_P` and `QWEN_TOP_K` only take effect when `QWEN_DO_SAMPLE=true`.

The index records the source PDF, embedding model, prefixes, chunking settings and FAISS settings in `storage/index_config.json`. When any of them differs from `.env`, `python -m src.main` rebuilds the index automatically. Use `--rebuild` to force a rebuild anyway.

Embedded images smaller than `IMAGE_MIN_SIDE` pixels on either side, and repeated references to the same image on one page, are skipped. Images exported by an earlier run for the same document are removed before a new export.

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
python -m src.retrieval_test --pdf "data/KinhteVN.pdf" --top-k 5
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

Benchmarks read questions from `EVALUATION_QUESTIONS`, or from `--questions`. The question file must match the PDF. `evaluation/chunking_questions.jsonl` belongs to an older machine-learning document and does not match `KinhteVN.pdf`.

The benchmark tables report `hit@k`, `precision@k`, `recall@k`, `mrr`, and `ndcg@k`. A retrieved chunk counts as relevant when it is on an expected page and contains at least one expected keyword. When a question has only one kind of label, that label alone decides. `recall@k` is the share of expected pages covered by relevant chunks, and `ndcg@k` uses the number of expected pages as the ideal count.

Run the end-to-end evaluation, including the local LLM:

```powershell
python evaluation/run_kinhtevn_rag_eval.py --limit 5
```

## Tests

```powershell
python -m unittest discover -s tests -t .
```
