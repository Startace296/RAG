import os
import re
from hashlib import sha256
from pathlib import Path
from typing import Any, TypedDict

from ..chunking import split_documents
from ..core import env_flag, load_env, resolve_path
from ..documents import DOCUMENT_PARSER_VERSION, load_pdf
from ..embeddings import DEFAULT_EMBEDDING_MODEL, EmbeddingService
from ..generation import (
    DEFAULT_LOCAL_LLM_MODEL,
    NO_CONTEXT_ANSWER,
    SYSTEM_PROMPT,
    LocalLLM,
    build_answer_prompt,
)
from ..retrieval import CrossEncoderReranker, SearchResult, VectorStore


RETRIEVAL_MODES = {"dense", "hybrid"}


class Source(TypedDict):
    """Source metadata returned with the final answer."""

    chunk_id: str
    source: str
    page: int
    page_chunk_index: int
    document_chunk_index: int
    chunk_index: int
    similarity_score: float
    document_id: str
    word_count: int
    token_count: int
    section_title: str
    chunking_strategy: str
    parent_id: str
    image_refs: list[dict[str, Any]]


class RagResponse(TypedDict):
    """Final RAG response."""

    answer: str
    sources: list[Source]


def _hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _model_name_or_path(value: str) -> str:
    """Resolve a local model folder against the project root; keep Hub names as they are."""
    value = value.strip()
    if value and resolve_path(value).is_dir():
        return str(resolve_path(value))
    return value


class RagService:
    """Orchestrate the RAG layers: ingestion, chunking, embedding, retrieval, generation."""

    def __init__(
        self,
        pdf_path: str | Path,
        storage_dir: str | Path,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        embedding_document_prefix: str = "",
        embedding_query_prefix: str = "",
        top_k: int = 5,
        chunk_size: int = 300,
        chunk_overlap: int = 50,
        chunking_strategy: str = "recursive",
        min_chunk_size: int = 50,
        vector_index_type: str = "flat",
        faiss_hnsw_m: int = 32,
        faiss_ivf_nlist: int = 64,
        faiss_ivf_nprobe: int = 8,
        similarity_threshold: float = 0.25,
        max_context_chars: int = 12000,
        llm_model: str = DEFAULT_LOCAL_LLM_MODEL,
        retrieval_mode: str = "hybrid",
        hybrid_candidates: int = 10,
        hybrid_rrf_k: int = 60,
        reranker_model: str = "",
        reranker_device: str = "cpu",
        strip_boilerplate: bool = True,
    ):
        load_env()

        self.pdf_path = Path(pdf_path)
        self.embedding_model = embedding_model
        self.embedding_document_prefix = embedding_document_prefix
        self.embedding_query_prefix = embedding_query_prefix
        self.top_k = top_k
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunking_strategy = chunking_strategy
        self.min_chunk_size = min_chunk_size
        self.similarity_threshold = similarity_threshold
        self.max_context_chars = max_context_chars
        self.retrieval_mode = retrieval_mode.strip().lower()
        if self.retrieval_mode not in RETRIEVAL_MODES:
            supported = ", ".join(sorted(RETRIEVAL_MODES))
            raise ValueError(f"Unsupported retrieval mode: {retrieval_mode}. Use one of: {supported}")
        self.hybrid_candidates = max(top_k, hybrid_candidates)
        self.hybrid_rrf_k = hybrid_rrf_k
        # An empty model name turns the cross-encoder re-ranking step off.
        self.reranker = (
            CrossEncoderReranker(reranker_model.strip(), device=reranker_device)
            if reranker_model.strip()
            else None
        )
        self.strip_boilerplate = strip_boilerplate
        self.embedder = EmbeddingService(
            embedding_model,
            document_prefix=embedding_document_prefix,
            query_prefix=embedding_query_prefix,
        )
        self.store = VectorStore(
            storage_dir,
            index_type=vector_index_type,
            hnsw_m=faiss_hnsw_m,
            ivf_nlist=faiss_ivf_nlist,
            ivf_nprobe=faiss_ivf_nprobe,
            pipeline_config=self._pipeline_config(),
        )
        self.llm = LocalLLM(llm_model)

    @classmethod
    def from_env(cls, pdf_path: str | Path, storage_dir: str | Path) -> "RagService":
        """Create a service using the settings in ``.env`` and the environment."""
        load_env()
        return cls(
            pdf_path=pdf_path,
            storage_dir=storage_dir,
            embedding_model=os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
            embedding_document_prefix=os.getenv("EMBEDDING_DOCUMENT_PREFIX", ""),
            embedding_query_prefix=os.getenv("EMBEDDING_QUERY_PREFIX", ""),
            top_k=int(os.getenv("TOP_K", "5")),
            chunk_size=int(os.getenv("CHUNK_SIZE", "300")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "50")),
            chunking_strategy=os.getenv("CHUNKING_STRATEGY", "recursive"),
            min_chunk_size=int(os.getenv("MIN_CHUNK_SIZE", "50")),
            vector_index_type=os.getenv("VECTOR_INDEX_TYPE", "flat"),
            faiss_hnsw_m=int(os.getenv("FAISS_HNSW_M", "32")),
            faiss_ivf_nlist=int(os.getenv("FAISS_IVF_NLIST", "64")),
            faiss_ivf_nprobe=int(os.getenv("FAISS_IVF_NPROBE", "8")),
            similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.25")),
            max_context_chars=int(os.getenv("MAX_CONTEXT_CHARS", "12000")),
            llm_model=os.getenv("LOCAL_LLM_MODEL", DEFAULT_LOCAL_LLM_MODEL),
            retrieval_mode=os.getenv("RETRIEVAL_MODE", "hybrid"),
            hybrid_candidates=int(os.getenv("HYBRID_CANDIDATES", "10")),
            hybrid_rrf_k=int(os.getenv("HYBRID_RRF_K", "60")),
            reranker_model=_model_name_or_path(os.getenv("RERANKER_MODEL", "")),
            reranker_device=os.getenv("RERANKER_DEVICE", "cpu"),
            strip_boilerplate=env_flag("DOCUMENT_STRIP_BOILERPLATE", True),
        )

    def index_mismatches(self) -> dict[str, tuple[Any, Any]]:
        """Return settings that differ between the saved index and this service."""
        return self.store.config_mismatches()

    def _pipeline_config(self) -> dict[str, Any]:
        return {
            "source_name": self.pdf_path.name,
            "source_sha256": _hash_file(self.pdf_path) if self.pdf_path.is_file() else "",
            "embedding_model": self.embedding_model,
            "embedding_document_prefix": self.embedding_document_prefix,
            "embedding_query_prefix": self.embedding_query_prefix,
            "chunking_strategy": self.chunking_strategy.strip().lower(),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "min_chunk_size": self.min_chunk_size,
            "strip_boilerplate": self.strip_boilerplate,
            "document_parser_version": DOCUMENT_PARSER_VERSION,
        }

    def rebuild_index(self) -> None:
        """Load the PDF, create chunks, embed them, and save the FAISS index."""
        pages = load_pdf(self.pdf_path, strip_boilerplate=self.strip_boilerplate)
        chunks = split_documents(
            pages,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            chunking_strategy=self.chunking_strategy,
            min_chunk_size=self.min_chunk_size,
        )
        embeddings = self.embedder.embed_documents(chunks)
        self.store.build(embeddings, chunks)
        self.store.save()

    def warm_up(self) -> None:
        """Load the local LLM before the first user question."""
        self.llm.load()

    def answer(self, question: str) -> RagResponse:
        """Answer a question using retrieved chunks from the saved FAISS index.

        The LLM is called only when at least one retrieved chunk passes the
        similarity threshold.
        """
        if self.store.index is None:
            self.store.load()

        relevant_results = self.retrieve(question)

        if not relevant_results:
            return {"answer": NO_CONTEXT_ANSWER, "sources": []}

        context = self._format_context(relevant_results)
        prompt = build_answer_prompt(context, question)
        answer = self.llm.generate(prompt, system_prompt=SYSTEM_PROMPT)

        return {
            "answer": answer,
            "sources": self._build_sources(relevant_results),
        }

    def retrieve(self, question: str) -> list[SearchResult]:
        """Return up to ``top_k`` chunks that pass the similarity threshold.

        In hybrid mode the dense candidates are re-ranked with Reciprocal Rank
        Fusion of their dense rank and their BM25 keyword rank. When a reranker
        model is set, a cross-encoder then gives the final order. The threshold
        still applies to the dense score, so off-topic questions keep getting
        no context.
        """
        if self.store.index is None:
            self.store.load()

        reranks = self.retrieval_mode == "hybrid" or self.reranker is not None
        candidate_count = self.hybrid_candidates if reranks else self.top_k
        query_embedding = self.embedder.embed_query(question)
        results = self._filter_low_similarity(
            self.store.search(query_embedding, top_k=candidate_count)
        )
        if self.retrieval_mode == "hybrid" and results:
            results = self._fuse_keyword_ranks(question, results)
        results = self._deduplicate_results(results)
        if self.reranker is not None and results:
            results = self._rerank(question, results)
        return results[: self.top_k]

    def _fuse_keyword_ranks(
        self, question: str, results: list[SearchResult]
    ) -> list[SearchResult]:
        keyword_ranks = self.store.keyword_ranks(question)
        fused: list[SearchResult] = []
        for dense_rank, result in enumerate(results, start=1):
            keyword_rank = int(keyword_ranks[result["position"]])
            fused.append(
                {
                    **result,
                    "keyword_rank": keyword_rank,
                    "hybrid_score": 1 / (self.hybrid_rrf_k + dense_rank)
                    + 1 / (self.hybrid_rrf_k + keyword_rank),
                }
            )
        return sorted(fused, key=lambda result: result["hybrid_score"], reverse=True)

    def _rerank(self, question: str, results: list[SearchResult]) -> list[SearchResult]:
        # Score the text the LLM will read: the parent text when there is one.
        texts = [str(result.get("parent_text") or result["text"]) for result in results]
        scores = self.reranker.scores(question, texts)
        reranked = [
            {**result, "rerank_score": float(score)} for result, score in zip(results, scores)
        ]
        # Stable sort: equal scores keep the dense / hybrid order.
        return sorted(reranked, key=lambda result: result["rerank_score"], reverse=True)

    def _filter_low_similarity(self, results: list[SearchResult]) -> list[SearchResult]:
        return [
            result
            for result in results
            if result["similarity_score"] >= self.similarity_threshold
        ]

    def _deduplicate_results(self, results: list[SearchResult]) -> list[SearchResult]:
        deduplicated: list[SearchResult] = []
        seen_chunks: set[str] = set()
        seen_texts: set[str] = set()

        for result in results:
            chunk_id = result.get("chunk_id", "")
            text_key = self._normalize_context_text(
                result.get("parent_text") or result.get("text", "")
            )
            if chunk_id and chunk_id in seen_chunks:
                continue
            if text_key and text_key in seen_texts:
                continue

            if chunk_id:
                seen_chunks.add(chunk_id)
            if text_key:
                seen_texts.add(text_key)
            deduplicated.append(result)

        return deduplicated

    def _format_context(self, results: list[SearchResult]) -> str:
        blocks: list[str] = []
        used_chars = 0

        for result in results:
            header = f"[{result['source']}, trang {result['page']}]"
            body = str(result.get("parent_text") or result["text"]).strip()
            block = f"{header}\n{body}"
            remaining = self.max_context_chars - used_chars
            if remaining <= 0:
                break
            if len(block) > remaining:
                block = self._trim_context_block(block, remaining)
            if block:
                blocks.append(block)
                used_chars += len(block) + 2

        return "\n\n".join(blocks)

    @staticmethod
    def _normalize_context_text(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip().lower()

    @staticmethod
    def _trim_context_block(text: str, max_chars: int) -> str:
        if max_chars <= 0:
            return ""
        if len(text) <= max_chars:
            return text
        trimmed = text[:max_chars].rsplit(" ", 1)[0].strip()
        return trimmed if trimmed else text[:max_chars].strip()

    def _build_sources(self, results: list[SearchResult]) -> list[Source]:
        sources: list[Source] = []
        seen: set[tuple[str, int, int]] = set()

        for result in results:
            key = (result["source"], result["page"], result["chunk_index"])
            if key in seen:
                continue

            seen.add(key)
            sources.append(
                {
                    "source": result["source"],
                    "page": result["page"],
                    "page_chunk_index": result.get(
                        "page_chunk_index", result["chunk_index"]
                    ),
                    "document_chunk_index": result.get(
                        "document_chunk_index", result["chunk_index"]
                    ),
                    "chunk_index": result["chunk_index"],
                    "similarity_score": result["similarity_score"],
                    "document_id": result.get("document_id", ""),
                    "chunk_id": result.get("chunk_id", ""),
                    "word_count": result.get("word_count", 0),
                    "token_count": result.get("token_count", 0),
                    "section_title": result.get("section_title", ""),
                    "chunking_strategy": result.get("chunking_strategy", ""),
                    "parent_id": result.get("parent_id", ""),
                    "image_refs": result.get("image_refs", []),
                }
            )

        return sources
