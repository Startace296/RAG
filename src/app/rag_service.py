import os
import re
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import TypedDict

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as transformers_logging

from ..chunking import split_documents
from ..documents import load_pdf
from ..embeddings import DEFAULT_EMBEDDING_MODEL, EmbeddingService
from ..retrieval import SearchResult, VectorStore


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NO_CONTEXT_ANSWER = "Tôi không tìm thấy đủ thông tin trong tài liệu để trả lời câu hỏi này."
DEFAULT_LOCAL_LLM_MODEL = "Qwen/Qwen3-0.6B"
transformers_logging.set_verbosity_error()


@contextmanager
def _quiet_model_load():
    if os.getenv("RAG_VERBOSE", "false").lower() == "true":
        yield
        return

    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            yield


class Source(TypedDict):
    """Source metadata returned with the final answer."""

    chunk_id: str
    source: str
    page: int
    chunk_index: int
    similarity_score: float
    document_id: str
    token_count: int
    section_title: str
    chunking_strategy: str
    parent_id: str


class RagResponse(TypedDict):
    """Final RAG response."""

    answer: str
    sources: list[Source]


def build_answer_prompt(context: str, question: str) -> str:
    """Build the final Vietnamese prompt for answer generation."""
    return f"""Bạn là trợ lý học tập. Hãy trả lời câu hỏi chỉ dựa trên phần CONTEXT được cung cấp.

Quy tắc:

1. Không sử dụng kiến thức bên ngoài CONTEXT.
2. Không tự suy đoán hoặc tạo ra thông tin không xuất hiện trong CONTEXT.
3. Nếu CONTEXT không đủ để trả lời, hãy nói:
   "{NO_CONTEXT_ANSWER}"
4. Trả lời bằng tiếng Việt, rõ ràng và dễ hiểu.
5. Mỗi ý quan trọng phải kèm nguồn theo định dạng [Tên tài liệu, trang X].
6. Không được tạo nguồn hoặc số trang không có trong CONTEXT.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""


class RagService:
    """Connect PDF loading, chunking, embeddings, FAISS retrieval, and LLM answering."""

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
    ):
        load_dotenv(PROJECT_ROOT / ".env")

        self.pdf_path = Path(pdf_path)
        self.top_k = top_k
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunking_strategy = chunking_strategy
        self.min_chunk_size = min_chunk_size
        self.similarity_threshold = similarity_threshold
        self.max_context_chars = max_context_chars
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
        )
        self.llm_model = llm_model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.llm = None
        self.enable_thinking = os.getenv("QWEN_ENABLE_THINKING", "false").lower() == "true"

    def rebuild_index(self) -> None:
        """Load the PDF, create chunks, embed them, and save the FAISS index."""
        pages = load_pdf(self.pdf_path)
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
        self._load_llm()

    def answer(self, question: str) -> RagResponse:
        """Answer a question using retrieved chunks from the saved FAISS index.

        The LLM is called only when at least one retrieved chunk passes the
        similarity threshold.
        """
        self.store.load()

        query_embedding = self.embedder.embed_query(question)
        search_results = self.store.search(query_embedding, top_k=self.top_k)
        relevant_results = self._deduplicate_results(
            self._filter_low_similarity(search_results)
        )

        if not relevant_results:
            return {"answer": NO_CONTEXT_ANSWER, "sources": []}

        context = self._format_context(relevant_results)
        prompt = build_answer_prompt(context, question)
        answer = self._call_llm(prompt)

        return {
            "answer": answer,
            "sources": self._build_sources(relevant_results),
        }

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
                    "chunk_index": result["chunk_index"],
                    "similarity_score": result["similarity_score"],
                    "document_id": result.get("document_id", ""),
                    "chunk_id": result.get("chunk_id", ""),
                    "token_count": result.get("token_count", 0),
                    "section_title": result.get("section_title", ""),
                    "chunking_strategy": result.get("chunking_strategy", ""),
                    "parent_id": result.get("parent_id", ""),
                }
            )

        return sources

    def _load_llm(self) -> None:
        if self.tokenizer is not None and self.llm is not None:
            return

        with _quiet_model_load():
            self.tokenizer = AutoTokenizer.from_pretrained(self.llm_model)
            self.llm = AutoModelForCausalLM.from_pretrained(
                self.llm_model,
                dtype=torch.float16 if self.device == "cuda" else torch.float32,
            ).to(self.device)
        self.llm.eval()

    def _call_llm(self, prompt: str) -> str:
        self._load_llm()
        if self.tokenizer is None or self.llm is None:
            raise RuntimeError("Local LLM was not loaded.")

        messages = [{"role": "user", "content": prompt}]
        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        except TypeError:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        do_sample = os.getenv("QWEN_DO_SAMPLE", "false").lower() == "true"
        generation_kwargs = {
            "max_new_tokens": int(os.getenv("MAX_NEW_TOKENS", "512")),
            "do_sample": do_sample,
            "repetition_penalty": float(os.getenv("QWEN_REPETITION_PENALTY", "1.1")),
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs.update(
                {
                    "temperature": float(os.getenv("QWEN_TEMPERATURE", "0.3")),
                    "top_p": float(os.getenv("QWEN_TOP_P", "0.8")),
                    "top_k": int(os.getenv("QWEN_TOP_K", "20")),
                }
            )

        generated_ids = self.llm.generate(**model_inputs, **generation_kwargs)
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
