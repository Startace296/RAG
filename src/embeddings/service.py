import os
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from collections.abc import Mapping, Sequence
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@contextmanager
def _quiet_model_load():
    if os.getenv("RAG_VERBOSE", "false").lower() == "true":
        yield
        return

    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            yield


class EmbeddingService:
    """Create normalized embeddings for document chunks and user queries."""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        document_prefix: str = "",
        query_prefix: str = "",
    ):
        """Load a local sentence-transformers embedding model.

        Args:
            model_name: Hugging Face model name. The default model is a light
                multilingual MiniLM model that works reasonably well for
                Vietnamese text and local semantic search.
            document_prefix: Optional prefix added to document texts before
                encoding. E5-style models commonly use ``passage: ``.
            query_prefix: Optional prefix added to user questions before
                encoding. E5-style models commonly use ``query: ``.
        """
        self.model_name = model_name
        self.document_prefix = document_prefix
        self.query_prefix = query_prefix
        with _quiet_model_load():
            self.model = SentenceTransformer(model_name)

    def embed_documents(self, chunks: Sequence[Mapping[str, Any]]) -> np.ndarray:
        """Embed many text chunks.

        Args:
            chunks: A sequence of chunk objects. Each chunk must have a ``text``
                field, for example: ``{"text": "...", "page": 1}``.

        Returns:
            A NumPy array with shape ``(number_of_chunks, embedding_dimension)``.
            With the default model, ``embedding_dimension`` is 384.
        """
        texts = [self._with_prefix(str(chunk["text"]), self.document_prefix) for chunk in chunks]
        return self._encode(texts)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed one user question.

        Args:
            query: The question or search query.

        Returns:
            A NumPy array with shape ``(1, embedding_dimension)``. With the
            default model, ``embedding_dimension`` is 384.
        """
        return self._encode([self._with_prefix(query, self.query_prefix)])

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Embed raw texts.

        This method keeps backward compatibility with older code. Prefer
        ``embed_documents`` for chunks and ``embed_query`` for questions.
        """
        return self._encode(texts)

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            embedding_dimension = self.model.get_sentence_embedding_dimension() or 0
            return np.empty((0, embedding_dimension), dtype=np.float32)

        embeddings = self.model.encode(
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.asarray(embeddings, dtype=np.float32)

    @staticmethod
    def _with_prefix(text: str, prefix: str) -> str:
        if not prefix:
            return text
        separator = "" if prefix[-1].isspace() else " "
        return f"{prefix}{separator}{text}"
