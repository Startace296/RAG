import json
import warnings
from pathlib import Path
from typing import Any, TypedDict

import faiss
import numpy as np


class SearchResult(TypedDict, total=False):
    """A chunk returned from vector search with its similarity score."""

    chunk_id: str
    text: str
    source: str
    page: int
    chunk_index: int
    similarity_score: float
    document_id: str
    token_count: int
    char_count: int
    start_word: int
    end_word: int
    section_title: str
    chunking_strategy: str
    parent_id: str
    parent_text: str


class VectorStore:
    """Store normalized embeddings in FAISS and chunk metadata in JSON."""

    SUPPORTED_INDEX_TYPES = {"flat", "hnsw", "ivf"}

    def __init__(
        self,
        storage_dir: str | Path,
        index_type: str = "flat",
        hnsw_m: int = 32,
        ivf_nlist: int = 64,
        ivf_nprobe: int = 8,
    ):
        """Create a vector store rooted at a local directory."""
        self.storage_dir = Path(storage_dir)
        self.index_path = self.storage_dir / "index.faiss"
        self.metadata_path = self.storage_dir / "metadata.json"
        self.config_path = self.storage_dir / "index_config.json"
        self.index_type = self._normalize_index_type(index_type)
        self.hnsw_m = hnsw_m
        self.ivf_nlist = ivf_nlist
        self.ivf_nprobe = ivf_nprobe
        self._validate_index_args()
        self.index: faiss.Index | None = None
        self.metadata: list[dict[str, Any]] = []

    def build(self, embeddings: np.ndarray, chunks: list[dict[str, Any]]) -> None:
        """Build a FAISS index from embeddings and keep chunk metadata.

        Args:
            embeddings: Normalized vectors with shape
                ``(number_of_chunks, embedding_dimension)``.
            chunks: Chunk metadata in the same order as the embeddings.

        Raises:
            ValueError: If embeddings are empty, not 2D, or the number of
                embeddings does not match the number of chunks.
        """
        vectors = np.asarray(embeddings, dtype="float32")
        if vectors.ndim != 2:
            raise ValueError("embeddings must be a 2D numpy array")
        if vectors.shape[0] == 0:
            raise ValueError("embeddings must contain at least one vector")
        if vectors.shape[0] != len(chunks):
            raise ValueError("embeddings and chunks must have the same length")

        self.index = self._build_index(vectors)
        self.metadata = [
            {
                "chunk_id": str(chunk.get("chunk_id", "")),
                "text": str(chunk["text"]),
                "source": str(chunk["source"]),
                "page": int(chunk["page"]),
                "chunk_index": int(chunk["chunk_index"]),
                "document_id": str(chunk.get("document_id", "")),
                "token_count": int(chunk.get("token_count", 0)),
                "char_count": int(chunk.get("char_count", 0)),
                "start_word": int(chunk.get("start_word", 0)),
                "end_word": int(chunk.get("end_word", 0)),
                "section_title": str(chunk.get("section_title", "")),
                "chunking_strategy": str(chunk.get("chunking_strategy", "")),
                "parent_id": str(chunk.get("parent_id", "")),
                "parent_text": str(chunk.get("parent_text", "")),
            }
            for chunk in chunks
        ]

    def save(self) -> None:
        """Save the FAISS index and metadata JSON to disk."""
        if self.index is None:
            raise RuntimeError("Build the index before saving it")

        self.storage_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        self.metadata_path.write_text(
            json.dumps(self.metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self.config_path.write_text(
            json.dumps(self._index_config(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load(self) -> None:
        """Load the FAISS index and metadata JSON from disk."""
        if not self.index_path.exists() or not self.metadata_path.exists():
            raise FileNotFoundError("Vector index not found. Run with --rebuild first.")

        self.index = faiss.read_index(str(self.index_path))
        self._warn_on_config_mismatch()
        self._apply_search_parameters(self.index)
        self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        if self.index.ntotal != len(self.metadata):
            raise ValueError("FAISS index and metadata do not have the same length")

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[SearchResult]:
        """Search the most similar chunks for a normalized query vector.

        ``IndexFlatIP`` returns inner product scores. Because document vectors
        and query vectors are normalized by the embedding service, inner product
        is equivalent to cosine similarity.
        """
        if self.index is None:
            raise RuntimeError("Load or build the vector index first")
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        query_vector = np.asarray(query_embedding, dtype="float32")
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        if query_vector.ndim != 2 or query_vector.shape[0] != 1:
            raise ValueError("query_embedding must have shape (dimension,) or (1, dimension)")
        if query_vector.shape[1] != self.index.d:
            raise ValueError(
                f"query_embedding dimension {query_vector.shape[1]} does not match "
                f"FAISS index dimension {self.index.d}"
            )

        scores, positions = self.index.search(query_vector, top_k)
        results: list[SearchResult] = []
        for score, position in zip(scores[0], positions[0]):
            if position < 0:
                continue

            metadata = self.metadata[int(position)]
            results.append(
                {
                    "chunk_id": str(metadata.get("chunk_id", "")),
                    "text": str(metadata["text"]),
                    "source": str(metadata["source"]),
                    "page": int(metadata["page"]),
                    "chunk_index": int(metadata["chunk_index"]),
                    "similarity_score": float(score),
                    "document_id": str(metadata.get("document_id", "")),
                    "token_count": int(metadata.get("token_count", 0)),
                    "char_count": int(metadata.get("char_count", 0)),
                    "start_word": int(metadata.get("start_word", 0)),
                    "end_word": int(metadata.get("end_word", 0)),
                    "section_title": str(metadata.get("section_title", "")),
                    "chunking_strategy": str(metadata.get("chunking_strategy", "")),
                    "parent_id": str(metadata.get("parent_id", "")),
                    "parent_text": str(metadata.get("parent_text", "")),
                }
            )

        return results

    def _normalize_index_type(self, index_type: str) -> str:
        normalized = index_type.strip().lower()
        if normalized not in self.SUPPORTED_INDEX_TYPES:
            supported = ", ".join(sorted(self.SUPPORTED_INDEX_TYPES))
            raise ValueError(f"Unsupported vector index type: {index_type}. Use one of: {supported}")
        return normalized

    def _validate_index_args(self) -> None:
        if self.hnsw_m <= 0:
            raise ValueError("hnsw_m must be greater than 0")
        if self.ivf_nlist <= 0:
            raise ValueError("ivf_nlist must be greater than 0")
        if self.ivf_nprobe <= 0:
            raise ValueError("ivf_nprobe must be greater than 0")

    def _build_index(self, vectors: np.ndarray) -> faiss.Index:
        dimension = vectors.shape[1]

        if self.index_type == "flat":
            index = faiss.IndexFlatIP(dimension)
            index.add(vectors)
            return index

        if self.index_type == "hnsw":
            try:
                index = faiss.IndexHNSWFlat(dimension, self.hnsw_m, faiss.METRIC_INNER_PRODUCT)
            except TypeError:
                index = faiss.IndexHNSWFlat(dimension, self.hnsw_m)
                index.metric_type = faiss.METRIC_INNER_PRODUCT
            index.add(vectors)
            return index

        nlist = max(1, min(self.ivf_nlist, vectors.shape[0]))
        quantizer = faiss.IndexFlatIP(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(vectors)
        index.nprobe = max(1, min(self.ivf_nprobe, nlist))
        index.add(vectors)
        return index

    def _apply_search_parameters(self, index: faiss.Index) -> None:
        if hasattr(index, "nprobe"):
            index.nprobe = max(1, min(self.ivf_nprobe, getattr(index, "nlist", self.ivf_nprobe)))

    def _warn_on_config_mismatch(self) -> None:
        if not self.config_path.exists():
            return

        saved_config = json.loads(self.config_path.read_text(encoding="utf-8"))
        current_config = self._index_config()
        mismatches = {
            key: (saved_config.get(key), current_config[key])
            for key in current_config
            if saved_config.get(key) != current_config[key]
        }
        if mismatches:
            details = ", ".join(
                f"{key}: saved={saved!r}, current={current!r}"
                for key, (saved, current) in mismatches.items()
            )
            warnings.warn(
                f"Loaded FAISS index was built with different vector config ({details}). "
                "Run with --rebuild if this was not intentional.",
                RuntimeWarning,
                stacklevel=2,
            )

    def _index_config(self) -> dict[str, Any]:
        return {
            "index_type": self.index_type,
            "hnsw_m": self.hnsw_m,
            "ivf_nlist": self.ivf_nlist,
            "ivf_nprobe": self.ivf_nprobe,
        }
