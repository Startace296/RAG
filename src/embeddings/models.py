from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingModelSpec:
    alias: str
    model_name: str
    dimension: int | None
    document_prefix: str = ""
    query_prefix: str = ""
    language_note: str = ""
    size_note: str = ""


EMBEDDING_MODEL_SPECS: dict[str, EmbeddingModelSpec] = {
    "miniLM-multilingual": EmbeddingModelSpec(
        alias="miniLM-multilingual",
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        dimension=384,
        language_note="Multilingual baseline, works reasonably for Vietnamese.",
        size_note="Lightweight and fast.",
    ),
    "mpnet-multilingual": EmbeddingModelSpec(
        alias="mpnet-multilingual",
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        dimension=768,
        language_note="Stronger multilingual sentence-transformers baseline.",
        size_note="Larger and slower than MiniLM.",
    ),
    "e5-small": EmbeddingModelSpec(
        alias="e5-small",
        model_name="intfloat/multilingual-e5-small",
        dimension=384,
        document_prefix="passage: ",
        query_prefix="query: ",
        language_note="Multilingual retrieval model; often strong for Vietnamese retrieval.",
        size_note="Small E5 model, good speed/quality tradeoff.",
    ),
    "e5-base": EmbeddingModelSpec(
        alias="e5-base",
        model_name="intfloat/multilingual-e5-base",
        dimension=768,
        document_prefix="passage: ",
        query_prefix="query: ",
        language_note="Larger multilingual retrieval model; good candidate for Vietnamese.",
        size_note="Higher quality potential, slower and heavier.",
    ),
}


DEFAULT_EMBEDDING_BENCHMARK_ALIASES = [
    "miniLM-multilingual",
    "mpnet-multilingual",
    "e5-small",
    "e5-base",
]


def resolve_embedding_model(value: str) -> EmbeddingModelSpec:
    """Resolve a benchmark alias or raw Hugging Face model name."""
    key = value.strip()
    if key in EMBEDDING_MODEL_SPECS:
        return EMBEDDING_MODEL_SPECS[key]

    return EmbeddingModelSpec(
        alias=key,
        model_name=key,
        dimension=None,
    )
