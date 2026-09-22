import re
from hashlib import sha256
from typing import Any, Literal, TypedDict


ChunkingStrategy = Literal[
    "fixed_word",
    "sentence",
    "paragraph",
    "recursive",
    "semantic",
    "parent_child",
    "sliding_window",
]
SUPPORTED_CHUNKING_STRATEGIES = {
    "fixed_word",
    "sentence",
    "paragraph",
    "recursive",
    "semantic",
    "parent_child",
    "sliding_window",
}


class TextChunk(TypedDict, total=False):
    """Text chunk with traceable source and chunking metadata."""

    chunk_id: str
    text: str
    page: int
    source: str
    chunk_index: int
    document_id: str
    start_word: int
    end_word: int
    token_count: int
    char_count: int
    section_title: str
    chunking_strategy: str
    parent_id: str
    parent_text: str


def split_documents(
    documents: list[dict[str, Any]],
    chunk_size: int = 300,
    chunk_overlap: int = 50,
    chunking_strategy: str = "recursive",
    min_chunk_size: int = 50,
) -> list[TextChunk]:
    """Split page documents into chunks using the selected strategy.

    Chunks are created inside each page only, so text from different pages is
    never mixed together.
    """
    _validate_chunking_args(chunk_size, chunk_overlap, min_chunk_size)
    strategy = _normalize_strategy(chunking_strategy)

    chunks: list[TextChunk] = []
    for document in documents:
        text = str(document["text"]).strip()
        if not text:
            continue

        if strategy in {"fixed_word", "sliding_window"}:
            text_chunks = _split_words(text.split(), chunk_size, chunk_overlap)
        elif strategy == "parent_child":
            text_chunks = _split_parent_child(
                text=text,
                child_size=chunk_size,
                child_overlap=chunk_overlap,
            )
        else:
            units = _split_text_by_strategy(text, strategy, chunk_size)
            if strategy == "semantic":
                text_chunks = _split_semantic_units(
                    units=units,
                    chunk_size=chunk_size,
                    min_chunk_size=min_chunk_size,
                )
            else:
                text_chunks = _pack_units(units, chunk_size, chunk_overlap, min_chunk_size)

        page_words = text.split()
        search_start_word = 0
        for chunk_index, chunk_text in enumerate(text_chunks):
            if strategy == "parent_child":
                parent_text, child_text = _unpack_parent_child(chunk_text)
            else:
                parent_text = ""
                child_text = chunk_text

            chunk_words = child_text.split()
            start_word = _find_word_offset(page_words, chunk_words, search_start_word)
            end_word = start_word + len(chunk_words)
            search_start_word = max(start_word + 1, end_word - chunk_overlap)
            chunks.append(
                _build_chunk(
                    document=document,
                    text=child_text,
                    chunk_index=chunk_index,
                    chunking_strategy=strategy,
                    start_word=start_word,
                    end_word=end_word,
                    parent_text=parent_text,
                )
            )

    return chunks


def _validate_chunking_args(
    chunk_size: int, chunk_overlap: int, min_chunk_size: int
) -> None:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be greater than or equal to 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    if min_chunk_size <= 0:
        raise ValueError("min_chunk_size must be greater than 0")
    if min_chunk_size > chunk_size:
        raise ValueError("min_chunk_size must be smaller than or equal to chunk_size")


def _normalize_strategy(chunking_strategy: str) -> ChunkingStrategy:
    strategy = chunking_strategy.strip().lower()
    if strategy not in SUPPORTED_CHUNKING_STRATEGIES:
        supported = ", ".join(sorted(SUPPORTED_CHUNKING_STRATEGIES))
        raise ValueError(f"Unsupported chunking strategy: {strategy}. Use one of: {supported}")
    return strategy  # type: ignore[return-value]


def _split_text_by_strategy(
    text: str, strategy: ChunkingStrategy, chunk_size: int
) -> list[str]:
    if strategy == "paragraph":
        return _split_paragraphs(text)
    if strategy == "sentence":
        return _split_sentences(text)
    if strategy == "semantic":
        return _split_sentences(text)
    if strategy == "recursive":
        return _split_recursive_units(text, chunk_size)
    return text.split()


def _split_words(
    words: list[str], chunk_size: int, chunk_overlap: int
) -> list[str]:
    chunks: list[str] = []
    step = chunk_size - chunk_overlap

    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + chunk_size]).strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(words):
            break

    return chunks


def _split_paragraphs(text: str) -> list[str]:
    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n+", text)
        if paragraph.strip()
    ]
    if len(paragraphs) > 1:
        return paragraphs
    return _split_sentences(text)


def _split_sentences(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []

    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", normalized)
        if sentence.strip()
    ]
    return sentences or [normalized]


def _split_recursive_units(text: str, chunk_size: int) -> list[str]:
    units: list[str] = []
    for paragraph in _split_paragraphs(text):
        paragraph_words = paragraph.split()
        if len(paragraph_words) <= chunk_size:
            units.append(paragraph)
            continue
        units.extend(_split_sentences(paragraph))
    return units


def _pack_units(
    units: list[str],
    chunk_size: int,
    chunk_overlap: int,
    min_chunk_size: int,
) -> list[str]:
    chunks: list[str] = []
    current_units: list[str] = []
    current_word_count = 0

    for unit in units:
        unit_words = unit.split()
        if not unit_words:
            continue

        if len(unit_words) > chunk_size:
            if current_units:
                chunks.append(" ".join(current_units).strip())
                current_units = []
                current_word_count = 0
            chunks.extend(_split_words(unit_words, chunk_size, chunk_overlap))
            continue

        would_exceed = current_word_count + len(unit_words) > chunk_size
        has_enough_context = current_word_count >= min_chunk_size
        if current_units and would_exceed and has_enough_context:
            chunks.append(" ".join(current_units).strip())
            current_units = _overlap_tail(current_units, chunk_overlap)
            current_word_count = len(" ".join(current_units).split())

        current_units.append(unit)
        current_word_count += len(unit_words)

    if current_units:
        chunks.append(" ".join(current_units).strip())

    return [chunk for chunk in chunks if chunk]


def _split_semantic_units(
    units: list[str],
    chunk_size: int,
    min_chunk_size: int,
    break_threshold: float = 0.12,
) -> list[str]:
    chunks: list[str] = []
    current_units: list[str] = []
    current_word_count = 0

    for unit in units:
        unit_words = unit.split()
        if not unit_words:
            continue

        if len(unit_words) > chunk_size:
            if current_units:
                chunks.append(" ".join(current_units).strip())
                current_units = []
                current_word_count = 0
            chunks.extend(_split_words(unit_words, chunk_size, 0))
            continue

        similarity = (
            _lexical_similarity(" ".join(current_units), unit)
            if current_units
            else 1.0
        )
        would_exceed = current_word_count + len(unit_words) > chunk_size
        semantic_break = similarity < break_threshold and current_word_count >= min_chunk_size

        if current_units and (would_exceed or semantic_break):
            chunks.append(" ".join(current_units).strip())
            current_units = []
            current_word_count = 0

        current_units.append(unit)
        current_word_count += len(unit_words)

    if current_units:
        chunks.append(" ".join(current_units).strip())

    return [chunk for chunk in chunks if chunk]


def _split_parent_child(
    text: str,
    child_size: int,
    child_overlap: int,
    parent_size_multiplier: int = 2,
) -> list[str]:
    parent_size = child_size * parent_size_multiplier
    parent_chunks = _split_words(text.split(), parent_size, child_overlap)
    packed_chunks: list[str] = []

    for parent_index, parent_text in enumerate(parent_chunks):
        child_chunks = _split_words(parent_text.split(), child_size, child_overlap)
        for child_text in child_chunks:
            packed_chunks.append(_pack_parent_child(parent_index, parent_text, child_text))

    return packed_chunks


def _pack_parent_child(parent_index: int, parent_text: str, child_text: str) -> str:
    return f"PARENT_INDEX:{parent_index}\nPARENT_TEXT:{parent_text}\nCHILD_TEXT:{child_text}"


def _unpack_parent_child(packed_text: str) -> tuple[str, str]:
    parent_marker = "\nPARENT_TEXT:"
    child_marker = "\nCHILD_TEXT:"
    if parent_marker not in packed_text or child_marker not in packed_text:
        return "", packed_text

    parent_start = packed_text.index(parent_marker) + len(parent_marker)
    child_start = packed_text.index(child_marker)
    parent_text = packed_text[parent_start:child_start].strip()
    child_text = packed_text[child_start + len(child_marker) :].strip()
    return parent_text, child_text


def _lexical_similarity(left: str, right: str) -> float:
    left_terms = _content_terms(left)
    right_terms = _content_terms(right)
    if not left_terms or not right_terms:
        return 0.0

    overlap = len(left_terms & right_terms)
    union = len(left_terms | right_terms)
    return overlap / union if union else 0.0


def _content_terms(text: str) -> set[str]:
    return {
        term
        for term in re.findall(r"\w+", text.lower())
        if len(term) > 2
    }


def _overlap_tail(units: list[str], chunk_overlap: int) -> list[str]:
    if chunk_overlap <= 0:
        return []

    tail: list[str] = []
    word_count = 0
    for unit in reversed(units):
        unit_word_count = len(unit.split())
        if word_count + unit_word_count > chunk_overlap and tail:
            break
        tail.insert(0, unit)
        word_count += unit_word_count
        if word_count >= chunk_overlap:
            break
    return tail


def _build_chunk(
    document: dict[str, Any],
    text: str,
    chunk_index: int,
    chunking_strategy: ChunkingStrategy,
    start_word: int,
    end_word: int,
    parent_text: str = "",
) -> TextChunk:
    words = text.split()
    document_id = str(document.get("document_id", ""))
    source = str(document["source"])
    page = int(document["page"])
    chunk_id = _build_chunk_id(document_id, source, page, chunk_index, text)
    parent_id = _build_parent_id(document_id, source, page, chunk_index, parent_text)

    return {
        "chunk_id": chunk_id,
        "text": text,
        "page": page,
        "source": source,
        "chunk_index": chunk_index,
        "document_id": document_id,
        "start_word": start_word,
        "end_word": end_word,
        "token_count": len(words),
        "char_count": len(text),
        "section_title": _guess_section_title(text),
        "chunking_strategy": chunking_strategy,
        "parent_id": parent_id,
        "parent_text": parent_text,
    }


def _find_word_offset(
    page_words: list[str], chunk_words: list[str], start_at: int
) -> int:
    if not chunk_words:
        return start_at

    max_start = len(page_words) - len(chunk_words)
    for index in range(max(start_at, 0), max_start + 1):
        if page_words[index : index + len(chunk_words)] == chunk_words:
            return index

    for index in range(0, max_start + 1):
        if page_words[index : index + len(chunk_words)] == chunk_words:
            return index

    return max(start_at, 0)


def _build_chunk_id(
    document_id: str, source: str, page: int, chunk_index: int, text: str
) -> str:
    raw_id = f"{document_id}|{source}|{page}|{chunk_index}|{text[:120]}"
    return sha256(raw_id.encode("utf-8")).hexdigest()[:16]


def _build_parent_id(
    document_id: str, source: str, page: int, chunk_index: int, parent_text: str
) -> str:
    if not parent_text:
        return ""
    raw_id = f"{document_id}|{source}|{page}|parent|{chunk_index}|{parent_text[:120]}"
    return sha256(raw_id.encode("utf-8")).hexdigest()[:16]


def _guess_section_title(text: str) -> str:
    for line in text.splitlines():
        candidate = line.strip()
        if not candidate:
            continue
        if len(candidate.split()) <= 12 and len(candidate) <= 120:
            return candidate
        break
    return ""
