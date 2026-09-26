import math
from typing import Any

from ..retrieval import SearchResult


def _expected_pages(question: dict[str, Any]) -> set[int]:
    return {int(page) for page in question.get("expected_pages", [])}


def _expected_keywords(question: dict[str, Any]) -> list[str]:
    return [
        str(keyword).lower()
        for keyword in question.get("expected_keywords", [])
        if str(keyword).strip()
    ]


def is_relevant(result: SearchResult, question: dict[str, Any]) -> bool:
    """Decide whether one retrieved chunk is relevant to a labeled question.

    - Pages and keywords labeled: the chunk must be on an expected page AND
      contain at least one expected keyword. A chunk from the right page that
      talks about something else is not counted.
    - Only pages labeled: the chunk must be on an expected page.
    - Only keywords labeled: the chunk must contain an expected keyword.
    """
    expected_pages = _expected_pages(question)
    expected_keywords = _expected_keywords(question)

    page_match = int(result["page"]) in expected_pages
    text = f"{result.get('text', '')} {result.get('parent_text', '')}".lower()
    keyword_match = any(keyword in text for keyword in expected_keywords)

    if expected_pages and expected_keywords:
        return page_match and keyword_match
    if expected_pages:
        return page_match
    if expected_keywords:
        return keyword_match
    return False


def evaluate_results(
    results: list[SearchResult], question: dict[str, Any], top_k: int
) -> dict[str, float]:
    """Compute hit, precision, recall, MRR and nDCG for one question.

    Recall is measured over the labeled pages: the share of expected pages
    covered by at least one relevant chunk in the top-k. Questions without page
    labels have no recall denominator, so recall falls back to the hit value.

    nDCG uses an ideal ranking with as many relevant items as the question has
    labeled pages (capped at top_k), instead of the number that happened to be
    retrieved. Missing a labeled page therefore lowers nDCG.
    """
    top_results = results[:top_k]
    relevance = [1 if is_relevant(result, question) else 0 for result in top_results]
    expected_pages = _expected_pages(question)

    if expected_pages:
        matched_pages = {
            int(result["page"])
            for result, rel in zip(top_results, relevance)
            if rel
        }
        recall = len(matched_pages & expected_pages) / len(expected_pages)
    else:
        recall = 1.0 if any(relevance) else 0.0

    reciprocal_rank = 0.0
    for index, rel in enumerate(relevance, start=1):
        if rel:
            reciprocal_rank = 1.0 / index
            break

    labeled_relevant = len(expected_pages) if expected_pages else 1
    ideal_count = min(max(labeled_relevant, sum(relevance)), top_k)
    ideal_relevance = [1] * ideal_count + [0] * (top_k - ideal_count)
    ideal_dcg = _dcg(ideal_relevance)
    ndcg = _dcg(relevance) / ideal_dcg if ideal_dcg else 0.0

    return {
        "hit": 1.0 if any(relevance) else 0.0,
        "precision": sum(relevance) / top_k if top_k else 0.0,
        "recall": recall,
        "mrr": reciprocal_rank,
        "ndcg": ndcg,
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _dcg(relevance: list[int]) -> float:
    return sum(
        rel / math.log2(rank + 2)
        for rank, rel in enumerate(relevance)
    )
