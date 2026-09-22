import math
from typing import Any

from ..retrieval import SearchResult


def is_relevant(result: SearchResult, question: dict[str, Any]) -> bool:
    expected_pages = {int(page) for page in question.get("expected_pages", [])}
    if expected_pages and int(result["page"]) in expected_pages:
        return True

    expected_keywords = [
        str(keyword).lower()
        for keyword in question.get("expected_keywords", [])
    ]
    text = f"{result.get('text', '')} {result.get('parent_text', '')}".lower()
    return any(keyword in text for keyword in expected_keywords)


def evaluate_results(
    results: list[SearchResult], question: dict[str, Any], top_k: int
) -> dict[str, float]:
    top_results = results[:top_k]
    relevance = [1 if is_relevant(result, question) else 0 for result in top_results]
    expected_pages = {int(page) for page in question.get("expected_pages", [])}

    if expected_pages:
        matched_pages = {
            int(result["page"])
            for result in top_results
            if int(result["page"]) in expected_pages
        }
        recall = len(matched_pages) / len(expected_pages)
    else:
        recall = 1.0 if any(relevance) else 0.0

    reciprocal_rank = 0.0
    for index, rel in enumerate(relevance, start=1):
        if rel:
            reciprocal_rank = 1.0 / index
            break

    ideal_count = min(sum(relevance), top_k)
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
