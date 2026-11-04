"""Compare retrieval configurations without calling the LLM.

Each configuration runs the questions through ``RagService.retrieve`` and is
scored on the top-k chunks that would become the LLM context.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.app import RagService  # noqa: E402
from src.core import DEFAULT_STORAGE_DIR, default_questions_path, load_env, resolve_path  # noqa: E402
from src.retrieval import CrossEncoderReranker  # noqa: E402

from run_kinhtevn_rag_eval import load_jsonl  # noqa: E402


# name -> (retrieval mode, use the cross-encoder)
CONFIGS = {
    "dense": ("dense", False),
    "hybrid": ("hybrid", False),
    "dense+rerank": ("dense", True),
    "hybrid+rerank": ("hybrid", True),
}


def score_question(results: list[dict[str, Any]], question: dict[str, Any]) -> dict[str, Any]:
    expected_pages = {int(page) for page in question.get("expected_pages", [])}
    pages = [int(result["page"]) for result in results]
    first_hit = next(
        (rank for rank, page in enumerate(pages, start=1) if page in expected_pages), None
    )
    context = " ".join(str(result.get("parent_text") or result["text"]) for result in results)
    context = context.lower()
    keywords = [str(keyword) for keyword in question.get("expected_keywords", [])]
    matched = [keyword for keyword in keywords if keyword.lower() in context]
    return {
        "id": question.get("id", ""),
        "pages": pages,
        "page_hit": first_hit is not None,
        "reciprocal_rank": 1 / first_hit if first_hit else 0.0,
        "keyword_ratio": len(matched) / len(keywords) if keywords else None,
        "context_chars": len(context),
    }


def summarize(rows: list[dict[str, Any]], seconds: list[float]) -> dict[str, Any]:
    keyword_rows = [row["keyword_ratio"] for row in rows if row["keyword_ratio"] is not None]
    return {
        "questions": len(rows),
        "page_hit_rate": sum(row["page_hit"] for row in rows) / len(rows),
        "mrr": sum(row["reciprocal_rank"] for row in rows) / len(rows),
        "context_keyword_ratio": sum(keyword_rows) / len(keyword_rows) if keyword_rows else None,
        "avg_context_chars": sum(row["context_chars"] for row in rows) / len(rows),
        "avg_retrieve_ms": 1000 * sum(seconds) / len(seconds),
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    load_env()

    parser = argparse.ArgumentParser(description="Compare dense, hybrid and cross-encoder retrieval")
    parser.add_argument("--pdf", type=Path, default=PROJECT_ROOT / "data" / "KinhteVN.pdf")
    parser.add_argument("--storage-dir", type=Path, default=DEFAULT_STORAGE_DIR)
    parser.add_argument("--questions", type=Path, default=default_questions_path())
    parser.add_argument("--configs", default=",".join(CONFIGS), help="Comma-separated: " + ", ".join(CONFIGS))
    parser.add_argument(
        "--top-k",
        default="",
        help="Comma-separated context sizes to score, e.g. 1,3,5,8,10 (default: TOP_K from .env)",
    )
    parser.add_argument("--report", type=Path, default=PROJECT_ROOT / "evaluation" / "results" / "reranker_eval.json")
    args = parser.parse_args()

    names = [name.strip() for name in args.configs.split(",") if name.strip()]
    unknown = [name for name in names if name not in CONFIGS]
    if unknown:
        raise SystemExit(f"Unknown configs: {unknown}. Use: {', '.join(CONFIGS)}")

    questions = load_jsonl(resolve_path(args.questions))
    service = RagService.from_env(pdf_path=resolve_path(args.pdf), storage_dir=resolve_path(args.storage_dir))
    if not service.store.exists() or service.index_mismatches():
        service.rebuild_index()
    service.store.load()
    # Retrieve once with the largest k; a smaller k is a prefix of the same ranking.
    top_ks = sorted({int(k) for k in args.top_k.split(",") if k.strip()}) or [service.top_k]
    service.top_k = max(top_ks)
    service.hybrid_candidates = max(service.hybrid_candidates, service.top_k)
    # RERANKER_MODEL from .env; the service keeps it only to reuse it per config.
    reranker = service.reranker
    if reranker is None and any(CONFIGS[name][1] for name in names):
        raise SystemExit("Set RERANKER_MODEL in .env to evaluate the +rerank configs.")

    report: dict[str, Any] = {
        "reranker_model": reranker.model_name if reranker else "",
        "configs": {},
    }
    for name in names:
        mode, use_reranker = CONFIGS[name]
        service.retrieval_mode = mode
        service.reranker = reranker if use_reranker else None
        if use_reranker:
            reranker.load()  # keep model loading out of the timing
        service.retrieve(questions[0]["question"])  # warm up caches (BM25 index, kernels)

        retrieved, seconds = [], []
        for question in questions:
            started = time.perf_counter()
            retrieved.append(service.retrieve(str(question["question"])))
            seconds.append(time.perf_counter() - started)

        for k in top_ks:
            label = name if len(top_ks) == 1 else f"{name}@{k}"
            rows = [score_question(results[:k], question) for results, question in zip(retrieved, questions)]
            summary = summarize(rows, seconds)
            report["configs"][label] = {"summary": summary, "questions": rows}
            print(f"{label:<17} " + " ".join(f"{key}={value:.3f}" for key, value in summary.items() if key != "questions"), flush=True)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
