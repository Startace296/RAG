"""Compare document models on the same questions, without calling the LLM.

Four configurations add one design decision at a time:

- A_page:        one text blob per page (the original model)
- B_structured:  pages split into body / figure / table segments with metadata
- C_header:      B + heading path / box / figure line embedded and indexed with the chunk
- D_filters:     C + the question's box / figure / table, time, value type and
                 source request used at retrieval time

Each configuration gets its own index under storage/eval/<name>. Retrieval runs
in dense and hybrid mode without the cross-encoder, so the numbers isolate the
document model.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.app import RagService  # noqa: E402
from src.core import DEFAULT_STORAGE_DIR, load_env, resolve_path  # noqa: E402

from run_kinhtevn_rag_eval import load_jsonl  # noqa: E402
from run_retrieval_eval import score_question  # noqa: E402


# name -> (DOCUMENT_MODEL, CONTEXT_HEADER, METADATA_FILTERS)
CONFIGS = {
    "A_page": ("page", False, False),
    "B_structured": ("structured", False, False),
    "C_header": ("structured", True, False),
    "D_filters": ("structured", True, True),
}
QUESTION_SETS = {
    "original": PROJECT_ROOT / "evaluation" / "kinhtevn_questions.jsonl",
    "metadata": PROJECT_ROOT / "evaluation" / "kinhtevn_metadata_questions.jsonl",
}


def precision(results: list[dict[str, Any]], question: dict[str, Any]) -> float:
    """Share of the context chunks that come from an answer page (less noise for the LLM)."""
    expected = {int(page) for page in question.get("expected_pages", [])}
    return sum(int(result["page"]) in expected for result in results) / len(results) if results else 0.0


def summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    keyword_rows = [row["keyword_ratio"] for row in rows if row["keyword_ratio"] is not None]
    count = len(rows)
    return {
        "page_hit": sum(row["page_hit"] for row in rows) / count,
        "mrr": sum(row["reciprocal_rank"] for row in rows) / count,
        "precision": sum(row["precision"] for row in rows) / count,
        "keyword": sum(keyword_rows) / len(keyword_rows) if keyword_rows else 0.0,
        "context_chars": sum(row["context_chars"] for row in rows) / count,
    }


def build_service(name: str, pdf: Path, storage_root: Path, rebuild: bool = False) -> RagService:
    document_model, context_header, metadata_filters = CONFIGS[name]
    # Environment variables set here win over .env (load_dotenv does not override).
    os.environ["DOCUMENT_MODEL"] = document_model
    os.environ["CONTEXT_HEADER"] = str(context_header).lower()
    os.environ["METADATA_FILTERS"] = str(metadata_filters).lower()
    os.environ["RERANKER_MODEL"] = ""
    service = RagService.from_env(pdf_path=pdf, storage_dir=storage_root / name)
    if rebuild or not service.store.exists() or service.index_mismatches():
        print(f"[{name}] building index ...", flush=True)
        service.rebuild_index()
    service.store.load()
    return service


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    load_env()

    parser = argparse.ArgumentParser(description="Compare the page and structured document models")
    parser.add_argument("--pdf", type=Path, default=PROJECT_ROOT / "data" / "KinhteVN.pdf")
    parser.add_argument("--storage-root", type=Path, default=DEFAULT_STORAGE_DIR / "eval")
    parser.add_argument("--configs", default=",".join(CONFIGS))
    parser.add_argument("--modes", default="dense,hybrid")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild every index")
    parser.add_argument(
        "--report", type=Path, default=PROJECT_ROOT / "evaluation" / "results" / "document_model_eval.json"
    )
    args = parser.parse_args()

    names = [name.strip() for name in args.configs.split(",") if name.strip()]
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    question_sets = {name: load_jsonl(path) for name, path in QUESTION_SETS.items()}

    report: dict[str, Any] = {}
    for name in names:
        service = build_service(name, resolve_path(args.pdf), resolve_path(args.storage_root), args.rebuild)
        report[name] = {"chunks": len(service.store.metadata)}
        for mode in modes:
            service.retrieval_mode = mode
            for set_name, questions in question_sets.items():
                rows = []
                for question in questions:
                    results = service.retrieve(str(question["question"]))
                    row = score_question(results, question)
                    # Score keywords on the context the LLM reads, labels included: a box
                    # title or a figure source in the label is information the model gets.
                    context = service._format_context(results).lower()
                    keywords = [str(keyword).lower() for keyword in question.get("expected_keywords", [])]
                    if keywords:
                        row["keyword_ratio"] = sum(keyword in context for keyword in keywords) / len(keywords)
                    row["context_chars"] = len(context)
                    row["precision"] = precision(results, question)
                    row["need"] = question.get("need", question.get("type", ""))
                    row["top_chunks"] = [
                        {
                            "page": result["page"],
                            "element_type": result.get("element_type", ""),
                            "label": result.get("context_header", "") or result.get("section_title", ""),
                        }
                        for result in results
                    ]
                    rows.append(row)
                needs = sorted({row["need"] for row in rows})
                report[name][f"{mode}/{set_name}"] = {
                    "summary": summarize(rows),
                    "by_need": {need: summarize([row for row in rows if row["need"] == need]) for need in needs},
                    "questions": rows,
                }
                summary = summarize(rows)
                print(
                    f"{name:<13} {mode:<6} {set_name:<8} "
                    + " ".join(f"{key}={value:.3f}" for key, value in summary.items()),
                    flush=True,
                )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
