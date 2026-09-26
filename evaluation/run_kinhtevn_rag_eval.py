import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.app.rag_service import NO_CONTEXT_ANSWER, RagService  # noqa: E402
from src.paths import DEFAULT_STORAGE_DIR, resolve_path  # noqa: E402


DEFAULT_QUESTIONS = PROJECT_ROOT / "evaluation" / "kinhtevn_questions.jsonl"
DEFAULT_ANSWER_CHECKS = PROJECT_ROOT / "evaluation" / "kinhtevn_answer_checks.jsonl"
DEFAULT_REPORT = PROJECT_ROOT / "evaluation" / "kinhtevn_rag_console_report.jsonl"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        row = json.loads(line)
        if not row.get("question"):
            raise ValueError(f"Missing question at {path}:{line_number}")
        rows.append(row)
    return rows


def build_service(pdf_path: Path, storage_dir: Path) -> RagService:
    return RagService.from_env(pdf_path=pdf_path, storage_dir=storage_dir)


def source_hits(sources: list[dict[str, Any]], question: dict[str, Any]) -> dict[str, Any]:
    expected_pages = {int(page) for page in question.get("expected_pages", [])}
    retrieved_pages = [int(source["page"]) for source in sources]
    matched_pages = sorted(set(retrieved_pages) & expected_pages)
    return {
        "retrieved_pages": retrieved_pages,
        "matched_pages": matched_pages,
        "page_hit": bool(matched_pages) if expected_pages else None,
    }


def answer_keyword_hits(answer: str, question: dict[str, Any]) -> dict[str, Any]:
    answer_lower = answer.lower()
    keywords = [str(keyword) for keyword in question.get("expected_keywords", [])]
    matched_keywords = [
        keyword
        for keyword in keywords
        if keyword.lower() in answer_lower
    ]
    return {
        "matched_keywords": matched_keywords,
        "keyword_hit_ratio": len(matched_keywords) / len(keywords) if keywords else None,
    }


def run_question(service: RagService, question: dict[str, Any]) -> dict[str, Any]:
    response = service.answer(str(question["question"]))
    sources = response["sources"]
    page_eval = source_hits(sources, question)
    keyword_eval = answer_keyword_hits(response["answer"], question)
    return {
        "id": question.get("id", ""),
        "type": question.get("type", ""),
        "question": question["question"],
        "expected_answer": question.get("expected_answer", ""),
        "answer": response["answer"],
        "sources": sources,
        **page_eval,
        **keyword_eval,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    page_rows = [row for row in rows if row["page_hit"] is not None]
    keyword_rows = [row for row in rows if row["keyword_hit_ratio"] is not None]
    no_context_rows = [
        row
        for row in rows
        if row["type"] == "no_answer"
    ]
    return {
        "questions": len(rows),
        "page_hit_rate": (
            sum(1 for row in page_rows if row["page_hit"]) / len(page_rows)
            if page_rows
            else None
        ),
        "avg_answer_keyword_hit_ratio": (
            sum(float(row["keyword_hit_ratio"]) for row in keyword_rows) / len(keyword_rows)
            if keyword_rows
            else None
        ),
        "no_context_pass_rate": (
            sum(1 for row in no_context_rows if NO_CONTEXT_ANSWER in row["answer"])
            / len(no_context_rows)
            if no_context_rows
            else None
        ),
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Run KinhteVN questions through RagService.answer")
    parser.add_argument("--pdf", type=Path, default=PROJECT_ROOT / "data" / "KinhteVN.pdf")
    parser.add_argument("--storage-dir", type=Path, default=DEFAULT_STORAGE_DIR)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--answer-checks", type=Path, default=DEFAULT_ANSWER_CHECKS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--include-answer-checks", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    pdf_path = resolve_path(args.pdf)
    storage_dir = resolve_path(args.storage_dir)
    questions = load_jsonl(args.questions)
    if args.include_answer_checks:
        questions.extend(load_jsonl(args.answer_checks))
    if args.limit > 0:
        questions = questions[: args.limit]

    service = build_service(pdf_path, storage_dir)
    mismatches = service.index_mismatches() if service.store.exists() else {}
    if args.rebuild or not service.store.exists() or mismatches:
        if mismatches:
            print(f"Rebuilding stale index: {sorted(mismatches)}", flush=True)
        service.rebuild_index()

    rows = []
    for index, question in enumerate(questions, start=1):
        print(f"[{index}/{len(questions)}] {question['question']}", flush=True)
        row = run_question(service, question)
        rows.append(row)
        print(
            f"  pages={row['retrieved_pages']} page_hit={row['page_hit']} "
            f"keyword_ratio={row['keyword_hit_ratio']}",
            flush=True,
        )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps(summarize(rows), ensure_ascii=False, indent=2))
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
