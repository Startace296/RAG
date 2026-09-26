import json
from pathlib import Path
from typing import Any


def load_questions(path: Path) -> list[dict[str, Any]]:
    """Load benchmark questions from a JSONL file, skipping blanks and comments."""
    questions: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        item = json.loads(line)
        if not item.get("question"):
            raise ValueError(f"Missing question at {path}:{line_number}")
        questions.append(item)
    return questions


def require_inputs(pdf: Path, questions_path: Path) -> list[dict[str, Any]]:
    """Check benchmark inputs exist and return the loaded questions."""
    if not pdf.exists():
        raise SystemExit(f"PDF not found: {pdf}")
    if not questions_path.exists():
        raise SystemExit(f"Question file not found: {questions_path}")

    questions = load_questions(questions_path)
    if not questions:
        raise SystemExit(f"No benchmark questions found: {questions_path}")
    return questions


def print_table(rows: list[dict[str, Any]], headers: list[str]) -> None:
    """Print rows as an aligned table. The first column is left-aligned."""
    formatted_rows = [
        {
            header: f"{row[header]:.4f}" if isinstance(row[header], float) else str(row[header])
            for header in headers
        }
        for row in rows
    ]
    widths = {
        header: max([len(header)] + [len(row[header]) for row in formatted_rows])
        for header in headers
    }

    def format_row(row: dict[str, str]) -> str:
        return "  ".join(
            row[header].ljust(widths[header])
            if index == 0
            else row[header].rjust(widths[header])
            for index, header in enumerate(headers)
        )

    print(format_row({header: header for header in headers}))
    print("  ".join("-" * widths[header] for header in headers))
    for row in formatted_rows:
        print(format_row(row))


METRIC_NAMES = ["hit", "precision", "recall", "mrr", "ndcg"]


def metric_headers(top_k: int) -> list[str]:
    return [
        f"hit@{top_k}",
        f"precision@{top_k}",
        f"recall@{top_k}",
        "mrr",
        f"ndcg@{top_k}",
    ]


def mean_metrics(per_question: list[dict[str, float]], top_k: int) -> dict[str, float]:
    """Average per-question metrics into table columns such as ``hit@5``."""
    return {
        header: (
            sum(metrics[name] for metrics in per_question) / len(per_question)
            if per_question
            else 0.0
        )
        for name, header in zip(METRIC_NAMES, metric_headers(top_k))
    }
