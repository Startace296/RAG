"""Query-facing metadata of a chunk in the structured document model.

Each field answers a kind of question asked about the report:

- ``heading_path``: "phần tài chính công nói gì?" (topic / section)
- ``box_id``, ``figure_no``, ``table_no``: "Hộp 2", "Hình 16", "Bảng 3"
- ``data_source``: "số liệu lấy từ nguồn nào?"
- ``time_periods``: "tháng 4/2026", "quý I/2026", "năm 2025"
- ``value_types``: "dự báo" vs "ước tính" vs reported figures
- ``figure_refs``: body text citing a figure, so text and chart can be joined

The same parsers read the question at query time, so a chunk and a question
are compared on the same normalized values.
"""

import re
from typing import Any, Mapping

YEAR = re.compile(r"(?<!\d)(20\d{2})[ef]?(?![\d/])")
MONTH_YEAR = re.compile(r"tháng\s+(\d{1,2})\s*(?:/|năm)\s*(20\d{2})", re.I)
BARE_MONTH = re.compile(r"tháng\s+(\d{1,2})(?!\d)(?!\s*(?:/|năm)\s*20\d{2})", re.I)
QUARTER = re.compile(r"quý\s+(IV|I{1,3}|[1-4])\s*(?:/|năm)\s*(20\d{2})", re.I)
FORECAST = re.compile(r"dự báo|dự kiến|kịch bản|(?<!\d)20\d{2}f\b", re.I)
ESTIMATE = re.compile(r"ước tính|ước đạt|(?<!\d)20\d{2}e\b", re.I)
FIGURE_REF = re.compile(r"\bHình\s+(B?\d+(?:\.\d+)?)")
ROMAN_QUARTERS = {"I": "1", "II": "2", "III": "3", "IV": "4"}
HEADER_LEVELS = 2

STRUCTURE_FIELDS = (
    "element_type",
    "heading_path",
    "box_id",
    "box_title",
    "figure_no",
    "table_no",
    "data_source",
    "figure_refs",
    "time_periods",
    "value_types",
    "context_header",
)


def time_periods(text: str) -> list[str]:
    """Return normalized periods: "2026", "2026-04", "2026-Q1", or "M04" for a month without year."""
    periods: set[str] = set()
    for month, year in MONTH_YEAR.findall(text):
        periods.add(f"{year}-{int(month):02d}")
    for month in BARE_MONTH.findall(text):
        if 1 <= int(month) <= 12:
            periods.add(f"M{int(month):02d}")
    for quarter, year in QUARTER.findall(text):
        periods.add(f"{year}-Q{ROMAN_QUARTERS.get(quarter.upper(), quarter)}")
    periods.update(YEAR.findall(text))
    return sorted(periods)


def value_types(text: str) -> list[str]:
    """Return "forecast" and/or "estimate" when the text reports such values."""
    types: list[str] = []
    if FORECAST.search(text):
        types.append("forecast")
    if ESTIMATE.search(text):
        types.append("estimate")
    return types


def figure_refs(text: str) -> list[str]:
    return sorted(set(FIGURE_REF.findall(text)))


def context_header(segment: Mapping[str, Any]) -> str:
    """Return a one-line description of where the chunk sits in the report."""
    parts: list[str] = []
    # Chapter and section only: the third level in this report is a whole sentence
    # ("Tuy nhiên, niềm tin kinh doanh đã suy giảm ...") that already opens the body
    # text, and repeating it on every chunk drowns the chunk's own content.
    heading_path = [str(heading) for heading in segment.get("heading_path", []) if heading][:HEADER_LEVELS]
    if heading_path:
        parts.append("Mục: " + " > ".join(heading_path))
    if segment.get("box_id"):
        parts.append(f"Hộp {segment['box_id']}: {segment.get('box_title', '')}".rstrip(": "))
    if segment.get("figure_no"):
        parts.append(f"Hình {segment['figure_no']}")
    if segment.get("table_no"):
        parts.append(f"Bảng {segment['table_no']}")
    return " | ".join(parts)


def structure_fields(segment: Mapping[str, Any], text: str) -> dict[str, Any]:
    """Return the structured-model metadata of one chunk cut from ``segment``."""
    return {
        "element_type": str(segment.get("element_type", "body")),
        "heading_path": [str(heading) for heading in segment.get("heading_path", [])],
        "box_id": str(segment.get("box_id", "")),
        "box_title": str(segment.get("box_title", "")),
        "figure_no": str(segment.get("figure_no", "")),
        "table_no": str(segment.get("table_no", "")),
        "data_source": str(segment.get("data_source", "")),
        "figure_refs": figure_refs(text),
        "time_periods": time_periods(text),
        "value_types": value_types(text),
        "context_header": context_header(segment),
    }
