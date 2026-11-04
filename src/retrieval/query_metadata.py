"""Read what a question asks about the report's structure, time and value type.

Two kinds of signal are used at retrieval time:

- explicit structure ("Hộp 2", "Hình 16", "Bảng 3") is a hard filter: only
  chunks of that box, figure or table (or body text citing the figure) are
  candidates;
- time ("tháng 4/2026"), value type ("dự báo") and a request for the source
  ("nguồn") are soft: matching candidates move up, nothing is removed.
"""

import re
from dataclasses import dataclass
from typing import Any, Mapping

from ..chunking.metadata import time_periods


BOX_QUERY = re.compile(r"\bhộp\s+(\d+)", re.I)
FIGURE_QUERY = re.compile(r"\bhình\s+(B?\d+(?:\.\d+)?)", re.I)
TABLE_QUERY = re.compile(r"\bbảng\s+(\d+)", re.I)
SOURCE_QUERY = re.compile(r"\bnguồn\b", re.I)
FORECAST_QUERY = re.compile(r"dự báo|dự kiến|kịch bản", re.I)
ESTIMATE_QUERY = re.compile(r"ước tính", re.I)
MONTH_PERIOD = re.compile(r"^(?:(20\d{2})-|M)(\d{2})$")

# Weights of the soft signals; a month is more specific than a year.
MONTH_WEIGHT = 2.0
QUARTER_WEIGHT = 2.0
YEAR_WEIGHT = 0.5
VALUE_TYPE_WEIGHT = 1.0
SOURCE_WEIGHT = 1.0


@dataclass(frozen=True)
class QueryMetadata:
    box_id: str = ""
    figure_no: str = ""
    table_no: str = ""
    time_periods: tuple[str, ...] = ()
    value_types: tuple[str, ...] = ()
    wants_source: bool = False

    @property
    def has_structure(self) -> bool:
        return bool(self.box_id or self.figure_no or self.table_no)

    @property
    def has_soft_signals(self) -> bool:
        return bool(self.time_periods or self.value_types or self.wants_source)


def parse_query(question: str) -> QueryMetadata:
    box = BOX_QUERY.search(question)
    figure = FIGURE_QUERY.search(question)
    table = TABLE_QUERY.search(question)
    value_types = tuple(
        name
        for name, pattern in (("forecast", FORECAST_QUERY), ("estimate", ESTIMATE_QUERY))
        if pattern.search(question)
    )
    return QueryMetadata(
        box_id=box.group(1) if box else "",
        figure_no=figure.group(1).upper() if figure else "",
        table_no=table.group(1) if table else "",
        time_periods=tuple(time_periods(question)),
        value_types=value_types,
        wants_source=bool(SOURCE_QUERY.search(question)),
    )


def matches_structure(chunk: Mapping[str, Any], query: QueryMetadata) -> bool:
    """Return True when the chunk belongs to the box / figure / table the question names."""
    if query.box_id and str(chunk.get("box_id", "")) != query.box_id:
        return False
    if query.figure_no:
        own = str(chunk.get("figure_no", "")).upper() == query.figure_no
        cited = query.figure_no in {str(ref).upper() for ref in chunk.get("figure_refs", [])}
        if not (own or cited):
            return False
    if query.table_no:
        own = str(chunk.get("table_no", "")) == query.table_no
        cited = re.search(rf"\bBảng\s+{query.table_no}\b", str(chunk.get("text", ""))) is not None
        if not (own or cited):
            return False
    return True


def soft_score(chunk: Mapping[str, Any], query: QueryMetadata) -> float:
    """Score how well the chunk's time, value type and source fit the question."""
    chunk_periods = set(chunk.get("time_periods", []))
    chunk_months = {_month(period) for period in chunk_periods} - {None}
    score = 0.0
    for period in query.time_periods:
        month = _month(period)
        if month is not None:
            year, number = month
            # "tháng 4" without a year matches "tháng 4/2026" and the other way round.
            if any(
                number == other_number and (year is None or other_year is None or year == other_year)
                for other_year, other_number in chunk_months
            ):
                score += MONTH_WEIGHT
        elif "-Q" in period:
            score += QUARTER_WEIGHT if period in chunk_periods else 0.0
        elif period in chunk_periods:
            score += YEAR_WEIGHT
    chunk_types = set(chunk.get("value_types", []))
    score += VALUE_TYPE_WEIGHT * len(chunk_types.intersection(query.value_types))
    if query.wants_source and chunk.get("data_source"):
        score += SOURCE_WEIGHT
    return score


def _month(period: str) -> tuple[str | None, str] | None:
    match = MONTH_PERIOD.match(period)
    if not match:
        return None
    return match.group(1), match.group(2)
