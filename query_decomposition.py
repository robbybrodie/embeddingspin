"""
Natural-Language Temporal Query Decomposition
=============================================

One question often carries several temporal constraints. *"What was the Q1 impact
on the full year for 2021, 2022 and 2023?"* is not one query — it is six: the first
quarter of each of three years, and the whole of each of those three years. Issued
as a single query it would have to pick one interval and lose the rest.

This module parses the temporal expressions out of a natural-language query, expands
them into independent sub-queries, and runs them in parallel against the retriever.
The semantic text is passed through unchanged to every sub-query; only the temporal
constraint differs. Results are merged and deduplicated on ``group_id``, so a chunk
that satisfies several sub-queries is returned once, at its best score.

Two axes are parsed separately and then crossed:

**Anchors** — which years the question is about (explicit years, ranges, or relative
expressions like "the last three years").

**Granularities** — at what resolution (a named quarter, a month, the full year). A
query naming both a quarter and the full year asks for both, which is where the
multiplication comes from.

Nothing here changes the index. Decomposition is a query-time concern; the encoding
a chunk received at ingestion is untouched.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from temporal_encoding import TemporalInterval

logger = logging.getLogger(__name__)

#: Ceiling on sub-queries produced from one natural-language query. A range like
#: "1995 through 2024" crossed with four quarters would otherwise fan out to 120
#: retrieval calls.
MAX_SUBQUERIES = 24


# ============================================================================
# Patterns
# ============================================================================

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12, "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7,
    "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}

_ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "1st": 1, "2nd": 2,
             "3rd": 3, "4th": 4}

# Q3 2023, Q3 FY2023, Q3/2023
_QUARTER_YEAR_RE = re.compile(r"\bQ([1-4])[\s,/-]*(?:FY)?\s*((?:19|20|21)\d{2})\b", re.IGNORECASE)
# "third quarter of 2023"
_ORDINAL_QUARTER_YEAR_RE = re.compile(
    r"\b(first|second|third|fourth|1st|2nd|3rd|4th)\s+quarter\s+(?:of\s+)?((?:19|20|21)\d{2})\b",
    re.IGNORECASE,
)
# A quarter named without a year — the year comes from the anchors.
_BARE_QUARTER_RE = re.compile(r"\bQ([1-4])\b", re.IGNORECASE)
_BARE_ORDINAL_QUARTER_RE = re.compile(
    r"\b(first|second|third|fourth|1st|2nd|3rd|4th)\s+quarter\b", re.IGNORECASE
)

_MONTH_YEAR_RE = re.compile(
    r"\b(" + "|".join(_MONTHS) + r")\s+((?:19|20|21)\d{2})\b", re.IGNORECASE
)

# "and" only joins a range when introduced by "between"/"from". Bare "2022 and
# 2023" is a list of two years, not a span — "for 2021, 2022 and 2023" must not
# collapse into a single 2022–2023 arc.
_BETWEEN_RE = re.compile(
    r"\b(?:between|from)\s+((?:19|20|21)\d{2})\s*(?:-|–|—|to|through|until|and)\s*"
    r"((?:19|20|21)\d{2})\b",
    re.IGNORECASE,
)
_RANGE_RE = re.compile(
    r"\b((?:19|20|21)\d{2})\s*(?:-|–|—|to|through|until)\s*((?:19|20|21)\d{2})\b",
    re.IGNORECASE,
)
_SINCE_RE = re.compile(r"\b(?:since|after|from)\s+((?:19|20|21)\d{2})\b", re.IGNORECASE)
_BEFORE_RE = re.compile(r"\b(?:before|prior\s+to|up\s+to|until)\s+((?:19|20|21)\d{2})\b", re.IGNORECASE)
_LAST_N_RE = re.compile(
    r"\b(?:last|past|previous|trailing)\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
    r"(year|quarter|month|decade)s?\b",
    re.IGNORECASE,
)
_YEAR_RE = re.compile(r"\b((?:19|20|21)\d{2})\b")

_ANNUAL_RE = re.compile(
    r"\b(full[-\s]?year|whole\s+year|annual(?:ly)?|year[-\s]?end|fiscal\s+year|FY(?=\d|\b))\b",
    re.IGNORECASE,
)
# "each year", "year by year" — expand a range instead of treating it as one arc.
_PER_YEAR_RE = re.compile(
    r"\b(each\s+year|every\s+year|year[-\s]by[-\s]year|per\s+year|yearly|year\s+over\s+year|annually)\b",
    re.IGNORECASE,
)

_NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
                 "seven": 7, "eight": 8, "nine": 9, "ten": 10}


# ============================================================================
# Types
# ============================================================================


@dataclass(frozen=True)
class SubQuery:
    """
    One temporally-constrained retrieval to issue.

    ``text`` is the original query, unmodified — the semantic half of the search is
    identical across sub-queries, and only ``interval`` distinguishes them. ``label``
    is a human-readable tag ("Q1 2022", "FY2023") for tracing and result grouping.
    """

    text: str
    interval: Optional[TemporalInterval]
    label: str

    def as_pair(self) -> Tuple[str, Optional[TemporalInterval]]:
        """The ``(text, interval)`` form :meth:`TemporalSpinRetriever.search_many` takes."""
        return (self.text, self.interval)


@dataclass(frozen=True)
class Decomposition:
    """
    The result of parsing one natural-language query.

    ``subqueries`` always has at least one entry. When no temporal expression is
    found it holds a single unconstrained sub-query, so callers can treat the
    decomposed path as the only path.
    """

    query_text: str
    subqueries: Tuple[SubQuery, ...]
    anchors: Tuple[int, ...] = ()
    granularities: Tuple[str, ...] = ()
    truncated: bool = False

    def __len__(self) -> int:
        return len(self.subqueries)

    def __iter__(self):
        return iter(self.subqueries)

    @property
    def is_temporal(self) -> bool:
        """True when at least one sub-query carries a temporal constraint."""
        return any(s.interval is not None for s in self.subqueries)

    def as_pairs(self) -> List[Tuple[str, Optional[TemporalInterval]]]:
        return [s.as_pair() for s in self.subqueries]

    def describe(self) -> str:
        lines = [f"{self.query_text!r} -> {len(self.subqueries)} sub-quer"
                 f"{'y' if len(self.subqueries) == 1 else 'ies'}"]
        if self.anchors:
            lines.append(f"  anchors      : {', '.join(str(a) for a in self.anchors)}")
        if self.granularities:
            lines.append(f"  granularities: {', '.join(self.granularities)}")
        for sub in self.subqueries:
            lines.append(f"    - {sub.label:<16} {sub.interval}")
        if self.truncated:
            lines.append(f"  (truncated at {MAX_SUBQUERIES})")
        return "\n".join(lines)


# ============================================================================
# Parsing
# ============================================================================


def _relative_interval(
    count: int, unit: str, reference: datetime
) -> Optional[TemporalInterval]:
    """Turn "last three years" into a half-open interval ending at ``reference``."""
    years = {"year": 1, "decade": 10, "quarter": 0.25, "month": 1 / 12}.get(unit.lower())
    if years is None:
        return None
    span_days = count * years * 365.2425
    start = datetime.fromtimestamp(
        reference.timestamp() - span_days * 86400, tz=timezone.utc
    )
    return TemporalInterval(start, reference)


def _collect_anchors(
    text: str, reference: datetime
) -> Tuple[List[int], List[Tuple[TemporalInterval, str]], bool]:
    """
    Find the years a query is about.

    Returns ``(years, standalone, expand_per_year)``. Standalone entries are
    ``(interval, label)`` pairs for ranges and relative expressions that are *not*
    being expanded year by year; they become sub-queries directly rather than being
    crossed with granularities.
    """
    years: List[int] = []
    standalone: List[Tuple[TemporalInterval, str]] = []
    per_year = bool(_PER_YEAR_RE.search(text))

    range_match = _BETWEEN_RE.search(text) or _RANGE_RE.search(text)
    if range_match:
        first, last = int(range_match.group(1)), int(range_match.group(2))
        if first > last:
            first, last = last, first
        if per_year or (last - first) < 1:
            years.extend(range(first, last + 1))
        else:
            # A span asked about as a whole stays one arc. Splitting it into
            # per-year sub-queries would be a different question.
            standalone.append((TemporalInterval.spanning(first, last), f"{first}-{last}"))
        return years, standalone, per_year

    since = _SINCE_RE.search(text)
    if since:
        first = int(since.group(1))
        last = max(first, reference.year)
        if per_year:
            years.extend(range(first, last + 1))
        else:
            standalone.append((TemporalInterval.spanning(first, last), f"{first}-{last}"))
        return years, standalone, per_year

    before = _BEFORE_RE.search(text)
    if before:
        last = int(before.group(1)) - 1
        first = last - 15
        standalone.append((TemporalInterval.spanning(first, last), f"pre-{last + 1}"))
        return years, standalone, per_year

    last_n = _LAST_N_RE.search(text)
    if last_n:
        token = last_n.group(1).lower()
        count = _NUMBER_WORDS.get(token) or int(token)
        unit = last_n.group(2).lower()
        if per_year and unit == "year":
            years.extend(range(reference.year - count + 1, reference.year + 1))
        else:
            interval = _relative_interval(count, unit, reference)
            if interval:
                standalone.append((interval, f"last {count} {unit}s"))
        return years, standalone, per_year

    # Plain years mentioned anywhere: "for 2021, 2022 and 2023".
    for match in _YEAR_RE.finditer(text):
        year = int(match.group(1))
        if year not in years:
            years.append(year)

    return years, standalone, per_year


def _collect_granularities(text: str) -> Tuple[List[int], List[int], bool]:
    """
    Find the resolutions a query asks for.

    Returns ``(quarters, months, wants_annual)``. Quarters and months here are those
    named *without* a year — they need an anchor year to become an interval.
    """
    quarters: List[int] = []
    for match in _BARE_QUARTER_RE.finditer(text):
        q = int(match.group(1))
        if q not in quarters:
            quarters.append(q)
    for match in _BARE_ORDINAL_QUARTER_RE.finditer(text):
        q = _ORDINALS[match.group(1).lower()]
        if q not in quarters:
            quarters.append(q)

    months: List[int] = []
    wants_annual = bool(_ANNUAL_RE.search(text))
    return quarters, months, wants_annual


def decompose(
    query_text: str,
    reference: Optional[datetime] = None,
    max_subqueries: int = MAX_SUBQUERIES,
) -> Decomposition:
    """
    Parse a natural-language query into independent temporal sub-queries.

    Args:
        query_text: The user's question, verbatim.
        reference: "Now" for relative expressions. Defaults to the current time.
        max_subqueries: Fan-out ceiling.

    Returns:
        A :class:`Decomposition` holding at least one sub-query.

    Examples:
        "Q1 impact on the full year for 2021, 2022 and 2023" produces six
        sub-queries: Q1 of each year, and each full year.

        "revenue between 2017 and 2022" produces one, spanning the range — the
        question is about the span, not about each year in it. Adding "year by year"
        switches it to six.
    """
    reference = reference or datetime.now(timezone.utc)
    subqueries: List[SubQuery] = []
    seen: set = set()

    def add(interval: Optional[TemporalInterval], label: str) -> None:
        key = (
            None
            if interval is None
            else (interval.start, interval.end)
        )
        if key in seen:
            return
        seen.add(key)
        subqueries.append(SubQuery(text=query_text, interval=interval, label=label))

    # 1. Fully-specified expressions first — these need no anchor and are the most
    #    precise reading of the query.
    explicit_years: set = set()
    for match in _QUARTER_YEAR_RE.finditer(query_text):
        quarter, year = int(match.group(1)), int(match.group(2))
        add(TemporalInterval.of_quarter(year, quarter), f"Q{quarter} {year}")
        explicit_years.add(year)
    for match in _ORDINAL_QUARTER_YEAR_RE.finditer(query_text):
        quarter = _ORDINALS[match.group(1).lower()]
        year = int(match.group(2))
        add(TemporalInterval.of_quarter(year, quarter), f"Q{quarter} {year}")
        explicit_years.add(year)
    for match in _MONTH_YEAR_RE.finditer(query_text):
        month = _MONTHS[match.group(1).lower()]
        year = int(match.group(2))
        add(TemporalInterval.of_month(year, month), f"{match.group(1).title()} {year}")
        explicit_years.add(year)

    # 2. Anchors and granularities, crossed.
    anchors, standalone, per_year = _collect_anchors(query_text, reference)
    quarters, _months, wants_annual = _collect_granularities(query_text)

    # A year already consumed by "Q3 2023" is not also a bare anchor.
    anchors = [y for y in anchors if y not in explicit_years]

    for interval, label in standalone:
        add(interval, label)

    for year in anchors:
        if quarters:
            for quarter in quarters:
                add(TemporalInterval.of_quarter(year, quarter), f"Q{quarter} {year}")
        if wants_annual or not quarters:
            add(TemporalInterval.of_year(year), f"FY{year}")

    # A quarter named with no year at all: apply it to the reference year.
    if quarters and not anchors and not explicit_years and not standalone:
        for quarter in quarters:
            add(
                TemporalInterval.of_quarter(reference.year, quarter),
                f"Q{quarter} {reference.year}",
            )

    # 3. Nothing temporal — one unconstrained sub-query.
    if not subqueries:
        add(None, "unconstrained")

    truncated = len(subqueries) > max_subqueries
    if truncated:
        logger.warning(
            "decomposition of %r produced %d sub-queries, truncating to %d",
            query_text[:60], len(subqueries), max_subqueries,
        )
        subqueries = subqueries[:max_subqueries]

    return Decomposition(
        query_text=query_text,
        subqueries=tuple(subqueries),
        anchors=tuple(sorted(explicit_years | set(anchors))),
        granularities=tuple(
            ([f"Q{q}" for q in quarters]) + (["annual"] if wants_annual else [])
        ),
        truncated=truncated,
    )


# ============================================================================
# Execution
# ============================================================================


def search_decomposed(
    retriever,
    query_text: str,
    beta: Optional[float] = None,
    top_k_final: int = 10,
    reference: Optional[datetime] = None,
    max_subqueries: int = MAX_SUBQUERIES,
    max_workers: int = 8,
    **kwargs: Any,
) -> Tuple[List, Decomposition]:
    """
    Decompose a query, run every sub-query in parallel, and merge the results.

    Merging deduplicates on ``group_id``: a chunk matched by several sub-queries —
    common when a query asks about both a quarter and the year containing it —
    appears once, at its highest score. Each surviving result gains a
    ``matched_subqueries`` metadata entry listing the labels that found it, which is
    what lets a caller see *why* a document answered a multi-part question.

    Returns:
        ``(results, decomposition)``. Results are re-ranked from 1.
    """
    from temporal_spin import deduplicate_by_group

    decomposition = decompose(query_text, reference=reference, max_subqueries=max_subqueries)
    per_subquery = retriever.search_many(
        decomposition.as_pairs(),
        beta=beta,
        top_k_final=top_k_final,
        max_workers=max_workers,
        **kwargs,
    )

    flattened: List = []
    labels: Dict[str, List[str]] = {}
    for sub, results in zip(decomposition.subqueries, per_subquery):
        for result in results:
            key = result.group_id or result.doc_id
            labels.setdefault(key, []).append(sub.label)
            flattened.append(result)

    merged = deduplicate_by_group(flattened)
    merged.sort(key=lambda r: r.combined_score, reverse=True)
    merged = merged[:top_k_final]

    for index, result in enumerate(merged):
        result.rank = index + 1
        key = result.group_id or result.doc_id
        result.metadata = dict(result.metadata or {})
        result.metadata["matched_subqueries"] = labels.get(key, [])

    logger.info(
        "decomposed %r into %d sub-queries -> %d merged result(s)",
        query_text[:60], len(decomposition), len(merged),
    )
    return merged, decomposition
