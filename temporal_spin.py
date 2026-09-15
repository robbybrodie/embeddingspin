"""
Temporal-Phase Spin Retrieval — Public Surface
==============================================

Hierarchical phase-encoded temporal vectors for semantic embeddings.

What this does
--------------

A frozen embedding model turns text into a semantic vector that captures *what* a
document says but nothing about *when* it says it. Two quarterly revenue reports
three years apart are near-identical in that space. This package encodes the "when"
as a continuous geometric phase on several concurrent circles, and concatenates the
result onto the semantic vector:

    modified_vector = [ semantic_embedding (N-D) , temporal_vector (3 x scales) ]

No retraining, no fine-tuning, no metadata joins. The temporal discrimination is
arithmetic, so it composes with any embedding model and any vector database.

The circles
-----------

Three concurrent periodic scales, each an integer power of two and a factor of the
next:

===========  ========  ==========================  =====================
Scale        Period    Divided into                Resolves
===========  ========  ==========================  =====================
``quarter``     1 y    4 calendar quarters         position within a year
``decade``     16 y    16 calendar years           which year
``century``   256 y    16 sixteen-year blocks      which era
===========  ========  ==========================  =====================

Each contributes ``[cos, sin, z]`` — the arc centre's phase components plus the arc
length — giving a nine-dimensional temporal vector. ``z = 0`` marks a single instant
(*point mode*); ``z > 0`` marks a duration (*arc mode*).

The hierarchy is not fixed at three. It is variable-length and self-describing: see
:class:`~temporal_config.TemporalHierarchy` and its ``extended`` method for adding a
4096-year circle past 2155.

Module map
----------

``temporal_config``
    Epoch, scales, segments, weights, schema version, epoch-migration checks.
``temporal_encoding``
    Intervals, Formula 1, arc algebra, segments, boundary splitting, lazy traversal.
``temporal_spin`` (this module)
    Document/query/result types and timestamp extraction.
``ingestion`` / ``retrieval`` / ``vector_store``
    Pipeline, two-pass search, storage backends.
``query_decomposition``
    Natural-language temporal parsing and multi-subquery decomposition.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from dateutil import parser as dateutil_parser

from temporal_config import (
    CENTURY_SCALE,
    DECADE_SCALE,
    DEFAULT_HIERARCHY,
    EPOCH_1900,
    EPOCH_2010_LEGACY,
    MILLENNIUM_SCALE,
    QUARTER_SCALE,
    SCHEMA_VERSION,
    TAU,
    ScaleSpec,
    TemporalHierarchy,
    describe_epoch_migration,
    epoch_shift_is_congruent,
    floored_mod,
    years_since_epoch,
)
from temporal_encoding import (
    EPS,
    MAX_REPRESENTATIONS,
    ScaleMatch,
    ScaleTuple,
    TemporalEncoding,
    TemporalInterval,
    TraversalPlan,
    angular_difference,
    arc_contains_point,
    arc_overlap,
    encode,
    encode_single,
    encode_tuple,
    evaluate_scale,
    jaccard_arcs,
    pad_to_hierarchy,
    phase_of,
    segment_label,
    segments_intersected,
    split_at_boundaries,
    temporal_alignment,
    traversal_plan,
)

__all__ = [
    # configuration
    "TemporalHierarchy",
    "ScaleSpec",
    "DEFAULT_HIERARCHY",
    "EPOCH_1900",
    "EPOCH_2010_LEGACY",
    "QUARTER_SCALE",
    "DECADE_SCALE",
    "CENTURY_SCALE",
    "MILLENNIUM_SCALE",
    "SCHEMA_VERSION",
    "describe_epoch_migration",
    "epoch_shift_is_congruent",
    # encoding
    "TemporalInterval",
    "TemporalEncoding",
    "ScaleTuple",
    "ScaleMatch",
    "TraversalPlan",
    "encode",
    "encode_single",
    "encode_tuple",
    "phase_of",
    "floored_mod",
    "split_at_boundaries",
    "traversal_plan",
    "evaluate_scale",
    "temporal_alignment",
    "angular_difference",
    "arc_overlap",
    "arc_contains_point",
    "jaccard_arcs",
    "pad_to_hierarchy",
    # documents
    "SpinDocument",
    "SpinQuery",
    "RetrievalResult",
    "extract_timestamp_from_text",
    "extract_interval_from_text",
    "cosine_similarity",
    "normalize_vector",
    "deduplicate_by_group",
]


# ============================================================================
# Timestamp extraction
# ============================================================================

# Common date patterns in financial and corporate documents, most specific first.
DATE_PATTERNS = [
    r"period\s+ended\s+(\d{1,2}\s+\w+\s+\d{4})",
    r"as\s+of\s+(\w+\s+\d{1,2},?\s+\d{4})",
    r"fiscal\s+year\s+(\d{4})",
    r"Q[1-4]\s+(\d{4})",
    r"(\d{4}-\d{2}-\d{2})",
    r"(\d{1,2}/\d{1,2}/\d{4})",
]

_QUARTER_RE = re.compile(r"\bQ([1-4])[\s,/-]*((?:FY)?\s*\d{4})\b", re.IGNORECASE)
_FY_RE = re.compile(r"\b(?:fiscal\s+year|FY)\s*(\d{4})\b", re.IGNORECASE)
_YEAR_RANGE_RE = re.compile(r"\b(\d{4})\s*(?:-|–|—|to|through)\s*(\d{4})\b")
_BARE_YEAR_RE = re.compile(r"\b(19|20|21)(\d{2})\b")


def extract_timestamp_from_text(
    text: str, fallback: Optional[datetime] = None
) -> datetime:
    """
    Best-effort single timestamp from document text.

    Tries the corporate/financial date patterns first, then dateutil fuzzy parsing,
    then the supplied fallback. Prefer :func:`extract_interval_from_text` when the
    document describes a *period* rather than an instant — encoding a quarterly
    report as a point discards the duration that makes hierarchical matching work.
    """
    for pattern in DATE_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                dt = dateutil_parser.parse(match.group(1), fuzzy=True)
                return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
            except (ValueError, TypeError, OverflowError):
                continue

    try:
        dt = dateutil_parser.parse(text[:500], fuzzy=True)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError, OverflowError):
        pass

    return fallback if fallback else datetime.now(timezone.utc)


def extract_interval_from_text(
    text: str, fallback: Optional[TemporalInterval] = None
) -> Optional[TemporalInterval]:
    """
    Extract a half-open :class:`TemporalInterval` from text.

    Recognises, in order of specificity: explicit quarters (``Q3 2023``), fiscal
    years (``FY2021``), year ranges (``2017-2022``), and bare four-digit years.
    Returns ``None`` if nothing temporal is found and no fallback is given, which
    lets the caller decide between point mode and skipping the document.
    """
    match = _QUARTER_RE.search(text)
    if match:
        year_token = re.sub(r"\D", "", match.group(2))
        if len(year_token) == 4:
            return TemporalInterval.of_quarter(int(year_token), int(match.group(1)))

    match = _FY_RE.search(text)
    if match:
        return TemporalInterval.of_year(int(match.group(1)))

    match = _YEAR_RANGE_RE.search(text)
    if match:
        first, last = int(match.group(1)), int(match.group(2))
        if first <= last:
            return TemporalInterval.spanning(first, last)

    match = _BARE_YEAR_RE.search(text)
    if match:
        return TemporalInterval.of_year(int(match.group(0)))

    return fallback


# ============================================================================
# Documents and queries
# ============================================================================


@dataclass
class SpinDocument:
    """
    One indexed representation of a chunk.

    A chunk whose interval crosses a period boundary produces several
    ``SpinDocument`` rows — distinct ``doc_id`` values, identical ``group_id``,
    identical semantic embedding, different temporal tuples. Deduplication on
    ``group_id`` happens at the application layer so the consumer sees the chunk
    once; see :func:`deduplicate_by_group`.

    Attributes:
        doc_id: Unique per representation. Suffixed ``#k`` when split.
        group_id: Shared across every representation of one source chunk.
        text: Chunk text.
        encoding: The temporal vector, tuples and header.
        semantic_embedding: Output of the frozen embedding model.
        full_embedding: ``semantic_embedding`` with the temporal vector appended.
        metadata: Arbitrary caller metadata.
    """

    doc_id: str
    text: str
    semantic_embedding: List[float]
    encoding: TemporalEncoding
    full_embedding: List[float] = field(default_factory=list)
    group_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.group_id:
            self.group_id = self.encoding.group_id
        if not self.full_embedding:
            self.full_embedding = list(self.semantic_embedding) + self.encoding.to_vector()

    # -- convenience -------------------------------------------------------

    @property
    def interval(self) -> TemporalInterval:
        """The component interval this representation covers."""
        return self.encoding.interval

    @property
    def source_interval(self) -> TemporalInterval:
        """The full interval of the underlying chunk, before any splitting."""
        return self.encoding.source_interval

    @property
    def timestamp(self) -> datetime:
        return self.encoding.interval.start

    @property
    def end_timestamp(self) -> Optional[datetime]:
        return self.encoding.interval.end

    @property
    def is_arc(self) -> bool:
        return not self.encoding.is_point

    @property
    def is_split(self) -> bool:
        return self.encoding.is_split

    @property
    def spin_vector(self) -> List[float]:
        return self.encoding.to_vector()

    @property
    def phi(self) -> Dict[str, float]:
        """Centre phase per scale, keyed by scale name."""
        return {t.scale_name: t.phi_center for t in self.encoding.tuples}

    @property
    def z(self) -> Dict[str, float]:
        """Arc length per scale, keyed by scale name."""
        return {t.scale_name: t.z for t in self.encoding.tuples}


@dataclass
class SpinQuery:
    """
    An encoded query: semantic embedding plus a query temporal vector.

    Queries are never split into multiple representations — a query arc is compared
    against stored arcs directly. ``lambda_factor`` scales the temporal block during
    the coarse first pass so that semantic similarity stays dominant while candidates
    are gathered.
    """

    query_text: str
    semantic_embedding: List[float]
    encoding: TemporalEncoding
    lambda_factor: float = 0.1
    full_embedding: List[float] = field(default_factory=list)
    plan: Optional[TraversalPlan] = None

    def __post_init__(self) -> None:
        if not self.full_embedding:
            weighted = [self.lambda_factor * x for x in self.encoding.to_vector()]
            self.full_embedding = list(self.semantic_embedding) + weighted
        if self.plan is None:
            self.plan = traversal_plan(self.encoding)

    @property
    def interval(self) -> TemporalInterval:
        return self.encoding.interval

    @property
    def is_arc(self) -> bool:
        return not self.encoding.is_point

    @property
    def query_timestamp(self) -> datetime:
        return self.encoding.interval.start

    @property
    def end_timestamp(self) -> Optional[datetime]:
        return self.encoding.interval.end

    @property
    def phi(self) -> Dict[str, float]:
        return {t.scale_name: t.phi_center for t in self.encoding.tuples}


@dataclass
class RetrievalResult:
    """
    A scored, ranked hit.

    Attributes:
        semantic_score: Cosine similarity of the semantic blocks alone.
        temporal_alignment: Weighted per-scale alignment over the traversed scales.
        combined_score: ``(1 - beta) * semantic + beta * temporal``, before any
            metadata priority multiplier.
        scale_matches: Per-scale detail for the scales that were actually traversed.
        traversed_scales: Which circles the lazy plan descended into.
        rejected_at: Scale name that caused a hard rejection, if any.
    """

    doc_id: str
    group_id: str
    text: str
    interval: TemporalInterval
    semantic_score: float
    temporal_alignment: float
    combined_score: float
    scale_matches: Dict[str, ScaleMatch] = field(default_factory=dict)
    traversed_scales: Tuple[str, ...] = ()
    rejected_at: Optional[str] = None
    rank: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    # -- convenience -------------------------------------------------------

    @property
    def timestamp(self) -> datetime:
        return self.interval.start

    @property
    def phi_difference(self) -> float:
        """Angular distance on the primary discriminating scale, for display."""
        for name in ("decade", *self.traversed_scales):
            if name in self.scale_matches:
                return self.scale_matches[name].delta_phi
        return 0.0

    def explain(self) -> str:
        lines = [
            f"Rank #{self.rank}  {self.doc_id}",
            f"  interval  : {self.interval}",
            f"  semantic  : {self.semantic_score:.4f}",
            f"  temporal  : {self.temporal_alignment:.4f}",
            f"  combined  : {self.combined_score:.4f}",
            f"  traversed : {', '.join(self.traversed_scales) or '(none)'}",
        ]
        for name in self.traversed_scales:
            m = self.scale_matches.get(name)
            if m is None:
                continue
            segs = ",".join(str(s) for s in m.shared_segments) or "-"
            lines.append(
                f"    {name:<10} overlap={'yes' if m.overlaps else 'no ':<3} "
                f"jaccard={m.jaccard:.4f} dphi={math.degrees(m.delta_phi):6.2f}deg "
                f"segments[{segs}]"
            )
        if self.rejected_at:
            lines.append(f"  REJECTED at {self.rejected_at}")
        preview = self.text.replace("\n", " ")[:160]
        lines.append(f"  text      : {preview}")
        return "\n".join(lines)


# ============================================================================
# Deduplication
# ============================================================================


def deduplicate_by_group(results: List[RetrievalResult]) -> List[RetrievalResult]:
    """
    Collapse multiple representations of one chunk down to its best-scoring hit.

    A chunk split across period boundaries is indexed several times, and a query arc
    straddling a divider can match more than one of those representations through
    different component indexing paths. Deduplicating on ``group_id`` at the
    application layer — rather than trying to prevent the multiple matches
    geometrically — is what lets the index stay in a single unified coordinate space
    while the consumer still receives each chunk exactly once.

    Input order is not required to be sorted; the highest ``combined_score`` wins and
    the surviving order follows the input.
    """
    best: Dict[str, RetrievalResult] = {}
    order: List[str] = []
    for result in results:
        key = result.group_id or result.doc_id
        if key not in best:
            best[key] = result
            order.append(key)
        elif result.combined_score > best[key].combined_score:
            best[key] = result
    return [best[k] for k in order]


# ============================================================================
# Vector utilities
# ============================================================================


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Cosine similarity in ``[-1, 1]``. Returns 0.0 if either vector is empty."""
    if not vec1 or not vec2:
        return 0.0
    if len(vec1) != len(vec2):
        raise ValueError(
            f"vectors must have same dimension, got {len(vec1)} and {len(vec2)}"
        )
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


def normalize_vector(vec: List[float]) -> List[float]:
    """L2-normalise to unit length. A zero vector is returned unchanged."""
    norm = math.sqrt(sum(x * x for x in vec))
    return vec if norm == 0 else [x / norm for x in vec]


# ============================================================================
# Backwards compatibility (schema version 1)
# ============================================================================

#: Legacy epoch constant. Schema v1 encoded against 2010-01-01; new corpora use
#: :data:`temporal_config.EPOCH_1900`. Kept so v1 code paths keep importing.
T0_EPOCH = EPOCH_2010_LEGACY
T0_SECONDS = EPOCH_2010_LEGACY.timestamp()

QUARTER_SCALE_YEARS = QUARTER_SCALE.period_years
DECADE_SCALE_YEARS = DECADE_SCALE.period_years
CENTURY_SCALE_YEARS = CENTURY_SCALE.period_years

QUARTER_WEIGHT = QUARTER_SCALE.weight
DECADE_WEIGHT = DECADE_SCALE.weight
CENTURY_WEIGHT = CENTURY_SCALE.weight


def compute_spin_vector(
    timestamp_seconds: float,
    t0_seconds: float = None,
    period_seconds: float = None,
    phase_offset: float = 0.0,
    temporal_scale: float = 1.0,
    end_timestamp_seconds: Optional[float] = None,
    hierarchy: TemporalHierarchy = DEFAULT_HIERARCHY,
) -> Tuple[List[float], Dict[str, float], Dict[str, Optional[float]], Dict[str, Optional[float]]]:
    """
    Schema-v1 compatibility shim.

    Returns the same ``(spin_vector, phi_centers, phi_starts, phi_ends)`` tuple the
    previous release did, computed through the current encoder. New code should call
    :func:`temporal_encoding.encode` and work with :class:`TemporalEncoding`, which
    carries the header, segments and group identity this signature cannot express.

    ``period_seconds`` and ``phase_offset`` are ignored; ``t0_seconds`` overrides the
    hierarchy epoch when supplied.
    """
    if t0_seconds is not None:
        hierarchy = TemporalHierarchy(
            epoch=datetime.fromtimestamp(t0_seconds, tz=timezone.utc),
            scales=hierarchy.scales,
            year_convention=hierarchy.year_convention,
        )

    start = datetime.fromtimestamp(timestamp_seconds, tz=timezone.utc)
    end = (
        datetime.fromtimestamp(end_timestamp_seconds, tz=timezone.utc)
        if end_timestamp_seconds is not None
        else None
    )
    encoding = encode_single(TemporalInterval(start, end), hierarchy)

    vector: List[float] = []
    centers: Dict[str, float] = {}
    starts: Dict[str, Optional[float]] = {}
    ends: Dict[str, Optional[float]] = {}
    for t in encoding.tuples:
        vector.extend([temporal_scale * t.cos, temporal_scale * t.sin, t.z])
        centers[t.scale_name] = t.phi_center
        starts[t.scale_name] = None if t.is_point else t.phi_start
        ends[t.scale_name] = None if t.is_point else t.phi_end
    return vector, centers, starts, ends
