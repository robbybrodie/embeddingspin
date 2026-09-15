"""
Hierarchical Phase-Encoded Temporal Vectors
===========================================

Core geometry. Maps a moment or an interval onto every circle in a
:class:`~temporal_config.TemporalHierarchy`, producing one ``[cos, sin, z]`` tuple
per scale.

Interval convention
-------------------

**All intervals are half-open: ``[start, end)``.** The start moment belongs to the
interval, the end moment does not. Q1 2026 is therefore
``2026-01-01T00:00:00Z <= t < 2026-04-01T00:00:00Z`` — a duration of exactly 90 days,
whose midpoint is 2026-02-15T00:00:00Z (45.0 elapsed days after the start).

This convention is applied without exception: durations are always
``end - start``, midpoints are always ``start + duration/2``, and adjacent periods
(Q1 ending, Q2 starting) share a boundary moment that belongs to exactly one of
them. Mixing half-open durations with inclusive ordinal day counts is what produces
the classic off-by-one where "45 days after 1 January" and "the 45th day of the
year" are quietly treated as the same instant; they are not.

The three components
--------------------

For each scale, the encoding emits:

``cos``, ``sin``
    The arc's **centre** phase on that circle. Storing both components rather than a
    bare angle is what makes the end of a period geometrically adjacent to the start
    of the next one.

``z``
    The arc length, in radians, of the interval on that circle. ``z == 0`` means a
    single instant (*point mode*); ``z > 0`` means a duration (*arc mode*). ``z`` is
    capped at 2π so an interval longer than the period saturates the circle rather
    than wrapping around it unboundedly.

Boundary splitting
------------------

An interval that crosses a circle's zero-degree boundary cannot be represented as a
single centred arc without ambiguity. :func:`encode` therefore splits such an
interval into component arcs at the finest scale's period boundaries and emits one
representation per component. Every representation carries an identical semantic
payload and a shared ``group_id``, and the application layer deduplicates on that
identifier so a chunk is returned to the consumer exactly once.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from temporal_config import (
    DEFAULT_HIERARCHY,
    TAU,
    ScaleSpec,
    TemporalHierarchy,
    calendar_year_position,
    floored_mod,
    years_since_epoch,
)

# Angular slop. Phases are O(1) radians, so 1e-9 rad (~2e-8 degrees, well under a
# millisecond of a one-year circle) is far below any meaningful temporal resolution
# while comfortably above double-precision noise.
EPS = 1e-9

#: Hard ceiling on how many representations a single interval may be split into.
#: A century-long document at a 1-year finest scale would otherwise emit 100 rows.
MAX_REPRESENTATIONS = 64


# ============================================================================
# Intervals
# ============================================================================


@dataclass(frozen=True)
class TemporalInterval:
    """
    A half-open span of time, ``[start, end)``.

    A *point* is an interval whose ``end`` is ``None`` (or equal to ``start``):
    a single instant, encoded with ``z = 0`` at every scale.
    """

    start: datetime
    end: Optional[datetime] = None

    def __post_init__(self) -> None:
        start = self.start
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
            object.__setattr__(self, "start", start)
        end = self.end
        if end is not None:
            if end.tzinfo is None:
                end = end.replace(tzinfo=timezone.utc)
                object.__setattr__(self, "end", end)
            if end < start:
                raise ValueError(f"interval end {end} precedes start {start}")
            if end == start:
                object.__setattr__(self, "end", None)

    @property
    def is_point(self) -> bool:
        return self.end is None

    @property
    def duration(self) -> timedelta:
        return timedelta(0) if self.end is None else self.end - self.start

    @property
    def duration_days(self) -> float:
        return self.duration.total_seconds() / 86400.0

    @property
    def midpoint(self) -> datetime:
        """
        Centre of the half-open interval: ``start + (end - start) / 2``.

        For Q1 2026 (``[2026-01-01, 2026-04-01)``, 90 days) this is
        2026-02-15T00:00:00Z — 45.0 elapsed days after the start, *not* 14 February.
        """
        return self.start if self.end is None else self.start + self.duration / 2

    def years_span(self, hierarchy: TemporalHierarchy = DEFAULT_HIERARCHY) -> float:
        """Duration expressed in years under the hierarchy's year convention."""
        if self.end is None:
            return 0.0
        return years_since_epoch(self.end, self.start, hierarchy.year_convention)

    @classmethod
    def point(cls, moment: datetime) -> "TemporalInterval":
        return cls(start=moment, end=None)

    @classmethod
    def of_year(cls, year: int) -> "TemporalInterval":
        return cls(
            datetime(year, 1, 1, tzinfo=timezone.utc),
            datetime(year + 1, 1, 1, tzinfo=timezone.utc),
        )

    @classmethod
    def of_quarter(cls, year: int, quarter: int) -> "TemporalInterval":
        """Calendar quarter, half-open. ``of_quarter(2026, 1)`` is [Jan 1, Apr 1)."""
        if not 1 <= quarter <= 4:
            raise ValueError(f"quarter must be 1-4, got {quarter}")
        start_month = 3 * (quarter - 1) + 1
        start = datetime(year, start_month, 1, tzinfo=timezone.utc)
        if quarter == 4:
            end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
        else:
            end = datetime(year, start_month + 3, 1, tzinfo=timezone.utc)
        return cls(start, end)

    @classmethod
    def of_month(cls, year: int, month: int) -> "TemporalInterval":
        start = datetime(year, month, 1, tzinfo=timezone.utc)
        end = (
            datetime(year + 1, 1, 1, tzinfo=timezone.utc)
            if month == 12
            else datetime(year, month + 1, 1, tzinfo=timezone.utc)
        )
        return cls(start, end)

    @classmethod
    def spanning(cls, first_year: int, last_year: int) -> "TemporalInterval":
        """Inclusive range of calendar years, stored half-open."""
        return cls(
            datetime(first_year, 1, 1, tzinfo=timezone.utc),
            datetime(last_year + 1, 1, 1, tzinfo=timezone.utc),
        )

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        if self.end is None:
            return f"TemporalInterval(point={self.start.isoformat()})"
        return (
            f"TemporalInterval([{self.start.isoformat()}, {self.end.isoformat()}), "
            f"{self.duration_days:.4g}d)"
        )


# ============================================================================
# Position <-> moment conversion
# ============================================================================


def moment_from_year_position(position: float) -> datetime:
    """
    Inverse of :func:`temporal_config.calendar_year_position`.

    Converts a continuous ``year + fraction`` coordinate back into an instant,
    respecting the actual length of the target calendar year.
    """
    year = math.floor(position)
    fraction = position - year
    start = datetime(year, 1, 1, tzinfo=timezone.utc)
    end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
    return start + (end - start) * fraction


def _moment_at_elapsed_years(
    elapsed: float, hierarchy: TemporalHierarchy = DEFAULT_HIERARCHY
) -> datetime:
    """Instant located ``elapsed`` years after the hierarchy epoch."""
    if hierarchy.year_convention == "calendar":
        return moment_from_year_position(calendar_year_position(hierarchy.epoch) + elapsed)
    from temporal_config import LINEAR_YEAR_SECONDS

    return hierarchy.epoch + timedelta(seconds=elapsed * LINEAR_YEAR_SECONDS)


# ============================================================================
# Phase (Formula 1)
# ============================================================================


def phase_of(
    moment: datetime,
    scale: ScaleSpec,
    hierarchy: TemporalHierarchy = DEFAULT_HIERARCHY,
) -> float:
    """
    Map an instant to its phase on one circle.

    Formula 1, with the modulo applied to the period ratio **before** multiplication
    by 2π::

        phi = 2 * pi * fmod_floored( (t - t0) / T , 1.0 )

    The order matters: reducing after scaling would be a modulo against 2π of an
    already-scaled value, which is equivalent here only because the modulus is
    scaled identically. Applying it first keeps the operation dimensionless and the
    non-negativity guarantee explicit.

    Returns a phase in ``[0, 2π)``. Exactly 0 at a period boundary.
    """
    elapsed_years = years_since_epoch(moment, hierarchy.epoch, hierarchy.year_convention)
    ratio = elapsed_years / scale.period_years
    return TAU * floored_mod(ratio, 1.0)


# ============================================================================
# Circular arc algebra
# ============================================================================


def angular_difference(phi1: float, phi2: float) -> float:
    """
    Shortest angular distance between two phases, accounting for wraparound::

        delta_phi = min( |phi_q - phi_d| , 2*pi - |phi_q - phi_d| )

    Returns a value in ``[0, π]``.
    """
    diff = abs(phi1 - phi2) % TAU
    return min(diff, TAU - diff)


def _linear_spans(start: float, length: float) -> List[Tuple[float, float]]:
    """
    Decompose a circular arc into one or two non-wrapping spans on ``[0, 2π]``.

    An arc that crosses the zero-degree boundary becomes two spans; a full circle
    becomes one span covering everything.
    """
    if length >= TAU - EPS:
        return [(0.0, TAU)]
    if length <= 0.0:
        return []
    start = floored_mod(start, TAU)
    end = start + length
    if end <= TAU:
        return [(start, end)]
    return [(start, TAU), (0.0, end - TAU)]


def arc_overlap(
    start_a: float, length_a: float, start_b: float, length_b: float
) -> float:
    """
    Length, in radians, of the intersection of two arcs on the unit circle.

    Arcs are given as ``(start_phase, arc_length)``. Wraparound in either arc is
    handled. Zero-length arcs (points) never contribute overlap length — use
    :func:`arc_contains_point` for point-in-arc questions.
    """
    total = 0.0
    for s1, e1 in _linear_spans(start_a, length_a):
        for s2, e2 in _linear_spans(start_b, length_b):
            total += max(0.0, min(e1, e2) - max(s1, s2))
    return total


def arc_overlap3(
    start_a: float,
    length_a: float,
    start_b: float,
    length_b: float,
    start_c: float,
    length_c: float,
) -> float:
    """
    Length of the common intersection of three arcs.

    Used to restrict a query/document overlap to a single segment window, so that
    each segment can be scored independently.
    """
    total = 0.0
    for s1, e1 in _linear_spans(start_a, length_a):
        for s2, e2 in _linear_spans(start_b, length_b):
            lo, hi = max(s1, s2), min(e1, e2)
            if hi <= lo:
                continue
            for s3, e3 in _linear_spans(start_c, length_c):
                total += max(0.0, min(hi, e3) - max(lo, s3))
    return total


def arc_contains_point(start: float, length: float, phi: float) -> bool:
    """True if phase ``phi`` lies within the half-open arc ``[start, start+length)``."""
    if length >= TAU - EPS:
        return True
    phi = floored_mod(phi, TAU)
    for s, e in _linear_spans(start, length):
        if s - EPS <= phi < e + EPS:
            return True
    return False


def jaccard_arcs(
    start_a: float, length_a: float, start_b: float, length_b: float
) -> float:
    """
    Jaccard overlap coefficient of two arcs: ``|A ∩ B| / |A ∪ B|``.

    A quarter inside a full year scores 0.25; two identical arcs score 1.0; disjoint
    arcs score 0.0. Two points score 1.0 if coincident, else 0.0.
    """
    length_a = min(max(length_a, 0.0), TAU)
    length_b = min(max(length_b, 0.0), TAU)

    if length_a <= EPS and length_b <= EPS:
        return 1.0 if angular_difference(start_a, start_b) <= EPS else 0.0
    if length_a <= EPS:
        return 1.0 if arc_contains_point(start_b, length_b, start_a) else 0.0
    if length_b <= EPS:
        return 1.0 if arc_contains_point(start_a, length_a, start_b) else 0.0

    intersection = arc_overlap(start_a, length_a, start_b, length_b)
    union = length_a + length_b - intersection
    return 0.0 if union <= 0 else intersection / union


# ============================================================================
# Segments
# ============================================================================


def segment_bounds(scale: ScaleSpec, index: int) -> Tuple[float, float]:
    """Angular ``(start, length)`` of segment ``index`` on ``scale``."""
    return scale.segment_span(index)


def segment_of_phase(scale: ScaleSpec, phi: float) -> int:
    """
    Index of the segment containing phase ``phi``, on the even circle.

    A multiply and a truncation — no divider table, no search, no branch. Segments
    are half-open like every other interval here: a phase landing exactly on a
    divider belongs to the segment that divider opens.

    This is the *geometric* index. It answers "which quarter of the circle", which
    is not quite "which calendar quarter", because the calendar is uneven: 1 April
    sits at 88.8° in a common year, not 90°. Callers that need calendar identity use
    :func:`calendar_segments_touched`, which is exact because it works from the dates
    rather than from a phase. See the module docstring on why the geometry is left
    even.
    """
    index = int(floored_mod(phi, TAU) / TAU * scale.segments)
    # Guard the endpoint: a phase a hair under 2π can round up to `segments`.
    return min(index, scale.segments - 1)


def segments_intersected(
    scale: ScaleSpec, start: float, length: float
) -> Tuple[int, ...]:
    """
    Every segment index an arc touches on the even circle, in traversal order.

    A point touches exactly one segment. A full-circle arc touches all of them. An
    arc straddling a divider touches both adjacent segments, which is the case
    Figure 10 addresses.

    Purely geometric, like :func:`segment_of_phase`. The encoder does not use this —
    it resolves segments from the calendar — but comparison and diagnostics do, where
    only phases are in hand.
    """
    if length >= TAU - EPS:
        return tuple(range(scale.segments))

    first = segment_of_phase(scale, start)
    if length <= EPS:
        return (first,)

    width = TAU / scale.segments
    phi_start = floored_mod(start, TAU)
    touched = [first]
    index = first
    while len(touched) < scale.segments:
        index = (index + 1) % scale.segments
        # How far past the arc's start this divider sits, going forwards.
        delta = floored_mod(index * width - phi_start, TAU)
        # The arc is half-open, so a divider exactly at its end is not crossed.
        if delta >= length - EPS:
            break
        touched.append(index)
    return tuple(touched)


# ---------------------------------------------------------------------------
# Calendar segment identity — the post-geometry correction
# ---------------------------------------------------------------------------


def calendar_segment(
    moment: datetime, scale: ScaleSpec, hierarchy: TemporalHierarchy = DEFAULT_HIERARCHY
) -> int:
    """
    Which calendar segment of ``scale`` contains ``moment``.

    Integer calendar arithmetic, not geometry. This is where the unevenness of the
    calendar is allowed in, and it is exact by construction: there is no divider to
    approximate, so quarters of 90/91/92 days and the leap day need no special case
    and no tolerance.

    Two shapes, decided by whether a segment is a whole number of years:

    * **whole-year segments** (the 16-year circle's years, the 256-year circle's
      16-year blocks) — the offset from the epoch year, floor-divided by the segment
      length, modulo the segment count.
    * **sub-year segments** of the 1-year circle — a whole number of months, so the
      month index divided by the months per segment. Quarters are the case that
      matters; halves and calendar months fall out of the same expression.

    A sub-year segmentation that is not a whole number of months has no calendar
    meaning to recover, so it falls back to the even geometry.
    """
    segment_years = scale.segment_years

    if segment_years >= 1.0 and float(segment_years).is_integer():
        epoch_year = hierarchy.epoch.year
        offset = moment.year - epoch_year
        return (offset // int(segment_years)) % scale.segments

    if scale.period_years == 1 and 12 % scale.segments == 0:
        return (moment.month - 1) // (12 // scale.segments)

    return segment_of_phase(scale, phase_of(moment, scale, hierarchy))


def calendar_segments_touched(
    interval: "TemporalInterval",
    scale: ScaleSpec,
    hierarchy: TemporalHierarchy = DEFAULT_HIERARCHY,
) -> Tuple[int, ...]:
    """
    Every calendar segment of ``scale`` that ``interval`` touches, in order.

    Intervals are half-open, so the last moment counted is the one just before
    ``end``; an interval ending exactly on a divider does not touch the segment that
    divider opens. An interval at least a full period long touches everything.
    """
    if interval.is_point:
        return (calendar_segment(interval.start, scale, hierarchy),)

    if interval.years_span(hierarchy) >= scale.period_years - 1e-12:
        return tuple(range(scale.segments))

    first = calendar_segment(interval.start, scale, hierarchy)
    last = calendar_segment(interval.end - _LAST_INSTANT, scale, hierarchy)

    touched = [first]
    index = first
    while index != last and len(touched) < scale.segments:
        index = (index + 1) % scale.segments
        touched.append(index)
    return tuple(touched)


#: Steps back from a half-open end to the final moment actually inside the interval.
_LAST_INSTANT = timedelta(microseconds=1)


def segment_label(
    scale: ScaleSpec, index: int, hierarchy: TemporalHierarchy = DEFAULT_HIERARCHY
) -> str:
    """
    Human-readable name for a segment, e.g. ``"Q2"`` or ``"year 2023"``.

    Segment identity is relative to the epoch: on the 16-year circle with a 1900
    epoch, segment 0 covers 1900, 1916, 1932, ... The label names the segment's role,
    not a unique calendar block — disambiguating between repeats is the job of the
    next coarser circle.
    """
    if scale.segments == 4 and scale.period_years == 1:
        return f"Q{index + 1}"
    epoch_year = math.floor(calendar_year_position(hierarchy.epoch))
    span = scale.segment_years
    if span == 1:
        return f"{scale.segment_label} +{index}y (e.g. {epoch_year + index})"
    return f"{scale.segment_label} +{index * span:g}y"


# ============================================================================
# Tuples and encodings
# ============================================================================


@dataclass(frozen=True)
class ScaleTuple:
    """
    The ``[cos, sin, z]`` triple for one scale, plus the derived quantities that
    retrieval needs but that are recoverable from the triple itself.

    Attributes:
        scale_name: Name of the scale this tuple belongs to.
        period_years: Period of that scale, carried so the tuple is self-describing.
        cos, sin: Components of the arc's centre phase.
        z: Arc length in radians. 0 for a point, capped at 2π.
        phi_center: Centre phase in radians, in ``[0, 2π)``.
        phi_start: Phase where the arc begins, in ``[0, 2π)``. Equals
            ``phi_center`` for a point.
        segments: Segment indices the arc touches.
    """

    scale_name: str
    period_years: float
    cos: float
    sin: float
    z: float
    phi_center: float
    phi_start: float
    segments: Tuple[int, ...]

    @property
    def is_point(self) -> bool:
        return self.z <= EPS

    @property
    def is_full_circle(self) -> bool:
        """True if the arc saturates this circle, making an overlap check vacuous."""
        return self.z >= TAU - EPS

    @property
    def phi_end(self) -> float:
        """Unwrapped end phase: always ``>= phi_start``, may exceed 2π."""
        return self.phi_start + self.z

    def as_triple(self) -> List[float]:
        return [self.cos, self.sin, self.z]

    def to_dict(self) -> Dict[str, object]:
        return {
            "scale": self.scale_name,
            "period_years": self.period_years,
            "cos": self.cos,
            "sin": self.sin,
            "z": self.z,
            "phi_center": self.phi_center,
            "phi_start": self.phi_start,
            "segments": list(self.segments),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "ScaleTuple":
        return cls(
            scale_name=str(data["scale"]),
            period_years=float(data["period_years"]),
            cos=float(data["cos"]),
            sin=float(data["sin"]),
            z=float(data["z"]),
            phi_center=float(data["phi_center"]),
            phi_start=float(data["phi_start"]),
            segments=tuple(int(s) for s in data.get("segments", ())),  # type: ignore[union-attr]
        )


def neutral_tuple(scale: ScaleSpec) -> ScaleTuple:
    """
    Placeholder tuple for a scale a vector never encoded.

    Represented as a full-circle arc at phase 0. A full circle overlaps everything,
    so a vector padded with neutral tuples is neither rejected nor artificially
    boosted on scales it predates — the scale simply carries no information, which
    is the honest reading of "this vector was written before that circle existed".
    """
    return ScaleTuple(
        scale_name=scale.name,
        period_years=scale.period_years,
        cos=1.0,
        sin=0.0,
        z=TAU,
        phi_center=0.0,
        phi_start=0.0,
        segments=tuple(range(scale.segments)),
    )


@dataclass(frozen=True)
class TemporalEncoding:
    """
    One representation of one interval: a self-describing, variable-length temporal
    vector.

    An interval that crosses a period boundary yields several of these, all sharing
    a ``group_id`` and differing only in the tuples for the scales whose boundary was
    crossed.

    Attributes:
        tuples: One :class:`ScaleTuple` per scale, finest first.
        hierarchy: The hierarchy that produced this encoding.
        interval: The component interval this representation covers. For a split
            interval this is the component, not the original span.
        source_interval: The full, unsplit interval the document actually refers to.
        group_id: Shared across every representation emitted from one source
            interval. Deduplication key at the application layer.
        representation_index: 0-based position within the emitted group.
        representation_count: Total representations emitted for the group.
    """

    tuples: Tuple[ScaleTuple, ...]
    hierarchy: TemporalHierarchy
    interval: TemporalInterval
    source_interval: TemporalInterval
    group_id: str
    representation_index: int = 0
    representation_count: int = 1

    # -- access ------------------------------------------------------------

    def tuple_for(self, scale_name: str) -> ScaleTuple:
        for t in self.tuples:
            if t.scale_name == scale_name:
                return t
        raise KeyError(f"no tuple for scale {scale_name!r}")

    @property
    def is_point(self) -> bool:
        return self.interval.is_point

    @property
    def is_split(self) -> bool:
        return self.representation_count > 1

    @property
    def scale_names(self) -> Tuple[str, ...]:
        return tuple(t.scale_name for t in self.tuples)

    # -- vector form -------------------------------------------------------

    def to_vector(self) -> List[float]:
        """Flat ``[cos, sin, z]`` per scale, finest first."""
        out: List[float] = []
        for t in self.tuples:
            out.extend(t.as_triple())
        return out

    @property
    def dimensions(self) -> int:
        return 3 * len(self.tuples)

    # -- serialisation -----------------------------------------------------

    def header(self) -> Dict[str, object]:
        """
        Prepended descriptor declaring how many tuples follow and what each means.

        This is what makes the vector length variable without being ambiguous: a
        reader never has to assume nine dimensions, it reads the tuple count and the
        period of each tuple from the header.
        """
        head = dict(self.hierarchy.header())
        head["group_id"] = self.group_id
        head["representation_index"] = self.representation_index
        head["representation_count"] = self.representation_count
        return head

    def to_dict(self) -> Dict[str, object]:
        return {
            "header": self.header(),
            "tuples": [t.to_dict() for t in self.tuples],
            "interval_start": self.interval.start.isoformat(),
            "interval_end": self.interval.end.isoformat() if self.interval.end else None,
            "source_start": self.source_interval.start.isoformat(),
            "source_end": (
                self.source_interval.end.isoformat() if self.source_interval.end else None
            ),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "TemporalEncoding":
        header = dict(data["header"])  # type: ignore[arg-type]
        group_id = str(header.pop("group_id", ""))
        rep_index = int(header.pop("representation_index", 0))
        rep_count = int(header.pop("representation_count", 1))
        hierarchy = TemporalHierarchy.from_header(header)

        def _interval(start_key: str, end_key: str) -> TemporalInterval:
            start = datetime.fromisoformat(str(data[start_key]))
            raw_end = data.get(end_key)
            end = datetime.fromisoformat(str(raw_end)) if raw_end else None
            return TemporalInterval(start, end)

        return cls(
            tuples=tuple(ScaleTuple.from_dict(t) for t in data["tuples"]),  # type: ignore[union-attr]
            hierarchy=hierarchy,
            interval=_interval("interval_start", "interval_end"),
            source_interval=_interval("source_start", "source_end"),
            group_id=group_id,
            representation_index=rep_index,
            representation_count=rep_count,
        )

    def describe(self) -> str:  # pragma: no cover - diagnostics only
        lines = [
            f"TemporalEncoding {self.group_id}"
            f" [{self.representation_index + 1}/{self.representation_count}]",
            f"  interval : {self.interval}",
            f"  epoch    : {self.hierarchy.epoch.date()} ({self.hierarchy.year_convention})",
        ]
        for t in self.tuples:
            mode = "point" if t.is_point else f"arc {math.degrees(t.z):7.2f}deg"
            segs = ",".join(str(s) for s in t.segments)
            lines.append(
                f"  {t.scale_name:<11} T={t.period_years:>6g}y  "
                f"phi={math.degrees(t.phi_center):7.2f}deg  {mode}  "
                f"[cos {t.cos:+.4f}, sin {t.sin:+.4f}, z {t.z:.4f}]  seg[{segs}]"
            )
        return "\n".join(lines)


# ============================================================================
# Encoding
# ============================================================================


def encode_tuple(
    interval: TemporalInterval,
    scale: ScaleSpec,
    hierarchy: TemporalHierarchy = DEFAULT_HIERARCHY,
) -> ScaleTuple:
    """Encode one interval onto one circle."""
    phi_start = phase_of(interval.start, scale, hierarchy)

    if interval.is_point:
        z = 0.0
        phi_center = phi_start
    else:
        span_years = interval.years_span(hierarchy)
        # Cap at one full revolution: a duration longer than the period saturates
        # the circle instead of wrapping around it repeatedly.
        z = min(TAU * span_years / scale.period_years, TAU)
        phi_center = floored_mod(phi_start + z / 2.0, TAU)

    return ScaleTuple(
        scale_name=scale.name,
        period_years=scale.period_years,
        cos=math.cos(phi_center),
        sin=math.sin(phi_center),
        z=z,
        phi_center=phi_center,
        phi_start=phi_start,
        # The geometry above is even and calendar-free. Segment identity is the one
        # thing that has to know about real quarters and leap days, so it is resolved
        # here — the last point at which the actual dates are still available — and
        # carried on the tuple from then on. Nothing downstream re-derives it.
        segments=calendar_segments_touched(interval, scale, hierarchy),
    )


def boundary_moments(
    interval: TemporalInterval,
    scale: ScaleSpec,
    hierarchy: TemporalHierarchy = DEFAULT_HIERARCHY,
) -> List[datetime]:
    """
    Instants strictly inside ``interval`` at which ``scale``'s circle wraps through
    zero degrees.

    With a January-1 epoch and a 1-year scale these are the New Year boundaries; with
    a mid-year epoch they are the anniversaries of the epoch.
    """
    if interval.is_point:
        return []
    period = scale.period_years
    u_start = years_since_epoch(interval.start, hierarchy.epoch, hierarchy.year_convention) / period
    u_end = years_since_epoch(interval.end, hierarchy.epoch, hierarchy.year_convention) / period

    first = math.floor(u_start + EPS) + 1
    last = math.ceil(u_end - EPS) - 1
    return [
        _moment_at_elapsed_years(k * period, hierarchy)
        for k in range(first, last + 1)
    ]


def split_at_boundaries(
    interval: TemporalInterval,
    hierarchy: TemporalHierarchy = DEFAULT_HIERARCHY,
    max_representations: int = MAX_REPRESENTATIONS,
) -> List[TemporalInterval]:
    """
    Split an interval into component arcs at every zero-degree crossing.

    Splitting is performed at the **finest** scale's boundaries only. Because each
    period is an exact factor of the next, a coarser circle's boundaries are a subset
    of the finest circle's, so one pass at the finest scale catches every crossing in
    the hierarchy. Components that do not cross a given coarser boundary produce
    identical tuples at that coarser scale — which is exactly the behaviour the
    specification describes for the 16-year and 256-year tuples of a multi-year
    document.

    A document spanning 2017 through 2022 crosses five 1-year boundaries and
    therefore yields six components: two partial-year arcs at the ends and four
    full-year arcs in between.

    If splitting would exceed ``max_representations``, the interval is left whole and
    its arcs saturate at 2π. That keeps a pathological input (a 500-year corpus-wide
    document) from exploding the index; the trade-off is that such a document matches
    broadly at the finest scale, which is the correct answer for something that
    genuinely spans everything.
    """
    if interval.is_point:
        return [interval]

    cuts = boundary_moments(interval, hierarchy.finest, hierarchy)
    if not cuts:
        return [interval]
    if len(cuts) + 1 > max_representations:
        return [interval]

    components: List[TemporalInterval] = []
    cursor = interval.start
    for cut in cuts:
        if cut > cursor:
            components.append(TemporalInterval(cursor, cut))
        cursor = cut
    if interval.end > cursor:
        components.append(TemporalInterval(cursor, interval.end))
    return components or [interval]


def encode(
    interval: TemporalInterval,
    hierarchy: TemporalHierarchy = DEFAULT_HIERARCHY,
    group_id: Optional[str] = None,
    split: bool = True,
    max_representations: int = MAX_REPRESENTATIONS,
) -> List[TemporalEncoding]:
    """
    Encode an interval into one or more temporal vectors.

    Args:
        interval: The span (or instant) to encode.
        hierarchy: Scales and epoch to encode against.
        group_id: Shared deduplication identifier. Generated if omitted.
        split: Split at zero-degree boundaries and emit one representation per
            component arc. Set False to force a single representation whose arcs
            wrap (and saturate) instead.
        max_representations: Ceiling on the emitted representation count.

    Returns:
        A list of :class:`TemporalEncoding`, always non-empty, each carrying the same
        ``group_id``.
    """
    gid = group_id or str(uuid.uuid4())
    components = (
        split_at_boundaries(interval, hierarchy, max_representations)
        if split
        else [interval]
    )
    count = len(components)
    return [
        TemporalEncoding(
            tuples=tuple(encode_tuple(component, scale, hierarchy) for scale in hierarchy.scales),
            hierarchy=hierarchy,
            interval=component,
            source_interval=interval,
            group_id=gid,
            representation_index=index,
            representation_count=count,
        )
        for index, component in enumerate(components)
    ]


def full_span_interval(
    hierarchy: TemporalHierarchy = DEFAULT_HIERARCHY,
) -> TemporalInterval:
    """
    The interval covering one full revolution of the outermost circle.

    Encoded, every one of its arcs saturates its circle, so it overlaps every
    document in the corpus. That is the honest encoding of "no temporal
    constraint": the gate cannot reject anything and ranking falls back to
    semantic similarity alone.
    """
    return TemporalInterval(
        hierarchy.epoch,
        _moment_at_elapsed_years(hierarchy.coarsest.period_years, hierarchy),
    )


def encode_single(
    interval: TemporalInterval,
    hierarchy: TemporalHierarchy = DEFAULT_HIERARCHY,
    group_id: Optional[str] = None,
) -> TemporalEncoding:
    """
    Encode without splitting, returning exactly one representation.

    Used for queries: a query arc is evaluated against stored arcs directly and does
    not need to be materialised as multiple index rows.
    """
    return encode(interval, hierarchy, group_id=group_id, split=False)[0]


def pad_to_hierarchy(
    encoding: TemporalEncoding, hierarchy: TemporalHierarchy
) -> TemporalEncoding:
    """
    Widen an encoding to a longer hierarchy by appending neutral tuples.

    This is the compatibility path for a corpus indexed before an outer scale was
    added: the older vectors gain a full-circle tuple for the new scale, which is
    vacuous in overlap checks, so they continue to retrieve correctly alongside newly
    encoded vectors.

    Raises if the target hierarchy is not an extension of the encoding's own — a
    changed epoch, year convention or inner period is a re-index, not a pad.
    """
    if not encoding.hierarchy.is_compatible_with(hierarchy):
        raise ValueError(
            f"cannot pad across incompatible hierarchies: "
            f"{encoding.hierarchy.fingerprint()} -> {hierarchy.fingerprint()}"
        )
    present = set(encoding.scale_names)
    padded = list(encoding.tuples) + [
        neutral_tuple(scale) for scale in hierarchy.scales if scale.name not in present
    ]
    return TemporalEncoding(
        tuples=tuple(padded),
        hierarchy=hierarchy,
        interval=encoding.interval,
        source_interval=encoding.source_interval,
        group_id=encoding.group_id,
        representation_index=encoding.representation_index,
        representation_count=encoding.representation_count,
    )


# ============================================================================
# Lazy traversal
# ============================================================================


@dataclass(frozen=True)
class TraversalPlan:
    """
    The scales a given query actually needs to check, coarsest first.

    Encoding precision and query traversal are deliberately separate concerns. Every
    scale in the hierarchy is written at ingestion, so nothing is discarded; but a
    query only descends as deeply as it needs, and the cost of retrieval tracks the
    precision the query asked for rather than the precision the corpus holds.

    A scale is skipped when the query arc saturates that circle, because a
    full-circle arc overlaps everything and the check cannot reject anything. Asking
    for "all of 2021" spans the entire 1-year circle, so the 1-year check is dropped
    and the year is resolved on the 16-year circle instead.

    Attributes:
        scale_names: Scales to evaluate, ordered coarsest to finest so the widest,
            cheapest rejections happen first.
        skipped: Scales deliberately not evaluated, with the reason.
    """

    scale_names: Tuple[str, ...]
    skipped: Tuple[Tuple[str, str], ...] = ()

    @property
    def depth(self) -> int:
        return len(self.scale_names)

    def __contains__(self, name: object) -> bool:
        return name in self.scale_names

    def __iter__(self) -> Iterator[str]:
        return iter(self.scale_names)


def traversal_plan(
    query: TemporalEncoding, hierarchy: Optional[TemporalHierarchy] = None
) -> TraversalPlan:
    """Build the lazy-traversal plan for a query encoding."""
    hierarchy = hierarchy or query.hierarchy
    needed: List[str] = []
    skipped: List[Tuple[str, str]] = []

    for scale in reversed(hierarchy.scales):  # coarsest -> finest
        try:
            t = query.tuple_for(scale.name)
        except KeyError:
            skipped.append((scale.name, "absent from query encoding"))
            continue
        if t.is_full_circle:
            skipped.append(
                (scale.name, f"query arc saturates the {scale.period_years:g}y circle")
            )
            continue
        needed.append(scale.name)

    if not needed:
        # Everything was vacuous (a query spanning the whole corpus). Keep the
        # outermost circle so scoring still has one axis to work with.
        coarsest = hierarchy.coarsest.name
        needed = [coarsest]
        skipped = tuple(s for s in skipped if s[0] != coarsest)  # type: ignore[assignment]

    return TraversalPlan(scale_names=tuple(needed), skipped=tuple(skipped))


# ============================================================================
# Overlap evaluation
# ============================================================================


@dataclass
class ScaleMatch:
    """Outcome of comparing a query tuple and a document tuple at one scale."""

    scale_name: str
    overlaps: bool
    jaccard: float
    delta_phi: float
    shared_segments: Tuple[int, ...] = ()
    segment_jaccard: Dict[int, float] = field(default_factory=dict)


def evaluate_scale(
    query_tuple: ScaleTuple,
    doc_tuple: ScaleTuple,
    scale: ScaleSpec,
) -> ScaleMatch:
    """
    Compare a query arc and a document arc on one circle.

    Returns both the soft signal (Jaccard coefficient, angular distance) and the hard
    signal (do they overlap at all), plus a per-segment breakdown.

    The per-segment breakdown is what handles a query window straddling a hard
    calendar divider — say a search running from within 2022 into 2023 on the
    16-year circle. Each intersected segment is scored independently and a match
    against *any* of them qualifies the document, so a genuinely relevant 2023
    document is not lost simply because the query's centre of mass sat in 2022.
    Without it, temporal bleeding is suppressed at the cost of dropping the
    documents on the far side of the divider.
    """
    q_start, q_len = query_tuple.phi_start, query_tuple.z
    d_start, d_len = doc_tuple.phi_start, doc_tuple.z

    if query_tuple.is_point and doc_tuple.is_point:
        overlaps = angular_difference(q_start, d_start) <= EPS
    elif query_tuple.is_point:
        overlaps = arc_contains_point(d_start, d_len, q_start)
    elif doc_tuple.is_point:
        overlaps = arc_contains_point(q_start, q_len, d_start)
    else:
        overlaps = arc_overlap(q_start, q_len, d_start, d_len) > EPS

    jaccard = jaccard_arcs(q_start, q_len, d_start, d_len)
    delta = angular_difference(query_tuple.phi_center, doc_tuple.phi_center)

    shared = tuple(sorted(set(query_tuple.segments) & set(doc_tuple.segments)))
    per_segment: Dict[int, float] = {}
    for index in shared:
        seg_start, seg_len = segment_bounds(scale, index)
        if query_tuple.is_point or doc_tuple.is_point:
            # One side has no extent inside the segment, so a length-based Jaccard
            # would be identically zero. Both arcs are known to touch this segment;
            # score it on whether they actually meet.
            per_segment[index] = 1.0 if overlaps else 0.0
            continue
        # Jaccard restricted to the segment window.
        q_in = arc_overlap(q_start, q_len, seg_start, seg_len)
        d_in = arc_overlap(d_start, d_len, seg_start, seg_len)
        both = arc_overlap3(q_start, q_len, d_start, d_len, seg_start, seg_len)
        union = q_in + d_in - both
        per_segment[index] = both / union if union > EPS else 0.0

    if per_segment:
        # A match on any intersecting segment qualifies the document.
        jaccard = max(jaccard, max(per_segment.values()))
        overlaps = overlaps or any(v > 0 for v in per_segment.values())

    return ScaleMatch(
        scale_name=scale.name,
        overlaps=overlaps,
        jaccard=jaccard,
        delta_phi=delta,
        shared_segments=shared,
        segment_jaccard=per_segment,
    )


def temporal_alignment(delta_phi: float, beta: float) -> float:
    """
    Gaussian alignment kernel::

        alignment = exp( -beta * (delta_phi)^2 )

    ``delta_phi`` is the circular angular distance from :func:`angular_difference`
    (already wraparound-aware, in ``[0, π]``), and it is the *difference* that is
    squared, not the phase. ``beta`` is the temporal-focus parameter: 0 flattens the
    kernel to 1 everywhere (pure semantic search), larger values sharpen it.
    """
    return math.exp(-beta * (delta_phi ** 2))
