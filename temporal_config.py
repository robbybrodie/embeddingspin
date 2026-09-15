"""
Temporal Hierarchy Configuration
================================

Declares the *shape* of the temporal encoding: the base epoch, the ordered set of
periodic scales, how each scale is divided into discrete segments, and the per-scale
weights used during retrieval.

This module is the single source of truth for everything that must stay identical
between ingestion and retrieval. Encoding a corpus under one hierarchy and querying
it under another produces meaningless overlap results, so every encoded vector
carries a header describing the hierarchy that produced it (see
``TemporalHierarchy.header``).

Design points
-------------

Base epoch (t0)
    Configurable. The default is 1900-01-01 UTC: it predates practically all
    financial, legal and medical records while remaining inside the Gregorian
    calendar (no proleptic-calendar complications), and combined with the 256-year
    outer scale it provides continuous coverage through the year 2155.

    Changing the epoch changes phases. It is *not* a free operation: phases are only
    preserved if the epoch shift is an exact multiple of every period in the
    hierarchy. Use :func:`epoch_shift_is_congruent` to check before migrating, and
    expect a re-index when it returns False.

Variable-length hierarchy
    The hierarchy is an ordered tuple of :class:`ScaleSpec` from finest to coarsest.
    The default is 1 / 16 / 256 years (integer powers of two, each a factor of the
    next), producing a nine-dimensional temporal vector of ``[cos, sin, z]`` per
    scale. Additional coarser scales (e.g. 4096 years) can be appended with
    :meth:`TemporalHierarchy.extended` without disturbing the existing tuples.

Segments
    Each circle is additionally divided into discrete segments: the 1-year circle
    into 4 calendar quarters, the 16-year circle into 16 calendar years, the
    256-year circle into 16 sixteen-year blocks. Segments are what make
    "does this document fall in 2022 or 2023?" a hard, checkable question rather
    than a soft angular distance.

Year convention
    ``calendar`` (default) measures position as a fraction of the *actual* calendar
    year, so 1 January is always phase 0 on the 1-year circle regardless of leap
    years. ``linear`` uses a fixed year length in seconds and is retained only for
    reading corpora encoded by earlier versions of this package.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Dict, Iterable, Optional, Sequence, Tuple

# ============================================================================
# Schema versioning
# ============================================================================

# Bumped whenever the on-disk layout of a temporal vector or its header changes.
#   1 = fixed 9D vector, epoch 2010-01-01, no segments, no boundary splitting
#   2 = variable-length tuples + header, configurable epoch, segments,
#       boundary splitting with shared group_id
SCHEMA_VERSION = 2

TAU = math.tau

# ============================================================================
# Base epochs
# ============================================================================

#: Patent-preferred epoch. Predates practically all relevant records, stays inside
#: the Gregorian calendar, and with the 256-year scale covers through 2155.
EPOCH_1900 = datetime(1900, 1, 1, tzinfo=timezone.utc)

#: Epoch used by schema version 1 of this package. Kept so that v1 corpora can still
#: be read and re-encoded; do not use it for new corpora.
EPOCH_2010_LEGACY = datetime(2010, 1, 1, tzinfo=timezone.utc)

#: Simplified epoch used by the worked example in the specification.
EPOCH_PATENT_EXAMPLE = EPOCH_2010_LEGACY


def _parse_epoch(raw: str) -> datetime:
    dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def default_epoch() -> datetime:
    """Resolve the default epoch, honouring ``EMBEDDINGSPIN_EPOCH`` if set."""
    raw = os.getenv("EMBEDDINGSPIN_EPOCH")
    return _parse_epoch(raw) if raw else EPOCH_1900


# ============================================================================
# Year conventions
# ============================================================================

#: Mean Gregorian year in days.
DAYS_PER_YEAR = 365.2425

#: Mean Gregorian year, used only by the ``linear`` convention.
LINEAR_YEAR_SECONDS = DAYS_PER_YEAR * 24 * 3600

YEAR_CONVENTIONS = ("calendar", "linear")


def default_year_convention() -> str:
    value = os.getenv("EMBEDDINGSPIN_YEAR_CONVENTION", "calendar").lower()
    if value not in YEAR_CONVENTIONS:
        raise ValueError(
            f"EMBEDDINGSPIN_YEAR_CONVENTION must be one of {YEAR_CONVENTIONS}, got {value!r}"
        )
    return value


def _jan1(year: int) -> datetime:
    return datetime(year, 1, 1, tzinfo=timezone.utc)


def calendar_year_position(moment: datetime) -> float:
    """
    Absolute position of ``moment`` on a continuous calendar-year axis.

    Returns ``year + fraction``, where ``fraction`` is the elapsed portion of that
    specific calendar year measured against its own actual length. A 366-day year is
    divided by 366, a 365-day year by 365, so 1 January is always ``year + 0.0`` and
    31 December 23:59 is always just under ``year + 1.0``. This is what keeps the
    1-year circle free of leap-year drift without any special-case logic at
    retrieval time.
    """
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    start = _jan1(moment.year)
    end = _jan1(moment.year + 1)
    return moment.year + (moment - start).total_seconds() / (end - start).total_seconds()


def years_since_epoch(moment: datetime, epoch: datetime, convention: str = "calendar") -> float:
    """
    Elapsed time between ``epoch`` and ``moment``, expressed in years.

    This is the ``(t - t0)`` term of Formula 1, already divided by the year unit.
    May be negative for moments preceding the epoch; the floored modulo in
    :func:`floored_mod` maps those onto the circle correctly.
    """
    if convention == "calendar":
        return calendar_year_position(moment) - calendar_year_position(epoch)
    if convention == "linear":
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        return (moment - epoch).total_seconds() / LINEAR_YEAR_SECONDS
    raise ValueError(f"Unknown year convention {convention!r}")


def floored_mod(value: float, modulus: float = 1.0) -> float:
    """
    Floored modulo: ``value - modulus * floor(value / modulus)``.

    The remainder is **non-negative and strictly less than the modulus**
    (``0 <= r < modulus``) for any input sign. It is zero — not the modulus — at an
    exact period boundary, which is why the guarantee is "non-negative" rather than
    "positive": 1 January of an epoch-aligned year sits at phase 0, not at 2π.

    Python's ``%`` operator already implements floored (not truncated) modulo for
    floats; this function exists to name the contract and to defend the boundary
    case against floating-point results that land fractionally below zero.
    """
    remainder = value - modulus * math.floor(value / modulus)
    # Guard against `value - modulus*floor(...)` producing -0.0 or a value that has
    # rounded up to exactly `modulus` for very small negative inputs.
    if remainder < 0.0 or remainder >= modulus:
        remainder = 0.0
    return remainder


# ============================================================================
# Scales
# ============================================================================


@dataclass(frozen=True)
class ScaleSpec:
    """
    One periodic scale (one "circle") in the hierarchy.

    **The circle is divided evenly, always.** Periods are integer powers of two and
    segment counts are too, so a segment index is ``int(fraction * segments)`` — a
    multiply and a truncation on a value Formula 1 has already reduced to ``[0, 1)``.
    No divider table, no search, no data-dependent branch, and the same arithmetic
    vectorises across a batch of candidates. That is the whole reason the periods are
    1 / 16 / 256 rather than 1 / 10 / 100.

    The calendar is *not* even — quarters are 90, 91, 92 and 92 days, and a leap day
    shifts every divider after February. None of that is allowed into the geometry.
    Segment *identity* is resolved from the calendar once, at encoding time, where
    the real dates are still in hand, and stored; see
    :func:`temporal_encoding.calendar_segments_touched`. Even geometry first,
    calendar correction after.

    Attributes:
        name: Stable identifier used as a dict key and in stored metadata.
        period_years: Length of one full revolution, in years.
        segments: Number of equal parts the circle is divided into. A segment is a
            calendar block — a quarter, a year, a 16-year block — and the count is
            what relates the even circle to it.
        weight: Relative importance of this scale when combining per-scale temporal
            alignments into a single score. Weights are normalised at use, so they
            need not sum to 1.
        segment_label: Human-readable name of one segment, for diagnostics.
    """

    name: str
    period_years: float
    segments: int
    weight: float
    segment_label: str = "segment"

    def __post_init__(self) -> None:
        if self.period_years <= 0:
            raise ValueError(f"scale {self.name!r}: period_years must be positive")
        if self.segments < 1:
            raise ValueError(f"scale {self.name!r}: segments must be >= 1")
        if self.weight < 0:
            raise ValueError(f"scale {self.name!r}: weight must be non-negative")

    @property
    def boundary_phases(self) -> Tuple[float, ...]:
        """
        Angular position of each divider, in radians, ascending from 0.

        Derived, not stored: the dividers are at ``i * 2π / segments``. Provided for
        display and for tests; the hot path computes an index arithmetically instead
        of searching this.
        """
        return tuple(i * TAU / self.segments for i in range(self.segments))

    @property
    def segment_years(self) -> float:
        """Duration of one segment, in years."""
        return self.period_years / self.segments

    @property
    def segment_radians(self) -> float:
        """Angular width of one segment, in radians. Every segment has the same."""
        return TAU / self.segments

    def segment_span(self, index: int) -> Tuple[float, float]:
        """Angular ``(start, length)`` of segment ``index``, handling the wrap."""
        width = TAU / self.segments
        return (index % self.segments) * width, width


# Default hierarchy: 1 / 16 / 256 years.
#
# Each period is an integer power of two and a factor of the next, so a boundary on
# a finer circle is always also a candidate boundary on a coarser one. That is what
# lets interval splitting happen once, at the finest scale, and remain valid for
# every coarser scale. The segment counts are powers of two for the same reason the
# periods are: the index arithmetic stays exact and branch-free.
QUARTER_SCALE = ScaleSpec(
    name="quarter",
    period_years=1,
    segments=4,  # four calendar quarters
    weight=0.4,
    segment_label="quarter",
)
DECADE_SCALE = ScaleSpec(
    name="decade",
    period_years=16,
    segments=16,  # sixteen calendar years
    weight=0.5,
    segment_label="year",
)
CENTURY_SCALE = ScaleSpec(
    name="century",
    period_years=256,
    segments=16,  # sixteen 16-year blocks
    weight=0.1,
    segment_label="16-year block",
)

#: Optional fourth circle, demonstrating extension of the hierarchy past 2155.
MILLENNIUM_SCALE = ScaleSpec(
    name="millennium",
    period_years=4096,
    segments=16,  # sixteen 256-year blocks
    weight=0.0,  # contributes nothing to scoring until a query actually needs it
    segment_label="256-year block",
)

DEFAULT_SCALES: Tuple[ScaleSpec, ...] = (QUARTER_SCALE, DECADE_SCALE, CENTURY_SCALE)


# ============================================================================
# Hierarchy
# ============================================================================


@dataclass(frozen=True)
class TemporalHierarchy:
    """
    The complete, ordered set of periodic scales plus the epoch they are measured
    from. Immutable: derive variants with :meth:`extended` or
    :func:`dataclasses.replace`.

    Scales are ordered finest-first. ``hierarchy.scales[0]`` is the highest-precision
    circle; ``hierarchy.scales[-1]`` is the widest-range circle.
    """

    epoch: datetime = None  # type: ignore[assignment]  # filled in __post_init__
    scales: Tuple[ScaleSpec, ...] = DEFAULT_SCALES
    year_convention: str = None  # type: ignore[assignment]
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.epoch is None:
            object.__setattr__(self, "epoch", default_epoch())
        if self.epoch.tzinfo is None:
            object.__setattr__(self, "epoch", self.epoch.replace(tzinfo=timezone.utc))
        if self.year_convention is None:
            object.__setattr__(self, "year_convention", default_year_convention())
        if not self.scales:
            raise ValueError("hierarchy must contain at least one scale")
        periods = [s.period_years for s in self.scales]
        if periods != sorted(periods):
            raise ValueError("scales must be ordered finest (shortest period) first")
        if len(set(s.name for s in self.scales)) != len(self.scales):
            raise ValueError("scale names must be unique")

    # -- lookup ------------------------------------------------------------

    @property
    def names(self) -> Tuple[str, ...]:
        return tuple(s.name for s in self.scales)

    @property
    def periods(self) -> Tuple[float, ...]:
        return tuple(s.period_years for s in self.scales)

    @property
    def finest(self) -> ScaleSpec:
        return self.scales[0]

    @property
    def coarsest(self) -> ScaleSpec:
        return self.scales[-1]

    @property
    def dimensions(self) -> int:
        """Length of the flat temporal vector: three components per scale."""
        return 3 * len(self.scales)

    def scale(self, name: str) -> ScaleSpec:
        for spec in self.scales:
            if spec.name == name:
                return spec
        raise KeyError(f"no scale named {name!r} in hierarchy {self.names}")

    def normalized_weights(self, names: Optional[Sequence[str]] = None) -> Dict[str, float]:
        """
        Per-scale weights normalised to sum to 1 over ``names``.

        Restricting to a subset matters for lazy traversal: when only the coarse
        scales are evaluated, their weights are renormalised over just those scales
        rather than being diluted by scales that were never checked.
        """
        selected = [s for s in self.scales if names is None or s.name in names]
        if not selected:
            return {}
        total = sum(s.weight for s in selected)
        if total <= 0:
            # All-zero weights (e.g. only the millennium scale was traversed):
            # fall back to a uniform split rather than dividing by zero.
            return {s.name: 1.0 / len(selected) for s in selected}
        return {s.name: s.weight / total for s in selected}

    # -- coverage ----------------------------------------------------------

    @property
    def coverage_end_year(self) -> int:
        """
        First year that is no longer unambiguously representable.

        The outermost circle wraps after ``coarsest.period_years``, so any two
        moments separated by exactly that span collide at every scale.
        """
        return int(calendar_year_position(self.epoch) + self.coarsest.period_years)

    def covers(self, moment: datetime) -> bool:
        """True if ``moment`` falls inside one revolution of the outermost circle."""
        elapsed = years_since_epoch(moment, self.epoch, self.year_convention)
        return 0.0 <= elapsed < self.coarsest.period_years

    # -- extension ---------------------------------------------------------

    def extended(self, scale: ScaleSpec) -> "TemporalHierarchy":
        """
        Return a new hierarchy with ``scale`` appended as the outermost circle.

        Appending is additive: every existing tuple keeps its period, its epoch and
        therefore its phase, and a new tuple is appended at the end. Corpora encoded
        under the shorter hierarchy remain readable — see
        :func:`temporal_spin.pad_to_hierarchy`, which supplies a neutral
        full-circle tuple for the missing scale so that old vectors neither match
        nor block on a scale they never encoded.

        Note what this does *not* do: it does not let you change the epoch for free.
        Extension preserves phases precisely because the epoch is held fixed.
        """
        if scale.period_years <= self.coarsest.period_years:
            raise ValueError(
                f"extension scale must be coarser than {self.coarsest.name} "
                f"({self.coarsest.period_years}y), got {scale.period_years}y"
            )
        return replace(self, scales=self.scales + (scale,))

    # -- serialisation -----------------------------------------------------

    def header(self) -> Dict[str, object]:
        """
        Compact, JSON-serialisable description of this hierarchy.

        Stored alongside every temporal vector so that a reader can tell how many
        tuples are present, which period each tuple corresponds to, and which epoch
        and year convention produced the phases.
        """
        return {
            "schema_version": self.schema_version,
            "epoch": self.epoch.isoformat(),
            "year_convention": self.year_convention,
            "tuple_count": len(self.scales),
            "scales": [
                {
                    "name": s.name,
                    "period_years": s.period_years,
                    "segments": s.segments,
                    "weight": s.weight,
                }
                for s in self.scales
            ],
        }

    @classmethod
    def from_header(cls, header: Dict[str, object]) -> "TemporalHierarchy":
        """Reconstruct a hierarchy from a stored header."""
        scales = tuple(
            ScaleSpec(
                name=str(s["name"]),
                period_years=float(s["period_years"]),
                segments=int(s["segments"]),
                weight=float(s["weight"]),
            )
            for s in header["scales"]  # type: ignore[union-attr]
        )
        return cls(
            epoch=_parse_epoch(str(header["epoch"])),
            scales=scales,
            year_convention=str(header.get("year_convention", "calendar")),
            schema_version=int(header.get("schema_version", SCHEMA_VERSION)),
        )

    def fingerprint(self) -> str:
        """Short string identifying the hierarchy, for compatibility checks."""
        parts = "+".join(f"{s.name}:{s.period_years:g}:{s.segments}" for s in self.scales)
        return f"v{self.schema_version}|{self.epoch.date().isoformat()}|{self.year_convention}|{parts}"

    def is_compatible_with(self, other: "TemporalHierarchy") -> bool:
        """
        True if vectors from ``other`` can be compared against vectors from ``self``
        without re-encoding.

        Compatible means: same epoch, same year convention, and ``other``'s scales
        are a prefix of ``self``'s (or vice versa) — i.e. one is an extension of the
        other. A differing epoch is never compatible unless the shift is congruent
        with every period, which :func:`epoch_shift_is_congruent` tests separately.
        """
        if self.year_convention != other.year_convention:
            return False
        if self.epoch != other.epoch and not epoch_shift_is_congruent(
            other.epoch, self.epoch, self.scales, self.year_convention
        ):
            return False
        shorter, longer = sorted((self.scales, other.scales), key=len)
        return all(
            a.period_years == b.period_years and a.segments == b.segments
            for a, b in zip(shorter, longer)
        )


DEFAULT_HIERARCHY = TemporalHierarchy()

#: Reproduces the hierarchy and epoch used by the specification's worked example.
PATENT_EXAMPLE_HIERARCHY = TemporalHierarchy(
    epoch=EPOCH_PATENT_EXAMPLE,
    scales=DEFAULT_SCALES,
    year_convention="calendar",
)


# ============================================================================
# Epoch migration
# ============================================================================


def epoch_shift_is_congruent(
    old_epoch: datetime,
    new_epoch: datetime,
    scales: Iterable[ScaleSpec],
    year_convention: str = "calendar",
    tolerance_years: float = 1e-9,
) -> bool:
    """
    True if moving from ``old_epoch`` to ``new_epoch`` leaves every phase unchanged.

    Phases survive an epoch change only when the shift is an exact whole multiple of
    *every* period in the hierarchy. Shifting 2010 → 1900 is 110 years: a multiple of
    the 1-year period, but not of the 16-year period (110 mod 16 = 14) or the
    256-year period. Every 16-year and 256-year phase in the corpus would move, so
    that migration requires a re-index.

    Call this before changing ``EMBEDDINGSPIN_EPOCH`` on a populated corpus. If it
    returns False, re-encode rather than mixing epochs.
    """
    shift = abs(years_since_epoch(new_epoch, old_epoch, year_convention))
    for spec in scales:
        remainder = floored_mod(shift, spec.period_years)
        # A remainder just under the period is also congruent (shift is a multiple
        # from below, off only by floating-point error).
        distance = min(remainder, spec.period_years - remainder)
        if distance > tolerance_years:
            return False
    return True


def describe_epoch_migration(
    old: TemporalHierarchy, new: TemporalHierarchy
) -> Dict[str, object]:
    """
    Explain what a hierarchy change costs. Returned keys:

    ``requires_reindex``
        True if stored vectors must be re-encoded.
    ``reasons``
        Human-readable list of what changed and why it matters.
    """
    reasons = []
    if old.year_convention != new.year_convention:
        reasons.append(
            f"year convention {old.year_convention!r} -> {new.year_convention!r}: "
            "phase of every non-January-1 timestamp changes"
        )
    if old.epoch != new.epoch:
        congruent = epoch_shift_is_congruent(
            old.epoch, new.epoch, new.scales, new.year_convention
        )
        if congruent:
            reasons.append(
                f"epoch {old.epoch.date()} -> {new.epoch.date()}: shift is a whole "
                "multiple of every period, phases preserved"
            )
        else:
            offenders = [
                s.name
                for s in new.scales
                if not epoch_shift_is_congruent(old.epoch, new.epoch, [s], new.year_convention)
            ]
            reasons.append(
                f"epoch {old.epoch.date()} -> {new.epoch.date()}: shift is not a whole "
                f"multiple of the {', '.join(offenders)} period(s), phases change"
            )
    old_prefix = old.scales[: len(new.scales)]
    new_prefix = new.scales[: len(old.scales)]
    if old_prefix != new_prefix[: len(old_prefix)]:
        reasons.append("scale list changed in place (not a pure extension)")
    elif len(new.scales) > len(old.scales):
        added = ", ".join(s.name for s in new.scales[len(old.scales):])
        reasons.append(
            f"appended outer scale(s) {added}: additive, existing tuples unchanged; "
            "older vectors read back with a neutral full-circle tuple for the new scale"
        )

    requires_reindex = not old.is_compatible_with(new)
    return {
        "requires_reindex": requires_reindex,
        "from": old.fingerprint(),
        "to": new.fingerprint(),
        "reasons": reasons or ["no change"],
    }
