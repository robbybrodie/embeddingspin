"""
Temporal-Phase Spin Retrieval System
=====================================

This module implements a novel retrieval algorithm that encodes time as an angular
spin state on the unit circle, enabling smooth temporal zoom without model retraining.

Core Concept:
-------------
- Each document's timestamp is mapped to an angle φ ∈ [0, 2π) via a periodic function
- Time becomes a 2D spin vector: [cos(φ), sin(φ)]
- Concatenated with semantic embeddings: v_full = [v_semantic, spin_vector]
- At query time, β (zoom factor) controls temporal alignment weighting

No Model Retraining Required:
------------------------------
The semantic embedding model is frozen. Time encoding happens in the vector space
via geometric augmentation, making this approach model-agnostic.

Temporal Zoom:
--------------
β acts as a "zoom knob":
  - β = 0: Pure semantic search (time ignored)
  - β = 100: Weak temporal preference
  - β = 1000: Moderate temporal focus
  - β = 5000: Strong temporal focus (exact year prioritized) [DEFAULT]
  - β = 10000+: Extreme temporal filter
  - score = semantic_sim × exp(-β × (Δφ)²)
"""

import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from dateutil import parser as dateutil_parser

# ============================================================================
# Configuration Constants
# ============================================================================

# Base epoch for timestamp normalization (2010-01-01 00:00:00 UTC)
T0_EPOCH = datetime(2010, 1, 1, tzinfo=timezone.utc)
T0_SECONDS = T0_EPOCH.timestamp()

# Multi-scale temporal encoding periods
# Three scales for hierarchical temporal resolution:
#   - QUARTER_SCALE: 1 year (quarterly precision within a year)
#   - DECADE_SCALE: 16 years (year-to-year discrimination)
#   - CENTURY_SCALE: 256 years (historical context)
# Powers of 2 maintain mathematical consistency across scales
QUARTER_SCALE_YEARS = 1
DECADE_SCALE_YEARS = 16
CENTURY_SCALE_YEARS = 256

QUARTER_PERIOD_SECONDS = QUARTER_SCALE_YEARS * 365.25 * 24 * 3600
DECADE_PERIOD_SECONDS = DECADE_SCALE_YEARS * 365.25 * 24 * 3600
CENTURY_PERIOD_SECONDS = CENTURY_SCALE_YEARS * 365.25 * 24 * 3600

# Weights for multi-scale temporal alignment
# Decade scale gets highest weight for year-to-year queries
QUARTER_WEIGHT = 0.4  # Within-year precision
DECADE_WEIGHT = 0.5  # Year discrimination (highest)
CENTURY_WEIGHT = 0.1  # Historical context

# Default embedding dimension (adjust based on your model)
DEFAULT_EMBEDDING_DIM = 384


# ============================================================================
# Temporal Spin Encoding (Point and Arc)
# ============================================================================


def compute_spin_vector(
    timestamp_seconds: float,
    t0_seconds: float = T0_SECONDS,
    period_seconds: float = None,  # Deprecated - now uses multi-scale
    phase_offset: float = 0.0,
    temporal_scale: float = 1.0,
    end_timestamp_seconds: Optional[float] = None,
) -> Tuple[
    List[float],
    Dict[str, float],
    Dict[str, Optional[float]],
    Dict[str, Optional[float]],
]:
    """
    Map a timestamp (or time interval) to a multi-scale temporal spin vector.

    Multi-scale encoding with 3 hierarchical periods (powers of 2):
    - Quarter scale (1 year): For quarterly precision within a year
    - Decade scale (16 years): For year-to-year discrimination
    - Century scale (256 years): For historical context

    Supports two modes (both return 9D vectors for consistent dimensionality):
    1. Point mode: Single timestamp → 9D vector [x_q, y_q, 0, x_d, y_d, 0, x_c, y_c, 0]
    2. Arc mode: Time period → 9D vector with arc_length in z components

    Args:
        timestamp_seconds: Unix timestamp in seconds (start time for arcs)
        t0_seconds: Base epoch timestamp (default: 2010-01-01)
        period_seconds: Deprecated (kept for backward compatibility, ignored)
        phase_offset: Optional phase shift in radians
        temporal_scale: Scaling factor for spin vector magnitude (default: 1.0)
        end_timestamp_seconds: Optional end timestamp for arc mode. If None, uses point mode.

    Returns:
        Tuple of (spin_vector, phi_centers, phi_starts, phi_ends):
        - spin_vector: Always 9D [x_q, y_q, z_q, x_d, y_d, z_d, x_c, y_c, z_c]
        - phi_centers: Dict with keys 'quarter', 'decade', 'century'
        - phi_starts: Dict with start angles for each scale (None for points)
        - phi_ends: Dict with end angles for each scale (None for points)

    Examples:
        >>> # Point mode - single instant in time
        >>> t = datetime(2023, 6, 15, tzinfo=timezone.utc).timestamp()
        >>> spin, phi_c, phi_s, phi_e = compute_spin_vector(t)
        >>> len(spin)  # Returns 9
        9

        >>> # Arc mode - Q1 2023 (period)
        >>> t_start = datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp()
        >>> t_end = datetime(2023, 3, 31, tzinfo=timezone.utc).timestamp()
        >>> spin, phi_c, phi_s, phi_e = compute_spin_vector(t_start, end_timestamp_seconds=t_end)
        >>> # Quarter scale will show ~90° arc, decade/century scales show smaller arcs
    """
    periods = {
        "quarter": QUARTER_PERIOD_SECONDS,
        "decade": DECADE_PERIOD_SECONDS,
        "century": CENTURY_PERIOD_SECONDS,
    }

    spin_vector = []
    phi_centers = {}
    phi_starts = {}
    phi_ends = {}

    # Encode at each scale
    for scale_name in ["quarter", "decade", "century"]:
        period_sec = periods[scale_name]

        # Point mode: Single timestamp
        if end_timestamp_seconds is None:
            # Normalize time to [0, 1) fractional position within period
            fraction = ((timestamp_seconds - t0_seconds) / period_sec) % 1.0

            # Convert to angle: φ ∈ [0, 2π)
            phi = math.tau * fraction + phase_offset

            # Spin vector on unit circle (3D with arc_length=0 for points)
            cos_phi = math.cos(phi)
            sin_phi = math.sin(phi)
            spin_vector.extend(
                [
                    temporal_scale * cos_phi,
                    temporal_scale * sin_phi,
                    0.0,  # No arc length for points
                ]
            )

            phi_centers[scale_name] = phi
            phi_starts[scale_name] = None
            phi_ends[scale_name] = None

        # Arc mode: Start and end timestamps
        else:
            # Compute start and end angles at this scale
            fraction_start = (
                (timestamp_seconds - t0_seconds) / period_sec
            ) % 1.0
            fraction_end = (
                (end_timestamp_seconds - t0_seconds) / period_sec
            ) % 1.0

            phi_start = math.tau * fraction_start + phase_offset
            phi_end = math.tau * fraction_end + phase_offset

            # Handle wrapping: if end < start, arc crosses 0°
            if phi_end < phi_start:
                phi_end += math.tau

            # Compute arc center and length
            phi_center = (phi_start + phi_end) / 2.0
            arc_length = phi_end - phi_start

            # Normalize phi_center back to [0, 2π)
            phi_center = phi_center % math.tau

            # Spin vector for arcs (3D: center + arc_length)
            cos_center = math.cos(phi_center)
            sin_center = math.sin(phi_center)
            spin_vector.extend(
                [
                    temporal_scale * cos_center,
                    temporal_scale * sin_center,
                    arc_length,  # Arc length in radians (not scaled)
                ]
            )

            phi_centers[scale_name] = phi_center
            phi_starts[scale_name] = phi_start
            phi_ends[scale_name] = phi_end

    return spin_vector, phi_centers, phi_starts, phi_ends


def angular_difference(phi1: float, phi2: float) -> float:
    """
    Compute the smallest angular difference between two angles.

    Returns Δφ ∈ [0, π] (always the shortest arc on the circle).

    Args:
        phi1, phi2: Angles in radians

    Returns:
        Smallest angular distance in radians
    """
    diff = abs(phi1 - phi2) % math.tau
    return min(diff, math.tau - diff)


def _normalize_phi_interval(phi_start, phi_end):
    """
    Normalize a phi interval to [0, 2π) and handle wrapping.
    If the interval covers the full circle or more, treat as full circle.
    Returns (start, end) such that end >= start.
    """
    tau = math.tau
    phi_start = phi_start % tau
    phi_end = phi_end % tau
    # If the interval covers the full circle or more, treat as full circle
    if (phi_end - phi_start) % tau >= tau - 1e-10:
        return 0.0, tau
    if phi_end < phi_start:
        phi_end += tau
    return phi_start, phi_end


def arc_overlap(
    phi_start1: float, phi_end1: float, phi_start2: float, phi_end2: float
) -> float:
    """
    Compute the overlap (intersection) between two arcs on the unit circle.

    Arcs are defined by [phi_start, phi_end]. This function handles wrapping.

    Args:
        phi_start1, phi_end1: First arc (start and end angles in radians)
        phi_start2, phi_end2: Second arc

    Returns:
        Overlap length in radians [0, 2π]

    Example:
        >>> # Two arcs covering Q1 and Q2 of a year
        >>> q1_start, q1_end = 0.0, math.pi/2
        >>> q2_start, q2_end = math.pi/2, math.pi
        >>> overlap = arc_overlap(q1_start, q1_end, q2_start, q2_end)
        >>> # Returns 0.0 (adjacent, no overlap)
    """
    tau = math.tau

    # Check for full circles BEFORE normalization
    raw_len1 = phi_end1 - phi_start1
    raw_len2 = phi_end2 - phi_start2

    is_full1 = abs(raw_len1) >= tau - 1e-10
    is_full2 = abs(raw_len2) >= tau - 1e-10

    # If either is a full circle, return the min of the two arc lengths
    if is_full1 and is_full2:
        return tau
    elif is_full1:
        return min(tau, abs(raw_len2))
    elif is_full2:
        return min(tau, abs(raw_len1))

    # Normalize both intervals for non-full-circle case
    phi_start1, phi_end1 = _normalize_phi_interval(phi_start1, phi_end1)
    phi_start2, phi_end2 = _normalize_phi_interval(phi_start2, phi_end2)

    # Robust circular interval overlap
    # Represent both intervals as (start, length)
    def interval_to_segments(start, end):
        # Returns a list of (seg_start, seg_end) in [0, 2π), handling wrapping
        tau = math.tau
        if end <= tau and start < end:
            return [(start, end)]
        elif end > tau:
            # Wraps around 2π
            return [(start, tau), (0.0, end - tau)]
        else:
            # Should not happen after normalization, but fallback
            return [(start, end)]

    tau = math.tau
    segs1 = interval_to_segments(phi_start1, phi_end1)
    segs2 = interval_to_segments(phi_start2, phi_end2)
    overlap = 0.0
    for s1, e1 in segs1:
        for s2, e2 in segs2:
            # Compute overlap between [s1, e1] and [s2, e2]
            left = max(s1, s2)
            right = min(e1, e2)
            if right > left:
                overlap += right - left
    return min(overlap, tau)


def jaccard_similarity_arcs(
    phi_start1: float, phi_end1: float, phi_start2: float, phi_end2: float
) -> float:
    """
    Compute Jaccard similarity between two arcs on the unit circle.

    Jaccard = |intersection| / |union|

    Args:
        phi_start1, phi_end1: First arc
        phi_start2, phi_end2: Second arc

    Returns:
        Jaccard similarity in [0, 1]

    Example:
        >>> # Annual report (full year) vs Q2 (quarter)
        >>> year_start, year_end = 0.0, 2*math.pi
        >>> q2_start, q2_end = math.pi/2, math.pi
        >>> sim = jaccard_similarity_arcs(year_start, year_end, q2_start, q2_end)
        >>> # Returns 0.25 (quarter is 25% of year)
    """
    # Normalize angles to [0, 2π)
    phi_start1 = phi_start1 % math.tau
    phi_end1 = phi_end1 % math.tau
    phi_start2 = phi_start2 % math.tau
    phi_end2 = phi_end2 % math.tau

    # Helper to compute arc length with proper wrapping
    def _arc_length(start, end):
        if end >= start:
            return end - start
        else:
            return math.tau - start + end

    arc1_length = _arc_length(phi_start1, phi_end1)
    arc2_length = _arc_length(phi_start2, phi_end2)

    # Compute intersection
    intersection = arc_overlap(phi_start1, phi_end1, phi_start2, phi_end2)

    # Compute union
    union = arc1_length + arc2_length - intersection

    # Handle edge cases with small epsilon
    epsilon = 1e-10
    if union <= epsilon:
        return 1.0 if intersection > epsilon else 0.0

    return intersection / union


# ============================================================================
# Timestamp Extraction
# ============================================================================

# Common date patterns in financial/corporate documents
DATE_PATTERNS = [
    # "for the period ended 31 December 2019"
    r"period\s+ended\s+(\d{1,2}\s+\w+\s+\d{4})",
    # "as of December 31, 2019"
    r"as\s+of\s+(\w+\s+\d{1,2},?\s+\d{4})",
    # "fiscal year 2019"
    r"fiscal\s+year\s+(\d{4})",
    # "Q4 2019", "Q1 2020"
    r"Q[1-4]\s+(\d{4})",
    # ISO format: "2019-12-31"
    r"(\d{4}-\d{2}-\d{2})",
    # US format: "12/31/2019"
    r"(\d{1,2}/\d{1,2}/\d{4})",
]


def extract_timestamp_from_text(
    text: str, fallback: Optional[datetime] = None
) -> datetime:
    """
    Extract timestamp from document text using regex patterns and dateutil.

    Strategy:
    1. Try regex patterns for common corporate/financial date formats
    2. Use dateutil fuzzy parsing as fallback
    3. Use provided fallback or current time if all else fails

    Args:
        text: Document text to parse
        fallback: Fallback datetime if extraction fails

    Returns:
        Extracted datetime (timezone-aware UTC)
    """
    # Try each regex pattern
    for pattern in DATE_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            date_str = match.group(1)
            try:
                # Parse the extracted date string
                dt = dateutil_parser.parse(date_str, fuzzy=True)
                # Ensure UTC timezone
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except (ValueError, TypeError):
                continue

    # Fallback: try fuzzy parsing on entire text (first 500 chars)
    try:
        dt = dateutil_parser.parse(text[:500], fuzzy=True)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        pass

    # Final fallback
    if fallback:
        return fallback
    return datetime.now(timezone.utc)


# ============================================================================
# Document and Query Representations
# ============================================================================


@dataclass
class SpinDocument:
    """
    A document with multi-scale temporal-phase spin encoding (point or arc mode).

    Attributes:
        doc_id: Unique identifier
        text: Original document text
        timestamp: Document timestamp (UTC) - start time for arcs
        semantic_embedding: Semantic embedding vector from model
        spin_vector: Always 9D [x_q, y_q, z_q, x_d, y_d, z_d, x_c, y_c, z_c]
                    - z components are 0 for points (instant)
                    - z components >0 for arcs (time period)
        phi: Dict with phase angles for each scale {'quarter': φ_q, 'decade': φ_d, 'century': φ_c}
        full_embedding: Concatenated [semantic_embedding + spin_vector]
        end_timestamp: Optional end timestamp for arc mode (None for points)
        phi_start: Dict with start angles for each scale (None values for points)
        phi_end: Dict with end angles for each scale (None values for points)
        is_arc: True if this is an arc (time period), False if point (instant)
    """

    doc_id: str
    text: str
    timestamp: datetime
    semantic_embedding: List[float]
    spin_vector: List[float]
    phi: Dict[str, float]
    full_embedding: List[float]
    metadata: Dict[str, Any] = None
    end_timestamp: Optional[datetime] = None
    phi_start: Optional[Dict[str, Optional[float]]] = None
    phi_end: Optional[Dict[str, Optional[float]]] = None
    is_arc: bool = False

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        # Auto-detect arc mode
        if self.end_timestamp is not None:
            self.is_arc = True


@dataclass
class SpinQuery:
    """
    A query with multi-scale temporal-phase spin encoding (point or arc mode).

    Attributes:
        query_text: Query string
        query_timestamp: Target timestamp for retrieval (start for arcs)
        semantic_embedding: Semantic query embedding
        spin_vector: Always 9D [x_q, y_q, z_q, x_d, y_d, z_d, x_c, y_c, z_c]
        phi: Dict with phase angles for each scale {'quarter': φ_q, 'decade': φ_d, 'century': φ_c}
        lambda_factor: Weight for spin component (default: 1.0)
        full_embedding: Concatenated query vector
        end_timestamp: Optional end timestamp for arc queries
        phi_start: Dict with start angles for each scale (None values for points)
        phi_end: Dict with end angles for each scale (None values for points)
        is_arc: True if querying a time period
    """

    query_text: str
    query_timestamp: datetime
    semantic_embedding: List[float]
    spin_vector: List[float]
    phi: Dict[str, float]
    lambda_factor: float = 1.0
    full_embedding: List[float] = None
    end_timestamp: Optional[datetime] = None
    phi_start: Optional[Dict[str, Optional[float]]] = None
    phi_end: Optional[Dict[str, Optional[float]]] = None
    is_arc: bool = False

    def __post_init__(self):
        if self.full_embedding is None:
            # Weighted concatenation: [semantic + λ * spin]
            weighted_spin = [self.lambda_factor * x for x in self.spin_vector]
            self.full_embedding = self.semantic_embedding + weighted_spin


@dataclass
class RetrievalResult:
    """
    A single retrieval result with scores and metadata.
    """

    doc_id: str
    text: str
    timestamp: datetime
    semantic_score: float
    phi_doc: float
    phi_query: float
    phi_difference: float
    temporal_alignment: float  # exp(-β × (Δφ)²)
    combined_score: float
    rank: int = 0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


# ============================================================================
# Embedding Utilities
# ============================================================================


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    Returns value in [-1, 1], where 1 = identical direction.
    """
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must have same dimension")

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def normalize_vector(vec: List[float]) -> List[float]:
    """L2-normalize a vector to unit length."""
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0:
        return vec
    return [x / norm for x in vec]
