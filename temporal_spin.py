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
  - β ≈ 0: broad semantic search across all time periods
  - β > 10: sharp temporal focus around query timestamp
  - score = semantic_sim × exp(-β × (Δφ)²)
"""

import math
import re
from datetime import datetime, timezone
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from dateutil import parser as dateutil_parser


# ============================================================================
# Configuration Constants
# ============================================================================

# Base epoch for timestamp normalization (2010-01-01 00:00:00 UTC)
T0_EPOCH = datetime(2010, 1, 1, tzinfo=timezone.utc)
T0_SECONDS = T0_EPOCH.timestamp()

# Period for spin encoding (365.25 days × 10 years = 10-year cycle)
PERIOD_SECONDS = 365.25 * 24 * 3600 * 10

# Default embedding dimension (adjust based on your model)
DEFAULT_EMBEDDING_DIM = 384


# ============================================================================
# Temporal Spin Encoding
# ============================================================================

def compute_spin_vector(
    timestamp_seconds: float,
    t0_seconds: float = T0_SECONDS,
    period_seconds: float = PERIOD_SECONDS,
    phase_offset: float = 0.0
) -> Tuple[List[float], float]:
    """
    Map a timestamp to a unit-circle spin vector.
    
    The timestamp is normalized to a fraction of the period, then converted
    to an angle φ ∈ [0, 2π). This creates a continuous, periodic representation
    of time as a 2D rotation.
    
    Args:
        timestamp_seconds: Unix timestamp in seconds
        t0_seconds: Base epoch timestamp (default: 2010-01-01)
        period_seconds: Period length for full rotation (default: 10 years)
        phase_offset: Optional phase shift in radians
    
    Returns:
        (spin_vector, phi): A 2D unit vector [cos(φ), sin(φ)] and angle φ
    
    Example:
        >>> t = datetime(2015, 1, 1, tzinfo=timezone.utc).timestamp()
        >>> spin, phi = compute_spin_vector(t)
        >>> # Documents 10 years apart have the same spin (periodic)
    """
    if period_seconds <= 0:
        raise ValueError("period_seconds must be positive")
    
    # Normalize time to [0, 1) fractional position within period
    fraction = ((timestamp_seconds - t0_seconds) / period_seconds) % 1.0
    
    # Convert to angle: φ ∈ [0, 2π)
    phi = math.tau * fraction + phase_offset  # tau = 2π
    
    # Spin vector on unit circle
    spin_vector = [math.cos(phi), math.sin(phi)]
    
    return spin_vector, phi


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


# ============================================================================
# Timestamp Extraction
# ============================================================================

# Common date patterns in financial/corporate documents
DATE_PATTERNS = [
    # "for the period ended 31 December 2019"
    r'period\s+ended\s+(\d{1,2}\s+\w+\s+\d{4})',
    # "as of December 31, 2019"
    r'as\s+of\s+(\w+\s+\d{1,2},?\s+\d{4})',
    # "fiscal year 2019"
    r'fiscal\s+year\s+(\d{4})',
    # "Q4 2019", "Q1 2020"
    r'Q[1-4]\s+(\d{4})',
    # ISO format: "2019-12-31"
    r'(\d{4}-\d{2}-\d{2})',
    # US format: "12/31/2019"
    r'(\d{1,2}/\d{1,2}/\d{4})',
]


def extract_timestamp_from_text(
    text: str,
    fallback: Optional[datetime] = None
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
    A document with temporal-phase spin encoding.
    
    Attributes:
        doc_id: Unique identifier
        text: Original document text
        timestamp: Document timestamp (UTC)
        semantic_embedding: Semantic embedding vector from model
        spin_vector: 2D temporal spin vector [cos(φ), sin(φ)]
        phi: Phase angle in radians
        full_embedding: Concatenated [semantic_embedding + spin_vector]
    """
    doc_id: str
    text: str
    timestamp: datetime
    semantic_embedding: List[float]
    spin_vector: List[float]
    phi: float
    full_embedding: List[float]
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class SpinQuery:
    """
    A query with temporal-phase spin encoding.
    
    Attributes:
        query_text: Query string
        query_timestamp: Target timestamp for retrieval
        semantic_embedding: Semantic query embedding
        spin_vector: 2D temporal spin vector
        phi: Phase angle in radians
        lambda_factor: Weight for spin component (default: 1.0)
        full_embedding: Concatenated query vector
    """
    query_text: str
    query_timestamp: datetime
    semantic_embedding: List[float]
    spin_vector: List[float]
    phi: float
    lambda_factor: float = 1.0
    full_embedding: List[float] = None
    
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

