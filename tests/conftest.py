"""
Pytest Fixtures and Test Utilities
===================================

Provides reusable fixtures for testing the temporal spin retrieval system.
"""

import math
import pytest
from datetime import datetime, timezone, timedelta
from typing import List, Tuple

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from temporal_spin import (
    SpinDocument,
    SpinQuery,
    T0_EPOCH,
    T0_SECONDS,
    QUARTER_PERIOD_SECONDS,
    DECADE_PERIOD_SECONDS,
    CENTURY_PERIOD_SECONDS
)
from llamastack_client import MockEmbeddingClient
from vector_store import InMemoryVectorStore
from ingestion import TemporalSpinIngestionPipeline
from retrieval import TemporalSpinRetriever


# ============================================================================
# Mock Embedding Client Fixture
# ============================================================================

@pytest.fixture
def mock_embedding_client():
    """Provide a mock embedding client for testing."""
    return MockEmbeddingClient(dimension=384)


# ============================================================================
# Vector Store Fixtures
# ============================================================================

@pytest.fixture
def empty_vector_store():
    """Provide an empty in-memory vector store."""
    return InMemoryVectorStore()


@pytest.fixture
def populated_vector_store(mock_embedding_client):
    """Provide a vector store with sample documents."""
    store = InMemoryVectorStore()
    pipeline = TemporalSpinIngestionPipeline(
        embedding_client=mock_embedding_client,
        vector_store=store
    )
    
    # Add documents with known temporal relationships
    documents = get_sample_documents()
    for text, timestamp, metadata in documents:
        pipeline.ingest_document(
            text=text,
            timestamp=timestamp,
            metadata=metadata
        )
    
    return store


# ============================================================================
# Pipeline Fixtures
# ============================================================================

@pytest.fixture
def ingestion_pipeline(mock_embedding_client, empty_vector_store):
    """Provide an ingestion pipeline."""
    return TemporalSpinIngestionPipeline(
        embedding_client=mock_embedding_client,
        vector_store=empty_vector_store
    )


@pytest.fixture
def retriever(mock_embedding_client, populated_vector_store):
    """Provide a retriever with populated data."""
    return TemporalSpinRetriever(
        embedding_client=mock_embedding_client,
        vector_store=populated_vector_store
    )


# ============================================================================
# Sample Data with Known Temporal Relationships
# ============================================================================

def get_sample_documents() -> List[Tuple[str, datetime, dict]]:
    """
    Generate sample documents with known temporal relationships.
    
    Returns:
        List of (text, timestamp, metadata) tuples
    """
    base_date = datetime(2020, 1, 1, tzinfo=timezone.utc)
    
    documents = [
        # Same day documents (should cluster together)
        (
            "Apple announces new iPhone on January 1, 2020",
            base_date,
            {"category": "tech", "company": "Apple"}
        ),
        (
            "Apple revenue report for January 1, 2020",
            base_date,
            {"category": "finance", "company": "Apple"}
        ),
        
        # One week apart (should be close)
        (
            "Apple software update released January 8, 2020",
            base_date + timedelta(days=7),
            {"category": "tech", "company": "Apple"}
        ),
        
        # One month apart (should be moderately close)
        (
            "Apple Q1 2020 earnings call February 1, 2020",
            base_date + timedelta(days=31),
            {"category": "finance", "company": "Apple"}
        ),
        
        # Three months apart (same quarter)
        (
            "Apple spring event March 15, 2020",
            base_date + timedelta(days=75),
            {"category": "tech", "company": "Apple"}
        ),
        
        # Six months apart (different quarter, same year)
        (
            "Apple WWDC 2020 June 22, 2020",
            base_date + timedelta(days=173),
            {"category": "tech", "company": "Apple"}
        ),
        
        # One year apart (should be distant)
        (
            "Apple announces new iPhone on January 1, 2021",
            base_date + timedelta(days=366),
            {"category": "tech", "company": "Apple"}
        ),
        
        # Multiple years apart (should be very distant)
        (
            "Apple announces Vision Pro June 5, 2023",
            datetime(2023, 6, 5, tzinfo=timezone.utc),
            {"category": "tech", "company": "Apple"}
        ),
        
        # Arc document - Q1 2020 (January to March)
        (
            "Apple Q1 2020 performance overview",
            datetime(2020, 1, 1, tzinfo=timezone.utc),
            {"category": "finance", "company": "Apple", "period": "Q1"}
        ),
        
        # Arc document - Full year 2020
        (
            "Apple Annual Report 2020",
            datetime(2020, 1, 1, tzinfo=timezone.utc),
            {"category": "finance", "company": "Apple", "period": "annual"}
        ),
    ]
    
    return documents


def get_quarterly_reports() -> List[Tuple[str, datetime, datetime, dict]]:
    """
    Generate quarterly reports with arc encoding.
    
    Returns:
        List of (text, start_date, end_date, metadata) tuples
    """
    reports = [
        # 2020 Q1
        (
            "Q1 2020 Financial Report - Revenue $58.3B",
            datetime(2020, 1, 1, tzinfo=timezone.utc),
            datetime(2020, 3, 31, tzinfo=timezone.utc),
            {"year": 2020, "quarter": "Q1", "revenue": 58.3}
        ),
        # 2020 Q2
        (
            "Q2 2020 Financial Report - Revenue $59.7B",
            datetime(2020, 4, 1, tzinfo=timezone.utc),
            datetime(2020, 6, 30, tzinfo=timezone.utc),
            {"year": 2020, "quarter": "Q2", "revenue": 59.7}
        ),
        # 2020 Q3
        (
            "Q3 2020 Financial Report - Revenue $64.7B",
            datetime(2020, 7, 1, tzinfo=timezone.utc),
            datetime(2020, 9, 30, tzinfo=timezone.utc),
            {"year": 2020, "quarter": "Q3", "revenue": 64.7}
        ),
        # 2020 Q4
        (
            "Q4 2020 Financial Report - Revenue $111.4B",
            datetime(2020, 10, 1, tzinfo=timezone.utc),
            datetime(2020, 12, 31, tzinfo=timezone.utc),
            {"year": 2020, "quarter": "Q4", "revenue": 111.4}
        ),
        # 2021 Q1
        (
            "Q1 2021 Financial Report - Revenue $89.6B",
            datetime(2021, 1, 1, tzinfo=timezone.utc),
            datetime(2021, 3, 31, tzinfo=timezone.utc),
            {"year": 2021, "quarter": "Q1", "revenue": 89.6}
        ),
    ]
    
    return reports


# ============================================================================
# Temporal Test Assertions
# ============================================================================

def assert_temporal_ordering(results, query_date: datetime, tolerance_days: int = 7):
    """
    Assert that results are ordered by temporal proximity to query date.
    
    Args:
        results: List of retrieval results
        query_date: Query timestamp
        tolerance_days: Allowed tolerance in days
    """
    if len(results) < 2:
        return  # Nothing to compare
    
    for i in range(len(results) - 1):
        current_doc = results[i]
        next_doc = results[i + 1]
        
        current_delta = abs((current_doc.timestamp - query_date).total_seconds())
        next_delta = abs((next_doc.timestamp - query_date).total_seconds())
        
        # Current document should be closer or within tolerance
        tolerance_seconds = tolerance_days * 24 * 3600
        assert current_delta <= next_delta + tolerance_seconds, (
            f"Temporal ordering violated: "
            f"Rank {current_doc.rank} is {current_delta}s from query, "
            f"Rank {next_doc.rank} is {next_delta}s from query"
        )


def assert_phase_alignment(phi1: float, phi2: float, max_difference: float = 0.1):
    """
    Assert that two phase angles are closely aligned.
    
    Args:
        phi1, phi2: Phase angles in radians
        max_difference: Maximum allowed angular difference
    """
    diff = abs(phi1 - phi2) % math.tau
    diff = min(diff, math.tau - diff)  # Shortest arc
    
    assert diff <= max_difference, (
        f"Phase alignment failed: phi1={phi1:.4f}, phi2={phi2:.4f}, "
        f"difference={diff:.4f} > {max_difference}"
    )


def assert_arc_contains_point(arc_start: float, arc_end: float, point: float):
    """
    Assert that an arc contains a point on the unit circle.
    
    Args:
        arc_start, arc_end: Arc boundaries in radians
        point: Point to check
    """
    # Normalize to [0, 2π)
    arc_start = arc_start % math.tau
    arc_end = arc_end % math.tau
    point = point % math.tau
    
    # Handle wrapping
    if arc_end < arc_start:
        # Arc crosses 0
        assert point >= arc_start or point <= arc_end, (
            f"Point {point:.4f} not in arc [{arc_start:.4f}, {arc_end:.4f}]"
        )
    else:
        assert arc_start <= point <= arc_end, (
            f"Point {point:.4f} not in arc [{arc_start:.4f}, {arc_end:.4f}]"
        )


# ============================================================================
# Property-Based Test Helpers
# ============================================================================

def is_valid_phase(phi: float) -> bool:
    """Check if a phase angle is valid (finite, not NaN)."""
    return math.isfinite(phi) and not math.isnan(phi)


def is_valid_spin_vector(spin: List[float]) -> bool:
    """Check if a spin vector is valid (9D, finite values)."""
    return (
        len(spin) == 9 and
        all(math.isfinite(x) for x in spin) and
        all(not math.isnan(x) for x in spin)
    )


def is_unit_circle_point(x: float, y: float, tolerance: float = 1e-6) -> bool:
    """Check if (x, y) is on the unit circle."""
    magnitude = math.sqrt(x**2 + y**2)
    return abs(magnitude - 1.0) < tolerance or magnitude == 0.0


# ============================================================================
# Date Generation Utilities
# ============================================================================

def generate_date_range(
    start: datetime,
    end: datetime,
    step_days: int = 1
) -> List[datetime]:
    """
    Generate a range of dates.
    
    Args:
        start: Start date
        end: End date
        step_days: Step size in days
    
    Returns:
        List of datetime objects
    """
    dates = []
    current = start
    while current <= end:
        dates.append(current)
        current += timedelta(days=step_days)
    return dates


def generate_quarterly_dates(year: int) -> List[Tuple[datetime, datetime]]:
    """
    Generate quarterly date ranges for a year.
    
    Args:
        year: Year to generate quarters for
    
    Returns:
        List of (start_date, end_date) tuples for each quarter
    """
    quarters = [
        (datetime(year, 1, 1, tzinfo=timezone.utc), datetime(year, 3, 31, tzinfo=timezone.utc)),
        (datetime(year, 4, 1, tzinfo=timezone.utc), datetime(year, 6, 30, tzinfo=timezone.utc)),
        (datetime(year, 7, 1, tzinfo=timezone.utc), datetime(year, 9, 30, tzinfo=timezone.utc)),
        (datetime(year, 10, 1, tzinfo=timezone.utc), datetime(year, 12, 31, tzinfo=timezone.utc)),
    ]
    return quarters
