"""
Tests for Document Ingestion Pipeline
======================================

Tests the ingestion pipeline:
- Single document ingestion (point and arc modes)
- Batch ingestion
- Timestamp extraction from text
- Embedding generation and storage
- Vector store integration
- Metadata handling
"""

import sys
import os
from datetime import datetime, timezone, timedelta
from typing import TYPE_CHECKING

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from temporal_spin import SpinDocument  # noqa: E402
from tests.conftest import get_quarterly_reports  # noqa: E402

if TYPE_CHECKING:
    from ingestion import TemporalSpinIngestionPipeline  # noqa: E402, F401
    from vector_store import InMemoryVectorStore  # noqa: E402, F401


# ============================================================================
# Tests for Single Document Ingestion
# ============================================================================

class TestSingleDocumentIngestion:
    """Test ingestion of individual documents."""

    def test_ingest_document_with_explicit_timestamp(
        self, ingestion_pipeline: "TemporalSpinIngestionPipeline"
    ) -> None:
        """Should ingest document with provided timestamp."""
        text = "Apple announces new iPhone on January 1, 2020"
        timestamp = datetime(2020, 1, 1, tzinfo=timezone.utc)

        doc = ingestion_pipeline.ingest_document(
            text=text,
            timestamp=timestamp,
            doc_id="test_doc_1"
        )

        assert doc is not None
        assert isinstance(doc, SpinDocument)
        assert doc.doc_id == "test_doc_1"
        assert doc.text == text
        assert doc.timestamp == timestamp
        assert len(doc.spin_vector) == 9  # Multi-scale: 3 scales × 3D
        assert len(doc.full_embedding) > 9  # Semantic + spin

    def test_ingest_document_extracts_timestamp_from_text(
        self, ingestion_pipeline
    ):
        """Should extract timestamp when not provided."""
        text = "For the period ended 31 December 2019, revenue increased."

        doc = ingestion_pipeline.ingest_document(
            text=text,
            doc_id="test_doc_2"
        )

        assert doc is not None
        assert doc.timestamp.year == 2019
        assert doc.timestamp.month == 12

    def test_ingest_document_generates_doc_id(
        self, ingestion_pipeline: "TemporalSpinIngestionPipeline"
    ) -> None:
        """Should generate doc_id if not provided."""
        text = "Test document"
        timestamp = datetime(2020, 1, 1, tzinfo=timezone.utc)

        doc = ingestion_pipeline.ingest_document(
            text=text,
            timestamp=timestamp
        )

        assert doc.doc_id is not None
        assert len(doc.doc_id) > 0

    def test_ingest_document_stores_metadata(
        self, ingestion_pipeline: "TemporalSpinIngestionPipeline"
    ) -> None:
        """Should store document metadata."""
        text = "Test document with metadata"
        timestamp = datetime(2020, 1, 1, tzinfo=timezone.utc)
        metadata = {"category": "tech", "author": "John Doe"}

        doc = ingestion_pipeline.ingest_document(
            text=text,
            timestamp=timestamp,
            metadata=metadata
        )

        assert doc.metadata == metadata

    def test_ingest_document_point_mode(
        self, ingestion_pipeline: "TemporalSpinIngestionPipeline"
    ) -> None:
        """Point mode ingestion should have z=0."""
        text = "News article from January 1, 2020"
        timestamp = datetime(2020, 1, 1, tzinfo=timezone.utc)

        doc = ingestion_pipeline.ingest_document(
            text=text,
            timestamp=timestamp
        )

        # Point mode: no end_timestamp
        assert doc.end_timestamp is None
        assert doc.is_arc is False

        # z components should be 0 (indices 2, 5, 8)
        assert doc.spin_vector[2] == 0.0
        assert doc.spin_vector[5] == 0.0
        assert doc.spin_vector[8] == 0.0

    def test_ingest_document_arc_mode(
        self, ingestion_pipeline: "TemporalSpinIngestionPipeline"
    ) -> None:
        """Arc mode ingestion should have non-zero z."""
        text = "Q1 2020 financial report"
        start = datetime(2020, 1, 1, tzinfo=timezone.utc)
        end = datetime(2020, 3, 31, tzinfo=timezone.utc)

        doc = ingestion_pipeline.ingest_document(
            text=text,
            timestamp=start,
            end_timestamp=end
        )

        # Arc mode: end_timestamp provided
        assert doc.end_timestamp == end
        assert doc.is_arc is True

        # z components should be non-zero (arc lengths)
        assert doc.spin_vector[2] > 0
        assert doc.spin_vector[5] > 0
        assert doc.spin_vector[8] > 0

    def test_ingest_document_added_to_vector_store(
        self,
        ingestion_pipeline: "TemporalSpinIngestionPipeline",
        empty_vector_store: "InMemoryVectorStore"
    ) -> None:
        """Ingested document should be added to vector store."""
        initial_count = empty_vector_store.count()

        ingestion_pipeline.ingest_document(
            text="Test document",
            timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc),
            doc_id="test_doc"
        )

        final_count = empty_vector_store.count()
        assert final_count == initial_count + 1

        # Verify document is retrievable
        retrieved = empty_vector_store.get_document("test_doc")
        assert retrieved is not None
        assert retrieved.doc_id == "test_doc"


# ============================================================================
# Tests for Batch Ingestion
# ============================================================================

class TestBatchIngestion:
    """Test batch ingestion of multiple documents."""

    def test_ingest_batch_multiple_documents(
        self,
        ingestion_pipeline: "TemporalSpinIngestionPipeline",
        empty_vector_store: "InMemoryVectorStore"
    ) -> None:
        """Should ingest multiple documents in batch."""
        texts = [
            "Document 1 about technology",
            "Document 2 about finance",
            "Document 3 about healthcare"
        ]
        timestamps = [
            datetime(2020, 1, 1, tzinfo=timezone.utc),
            datetime(2020, 2, 1, tzinfo=timezone.utc),
            datetime(2020, 3, 1, tzinfo=timezone.utc)
        ]
        doc_ids = ["doc_1", "doc_2", "doc_3"]

        docs = ingestion_pipeline.ingest_batch(
            texts=texts,
            timestamps=timestamps,
            doc_ids=doc_ids
        )

        assert len(docs) == 3
        assert empty_vector_store.count() == 3

        # Verify each document
        for i, doc_id in enumerate(doc_ids):
            retrieved = empty_vector_store.get_document(doc_id)
            assert retrieved is not None
            assert retrieved.text == texts[i]
            assert retrieved.timestamp == timestamps[i]

    def test_ingest_batch_with_arc_mode(
        self,
        ingestion_pipeline: "TemporalSpinIngestionPipeline",
        empty_vector_store: "InMemoryVectorStore"
    ) -> None:
        """Should ingest batch with arc-encoded documents."""
        reports = get_quarterly_reports()

        texts = [text for text, _, _, _ in reports]
        starts = [start for _, start, _, _ in reports]
        ends = [end for _, _, end, _ in reports]
        metadatas = [metadata for _, _, _, metadata in reports]

        docs = ingestion_pipeline.ingest_batch(
            texts=texts,
            timestamps=starts,
            end_timestamps=ends,
            metadatas=metadatas
        )

        assert len(docs) == len(reports)

        # Verify arc encoding
        for doc in docs:
            assert doc.is_arc is True
            assert doc.end_timestamp is not None
            assert doc.spin_vector[2] > 0  # Arc length

    def test_ingest_batch_mixed_point_and_arc(
        self, ingestion_pipeline: "TemporalSpinIngestionPipeline"
    ) -> None:
        """Should handle mixed point and arc documents in batch."""
        texts = ["Point doc", "Arc doc"]
        timestamps = [
            datetime(2020, 1, 1, tzinfo=timezone.utc),
            datetime(2020, 1, 1, tzinfo=timezone.utc)
        ]
        end_timestamps = [
            None,  # Point mode
            datetime(2020, 3, 31, tzinfo=timezone.utc)  # Arc mode
        ]

        docs = ingestion_pipeline.ingest_batch(
            texts=texts,
            timestamps=timestamps,
            end_timestamps=end_timestamps
        )

        assert len(docs) == 2
        assert docs[0].is_arc is False
        assert docs[1].is_arc is True

    def test_ingest_batch_generates_doc_ids(
        self, ingestion_pipeline: "TemporalSpinIngestionPipeline"
    ) -> None:
        """Should generate doc_ids for batch if not provided."""
        texts = ["Doc 1", "Doc 2", "Doc 3"]
        timestamps = [datetime(2020, 1, 1, tzinfo=timezone.utc)] * 3

        docs = ingestion_pipeline.ingest_batch(
            texts=texts,
            timestamps=timestamps
        )

        assert len(docs) == 3

        # All should have unique doc_ids
        doc_ids = [doc.doc_id for doc in docs]
        assert len(doc_ids) == len(set(doc_ids))  # All unique

    def test_ingest_batch_with_partial_doc_ids(
        self, ingestion_pipeline: "TemporalSpinIngestionPipeline"
    ) -> None:
        """Should handle partial doc_id list."""
        texts = ["Doc 1", "Doc 2", "Doc 3"]
        timestamps = [datetime(2020, 1, 1, tzinfo=timezone.utc)] * 3
        doc_ids = ["explicit_id", None, None]  # Only first has ID

        docs = ingestion_pipeline.ingest_batch(
            texts=texts,
            timestamps=timestamps,
            doc_ids=doc_ids
        )

        assert docs[0].doc_id == "explicit_id"
        assert docs[1].doc_id is not None  # Generated
        assert docs[2].doc_id is not None  # Generated

    def test_ingest_batch_extracts_timestamps(
        self, ingestion_pipeline: "TemporalSpinIngestionPipeline"
    ) -> None:
        """Should extract timestamps from text when not provided."""
        texts = [
            "For fiscal year 2020, revenue was $100M",
            "For fiscal year 2021, revenue was $150M"
        ]

        docs = ingestion_pipeline.ingest_batch(
            texts=texts,
            timestamps=[None, None]  # Will extract
        )

        assert docs[0].timestamp.year == 2020
        assert docs[1].timestamp.year == 2021


# ============================================================================
# Tests for Embedding Generation
# ============================================================================

class TestEmbeddingGeneration:
    """Test embedding generation and concatenation."""

    def test_semantic_embedding_dimension(
        self, ingestion_pipeline, mock_embedding_client
    ):
        """Semantic embedding should match client dimension."""
        doc = ingestion_pipeline.ingest_document(
            text="Test document",
            timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc)
        )

        expected_dim = mock_embedding_client.dimension
        semantic_dim = len(doc.semantic_embedding)

        assert semantic_dim == expected_dim

    def test_full_embedding_concatenation(
        self,
        ingestion_pipeline: "TemporalSpinIngestionPipeline",
        mock_embedding_client: object
    ) -> None:
        """Full embedding should be semantic + spin (9D)."""
        doc = ingestion_pipeline.ingest_document(
            text="Test document",
            timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc)
        )

        semantic_dim = mock_embedding_client.dimension
        spin_dim = 9  # Multi-scale: 3 scales × 3D
        expected_full_dim = semantic_dim + spin_dim

        assert len(doc.full_embedding) == expected_full_dim

        # Verify concatenation
        assert doc.full_embedding[:semantic_dim] == doc.semantic_embedding
        assert doc.full_embedding[semantic_dim:] == doc.spin_vector

    def test_different_texts_different_embeddings(
        self, ingestion_pipeline: "TemporalSpinIngestionPipeline"
    ) -> None:
        """Different texts should produce different semantic embeddings."""
        doc1 = ingestion_pipeline.ingest_document(
            text="Apple announces new iPhone",
            timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc)
        )

        doc2 = ingestion_pipeline.ingest_document(
            text="Microsoft releases Windows update",
            timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc)
        )

        # Semantic embeddings should differ (mock client uses simple hash)
        assert doc1.semantic_embedding != doc2.semantic_embedding


# ============================================================================
# Tests for Timestamp Handling
# ============================================================================

class TestTimestampHandling:
    """Test various timestamp handling scenarios."""

    def test_timezone_naive_converted_to_utc(
        self, ingestion_pipeline: "TemporalSpinIngestionPipeline"
    ) -> None:
        """Naive timestamps should be converted to UTC."""
        timestamp_naive = datetime(2020, 1, 1)  # No timezone

        doc = ingestion_pipeline.ingest_document(
            text="Test",
            timestamp=timestamp_naive
        )

        assert doc.timestamp.tzinfo is not None
        assert doc.timestamp.tzinfo == timezone.utc

    def test_non_utc_timestamp_converted(
        self, ingestion_pipeline: "TemporalSpinIngestionPipeline"
    ) -> None:
        """Non-UTC timestamps should be converted to UTC."""
        # Create EST timestamp (UTC-5)
        from datetime import timezone as tz
        est = tz(timedelta(hours=-5))
        timestamp_est = datetime(2020, 1, 1, 12, 0, 0, tzinfo=est)

        doc = ingestion_pipeline.ingest_document(
            text="Test",
            timestamp=timestamp_est
        )

        # Should be stored in UTC
        assert doc.timestamp.tzinfo == timezone.utc
        # Time should be adjusted (12:00 EST = 17:00 UTC)
        assert doc.timestamp.hour == 17

    def test_future_timestamp_accepted(
        self, ingestion_pipeline: "TemporalSpinIngestionPipeline"
    ) -> None:
        """Future timestamps should be accepted."""
        future_date = datetime(2030, 1, 1, tzinfo=timezone.utc)

        doc = ingestion_pipeline.ingest_document(
            text="Test",
            timestamp=future_date
        )

        assert doc.timestamp == future_date

    def test_very_old_timestamp_accepted(
        self, ingestion_pipeline: "TemporalSpinIngestionPipeline"
    ) -> None:
        """Very old timestamps should be accepted."""
        old_date = datetime(1990, 1, 1, tzinfo=timezone.utc)

        doc = ingestion_pipeline.ingest_document(
            text="Test",
            timestamp=old_date
        )

        assert doc.timestamp == old_date


# ============================================================================
# Tests for Metadata Handling
# ============================================================================

class TestMetadataHandling:
    """Test metadata storage and retrieval."""

    def test_metadata_preserved_in_document(
        self, ingestion_pipeline: "TemporalSpinIngestionPipeline"
    ) -> None:
        """Metadata should be preserved in SpinDocument."""
        metadata = {
            "author": "John Doe",
            "category": "technology",
            "tags": ["AI", "ML", "deep learning"],
            "priority": 5
        }

        doc = ingestion_pipeline.ingest_document(
            text="Test",
            timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc),
            metadata=metadata
        )

        assert doc.metadata == metadata
        assert doc.metadata["author"] == "John Doe"
        assert "AI" in doc.metadata["tags"]

    def test_metadata_preserved_in_vector_store(
        self,
        ingestion_pipeline: "TemporalSpinIngestionPipeline",
        empty_vector_store: "InMemoryVectorStore"
    ) -> None:
        """Metadata should be retrievable from vector store."""
        metadata = {"company": "Apple", "quarter": "Q1"}

        doc = ingestion_pipeline.ingest_document(
            text="Test",
            timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc),
            doc_id="test_metadata",
            metadata=metadata
        )

        retrieved = empty_vector_store.get_document("test_metadata")
        assert retrieved.metadata == metadata

    def test_batch_ingestion_preserves_metadata(
        self, ingestion_pipeline: "TemporalSpinIngestionPipeline"
    ) -> None:
        """Batch ingestion should preserve metadata for each document."""
        metadatas = [
            {"id": 1, "type": "report"},
            {"id": 2, "type": "article"},
            {"id": 3, "type": "memo"}
        ]

        docs = ingestion_pipeline.ingest_batch(
            texts=["Doc 1", "Doc 2", "Doc 3"],
            timestamps=[datetime(2020, 1, 1, tzinfo=timezone.utc)] * 3,
            metadatas=metadatas
        )

        for i, doc in enumerate(docs):
            assert doc.metadata == metadatas[i]


# ============================================================================
# Tests for Error Handling
# ============================================================================

class TestErrorHandling:
    """Test error handling in ingestion pipeline."""

    def test_empty_text_handled(
        self, ingestion_pipeline: "TemporalSpinIngestionPipeline"
    ) -> None:
        """Empty text should be handled gracefully."""
        doc = ingestion_pipeline.ingest_document(
            text="",
            timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc)
        )

        assert doc is not None
        assert doc.text == ""

    def test_very_long_text_handled(
        self, ingestion_pipeline: "TemporalSpinIngestionPipeline"
    ) -> None:
        """Very long text should be handled."""
        long_text = "A" * 10000

        doc = ingestion_pipeline.ingest_document(
            text=long_text,
            timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc)
        )

        assert doc is not None
        assert len(doc.text) == 10000

    def test_special_characters_in_text(
        self, ingestion_pipeline: "TemporalSpinIngestionPipeline"
    ) -> None:
        """Special characters should be handled."""
        text = "Test with émojis 🚀 and spëcial çharacters!"

        doc = ingestion_pipeline.ingest_document(
            text=text,
            timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc)
        )

        assert doc.text == text

    def test_duplicate_doc_id_overwritten(
        self,
        ingestion_pipeline: "TemporalSpinIngestionPipeline",
        empty_vector_store: "InMemoryVectorStore"
    ) -> None:
        """Duplicate doc_id should overwrite previous document."""
        doc_id = "duplicate_test"

        # First ingestion
        doc1 = ingestion_pipeline.ingest_document(
            text="First version",
            timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc),
            doc_id=doc_id
        )

        # Second ingestion with same ID
        doc2 = ingestion_pipeline.ingest_document(
            text="Second version",
            timestamp=datetime(2020, 2, 1, tzinfo=timezone.utc),
            doc_id=doc_id
        )

        # Should be overwritten
        retrieved = empty_vector_store.get_document(doc_id)
        assert retrieved.text == "Second version"


# ============================================================================
# Tests for Arc Period Validation
# ============================================================================

class TestArcPeriodValidation:
    """Test validation of arc periods."""

    def test_arc_end_before_start_handled(
        self, ingestion_pipeline: "TemporalSpinIngestionPipeline"
    ) -> None:
        """Arc with end before start should be handled."""
        start = datetime(2020, 3, 31, tzinfo=timezone.utc)
        end = datetime(2020, 1, 1, tzinfo=timezone.utc)  # Before start

        # Should either swap or raise error - implementation dependent
        # At minimum, should not crash
        try:
            doc = ingestion_pipeline.ingest_document(
                text="Invalid arc",
                timestamp=start,
                end_timestamp=end
            )
            # If accepted, end should be after start in normalized form
            # or arc should handle wrapping
            assert doc is not None
        except ValueError:
            # Acceptable to raise error for invalid period
            pass

    def test_arc_same_start_and_end(
        self, ingestion_pipeline: "TemporalSpinIngestionPipeline"
    ) -> None:
        """Arc with same start and end should be handled."""
        timestamp = datetime(2020, 1, 1, tzinfo=timezone.utc)

        doc = ingestion_pipeline.ingest_document(
            text="Zero-length arc",
            timestamp=timestamp,
            end_timestamp=timestamp
        )

        # Should produce arc with zero length
        # z components should be 0 or very small
        assert doc.is_arc is True
        assert abs(doc.spin_vector[2]) < 0.01  # Quarter scale

    def test_arc_very_long_period(
        self, ingestion_pipeline: "TemporalSpinIngestionPipeline"
    ) -> None:
        """Arc spanning multiple years should work."""
        start = datetime(2010, 1, 1, tzinfo=timezone.utc)
        end = datetime(2025, 12, 31, tzinfo=timezone.utc)  # 15+ years

        doc = ingestion_pipeline.ingest_document(
            text="Multi-year period",
            timestamp=start,
            end_timestamp=end
        )

        assert doc.is_arc is True
        # Should have large arc lengths at all scales
        assert doc.spin_vector[2] > 0  # Quarter scale
        assert doc.spin_vector[5] > 0  # Decade scale
        assert doc.spin_vector[8] > 0  # Century scale


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
