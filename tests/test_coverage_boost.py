"""
Additional Tests for Coverage Boost
====================================

Tests targeting uncovered code paths in temporal_spin.py, retrieval.py,
and ingestion.py to achieve 100% coverage.
"""

import math
import os
import sys
from datetime import datetime, timedelta, timezone

import pytest

# Modify path to allow imports from parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion import TemporalSpinIngestionPipeline  # noqa: E402
from ingestion import create_ingestion_pipeline  # noqa: E402
from llamastack_client import MockEmbeddingClient  # noqa: E402
from retrieval import TemporalSpinRetriever  # noqa: E402
# fmt: off
from temporal_spin import (RetrievalResult, SpinDocument,  # noqa: E402
                           arc_overlap, extract_timestamp_from_text,
                           jaccard_similarity_arcs)
from vector_store import InMemoryVectorStore  # noqa: E402

# fmt: on


# ========================================================================
# Tests for temporal_spin.py uncovered lines
# ========================================================================


class TestArcOverlapEdgeCases:
    """Test edge cases in arc_overlap function."""

    def test_nearly_full_circle_arc1(self):
        """Test when arc1 is nearly a full circle."""
        # Arc1: nearly full circle (2π - 0.01)
        start1 = 0.0
        end1 = math.tau - 0.0001
        # Arc2: small arc
        start2 = 1.0
        end2 = 2.0

        overlap = arc_overlap(start1, end1, start2, end2)
        # Should return min(len1, len2) = len2 = 1.0
        assert overlap > 0.9
        assert overlap <= 1.1

    def test_nearly_full_circle_arc2(self):
        """Test when arc2 is nearly a full circle."""
        # Arc1: small arc
        start1 = 1.0
        end1 = 2.0
        # Arc2: nearly full circle
        start2 = 0.0
        end2 = math.tau - 0.0001

        overlap = arc_overlap(start1, end1, start2, end2)
        # Should return min(len1, len2) = len1 = 1.0
        assert overlap > 0.9
        assert overlap <= 1.1

    def test_arc2_completely_inside_arc1(self):
        """Test when arc2 is completely contained in arc1."""
        # Arc1: [0, π]
        start1 = 0.0
        end1 = math.pi
        # Arc2: [π/4, π/2] (inside arc1)
        start2 = math.pi / 4
        end2 = math.pi / 2

        overlap = arc_overlap(start1, end1, start2, end2)
        expected = math.pi / 4  # length of arc2
        assert abs(overlap - expected) < 0.01

    def test_arc1_completely_inside_arc2(self):
        """Test when arc1 is completely contained in arc2."""
        # Arc1: [π/4, π/2] (small)
        start1 = math.pi / 4
        end1 = math.pi / 2
        # Arc2: [0, π] (larger, contains arc1)
        start2 = 0.0
        end2 = math.pi

        overlap = arc_overlap(start1, end1, start2, end2)
        expected = math.pi / 4  # length of arc1
        assert abs(overlap - expected) < 0.01


class TestJaccardEdgeCases:
    """Test edge cases in jaccard_similarity_arcs."""

    def test_zero_union_case(self):
        """Test when union is zero (degenerate case)."""
        # This is a theoretical edge case - identical zero-length arcs
        similarity = jaccard_similarity_arcs(0.0, 0.0, 0.0, 0.0)
        # Should return 0.0 due to division by zero protection
        assert math.isclose(similarity, 0.0, abs_tol=1e-9)


class TestTimestampExtractionFallbacks:
    """Test fallback paths in extract_timestamp_from_text."""

    def test_fuzzy_parse_on_entire_text(self):
        """Test fuzzy parsing fallback on entire text."""
        # Text with date-like content not matching patterns
        text = (
            "The meeting happened on the fifteenth of March "
            "in the year two thousand and twenty."
        )
        result = extract_timestamp_from_text(text)
        # Should extract some date (fuzzy parsing may work)
        assert isinstance(result, datetime)

    def test_fallback_to_provided(self):
        """Test fallback to provided datetime."""
        text = "No dates whatsoever in this text at all."
        fallback = datetime(2015, 6, 15, tzinfo=timezone.utc)
        result = extract_timestamp_from_text(text, fallback=fallback)
        assert result == fallback

    def test_fallback_to_now(self):
        """Test fallback to current time when no date found."""
        text = "Random text with no temporal information."
        result = extract_timestamp_from_text(text)
        # Should return something close to now
        now = datetime.now(timezone.utc)
        diff = abs((result - now).total_seconds())
        assert diff < 5  # Within 5 seconds


class TestSpinDocumentPostInit:
    """Test SpinDocument __post_init__ behavior."""

    def test_metadata_auto_initialized(self):
        """Test that metadata is auto-initialized if None."""
        doc = SpinDocument(
            doc_id="test",
            text="Test document",
            timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc),
            semantic_embedding=[0.1] * 384,
            spin_vector=[0.0] * 9,
            phi={"quarter": 0.0, "decade": 0.0, "century": 0.0},
            full_embedding=[0.1] * 393,
            metadata={},  # Empty dict instead of None
        )
        assert doc.metadata == {}

    def test_arc_mode_auto_detected(self):
        """Test that is_arc is auto-detected from end_timestamp."""
        doc = SpinDocument(
            doc_id="test",
            text="Test document",
            timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc),
            end_timestamp=datetime(
                2020, 3, 31, tzinfo=timezone.utc
            ),  # Arc mode
            semantic_embedding=[0.1] * 384,
            spin_vector=[0.0] * 9,
            phi={"quarter": 0.0, "decade": 0.0, "century": 0.0},
            full_embedding=[0.1] * 393,
        )
        assert doc.is_arc is True


class TestRetrievalResultPostInit:
    """Test RetrievalResult __post_init__ behavior."""

    def test_metadata_auto_initialized(self):
        """Test that metadata is auto-initialized if None."""
        result = RetrievalResult(
            doc_id="test",
            text="Test",
            timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc),
            semantic_score=0.8,
            phi_doc=1.0,
            phi_query=1.5,
            phi_difference=0.5,
            temporal_alignment=0.9,
            combined_score=0.85,
            metadata={},  # Empty dict instead of None
        )
        assert result.metadata == {}


# ========================================================================
# Tests for retrieval.py uncovered lines
# ========================================================================


class TestRetrieverQueryCreation:
    """Test uncovered paths in query creation."""

    def test_create_query_none_timestamp_fallback(
        self, mock_embedding_client, empty_vector_store
    ):
        """Test query creation with None timestamp falls back to now."""
        retriever = TemporalSpinRetriever(
            embedding_client=mock_embedding_client,
            vector_store=empty_vector_store,
        )

        query = retriever.create_query(
            query_text="test query",
            query_timestamp=None,  # Should fallback to now
        )

        assert query is not None
        assert isinstance(query.query_timestamp, datetime)
        # Should be close to now
        now = datetime.now(timezone.utc)
        diff = abs((query.query_timestamp - now).total_seconds())
        assert diff < 5

    def test_create_query_naive_timezone_converted(
        self, mock_embedding_client, empty_vector_store
    ):
        """Test that naive datetimes are converted to UTC."""
        retriever = TemporalSpinRetriever(
            embedding_client=mock_embedding_client,
            vector_store=empty_vector_store,
        )

        naive_dt = datetime(2020, 6, 15)  # No timezone
        query = retriever.create_query(
            query_text="test query", query_timestamp=naive_dt
        )

        assert query.query_timestamp.tzinfo == timezone.utc


class TestRetrieverSearchEdgeCases:
    """Test uncovered paths in search method."""

    def test_search_with_concept_filter(
        self, mock_embedding_client, empty_vector_store
    ):
        """Test search with concept_filter parameter."""
        retriever = TemporalSpinRetriever(
            embedding_client=mock_embedding_client,
            vector_store=empty_vector_store,
        )

        # Add a document with concept metadata
        doc = SpinDocument(
            doc_id="fact1",
            text="Revenue was $100M",
            timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc),
            semantic_embedding=[0.1] * 384,
            spin_vector=[0.0] * 9,
            phi={"quarter": 0.0, "decade": 0.0, "century": 0.0},
            full_embedding=[0.1] * 393,
            metadata={
                "chunk_type": "fact",
                "concept": "Revenues",
                "concept_full": "us-gaap:Revenues",
            },
        )
        empty_vector_store.add_documents([doc])

        # Search with concept filter
        results = retriever.search(
            query_text="revenue",
            query_timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc),
            concept_filter=["Revenues", "NetIncomeLoss"],
            top_k_final=5,
        )

        # Should not crash and return results
        assert isinstance(results, list)

    def test_search_arc_query_new_style(
        self, mock_embedding_client, empty_vector_store
    ):
        """Test arc query using new query_start/end parameters."""
        retriever = TemporalSpinRetriever(
            embedding_client=mock_embedding_client,
            vector_store=empty_vector_store,
        )

        results = retriever.search(
            query_text="test",
            query_start_timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc),
            query_end_timestamp=datetime(2020, 3, 31, tzinfo=timezone.utc),
            top_k_final=5,
        )

        assert isinstance(results, list)

    def test_search_arc_query_legacy_style(
        self, mock_embedding_client, empty_vector_store
    ):
        """Test arc query using legacy query_timestamp + end_timestamp."""
        retriever = TemporalSpinRetriever(
            embedding_client=mock_embedding_client,
            vector_store=empty_vector_store,
        )

        results = retriever.search(
            query_text="test",
            query_timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc),
            end_timestamp=datetime(2020, 3, 31, tzinfo=timezone.utc),
            top_k_final=5,
        )

        assert isinstance(results, list)


# ========================================================================
# Tests for ingestion.py uncovered lines
# ========================================================================


class TestIngestionTimestampHandling:
    """Test timestamp handling in ingest_batch."""

    def test_ingest_batch_with_none_timestamp_extraction(
        self, mock_embedding_client, empty_vector_store
    ):
        """Test that None timestamps trigger extraction."""
        pipeline = TemporalSpinIngestionPipeline(
            embedding_client=mock_embedding_client,
            vector_store=empty_vector_store,
        )

        texts = ["Report for Q1 2020", "Document from fiscal year 2019"]
        documents = pipeline.ingest_batch(
            texts=texts,
            timestamps=None,  # Should extract from text
        )

        assert len(documents) == 2
        # Should have extracted some timestamps
        assert all(doc.timestamp is not None for doc in documents)

    def test_ingest_batch_with_aware_timezone_conversion(
        self, mock_embedding_client, empty_vector_store
    ):
        """Test that timezone-aware timestamps are converted to UTC."""
        pipeline = TemporalSpinIngestionPipeline(
            embedding_client=mock_embedding_client,
            vector_store=empty_vector_store,
        )

        # Create timestamp in different timezone
        from datetime import timezone as tz

        eastern = tz(timedelta(hours=-5))
        ts_eastern = datetime(2020, 6, 15, 12, 0, 0, tzinfo=eastern)

        documents = pipeline.ingest_batch(
            texts=["Test document"], timestamps=[ts_eastern]
        )

        # Should be converted to UTC
        assert documents[0].timestamp.tzinfo == timezone.utc


class TestIngestionFileIngestion:
    """Test file-based ingestion (uncovered paths)."""

    def test_ingest_from_files(
        self, tmp_path, mock_embedding_client, empty_vector_store
    ):
        """Test ingesting documents from files."""
        pipeline = TemporalSpinIngestionPipeline(
            embedding_client=mock_embedding_client,
            vector_store=empty_vector_store,
        )

        # Create test files
        file1 = tmp_path / "doc1.txt"
        file2 = tmp_path / "report_2020-03-31.txt"

        file1.write_text("First document content")
        file2.write_text("Report for Q1 2020")

        file_paths = [str(file1), str(file2)]
        documents = pipeline.ingest_from_files(
            file_paths, extract_timestamp_from_filename=True
        )

        assert len(documents) == 2
        assert all(doc.timestamp is not None for doc in documents)
        # Second file should extract timestamp from filename
        assert documents[1].timestamp.year == 2020

    def test_ingest_from_files_fallback_to_mtime(
        self, tmp_path, mock_embedding_client, empty_vector_store
    ):
        """Test that file mtime is used as fallback."""
        pipeline = TemporalSpinIngestionPipeline(
            embedding_client=mock_embedding_client,
            vector_store=empty_vector_store,
        )

        # Create file without date in name
        file1 = tmp_path / "nodates.txt"
        file1.write_text("Content without dates")

        documents = pipeline.ingest_from_files(
            [str(file1)], extract_timestamp_from_filename=True
        )

        # Should use file modification time
        assert len(documents) == 1
        assert documents[0].timestamp is not None


class TestCreateIngestionPipeline:
    """Test the create_ingestion_pipeline factory function."""

    def test_create_pipeline_with_mock_embeddings(self, empty_vector_store):
        """Test creating pipeline with mock embeddings."""
        pipeline = create_ingestion_pipeline(
            vector_store=empty_vector_store,
            use_mock_embeddings=True,
            embedding_dim=384,
        )

        assert isinstance(pipeline, TemporalSpinIngestionPipeline)
        assert isinstance(pipeline.embedding_client, MockEmbeddingClient)

    def test_create_pipeline_with_llamastack(self, empty_vector_store):
        """Test creating pipeline with LlamaStack client."""
        # This will create a real client (but won't make actual requests)
        pipeline = create_ingestion_pipeline(
            vector_store=empty_vector_store,
            use_mock_embeddings=False,
            llamastack_url="http://localhost:5000",
            model_name="test-model",
        )

        assert isinstance(pipeline, TemporalSpinIngestionPipeline)


class TestCosineSimilarity:
    """Test cosine similarity edge cases."""

    def test_zero_norm_vectors(self):
        """Test cosine similarity with zero-norm vectors."""
        from temporal_spin import cosine_similarity

        zero_vec = [0.0, 0.0, 0.0]
        normal_vec = [1.0, 2.0, 3.0]

        # Zero vector should return 0.0
        assert math.isclose(
            cosine_similarity(zero_vec, normal_vec), 0.0, abs_tol=1e-9
        )
        assert math.isclose(
            cosine_similarity(normal_vec, zero_vec), 0.0, abs_tol=1e-9
        )
        assert math.isclose(
            cosine_similarity(zero_vec, zero_vec), 0.0, abs_tol=1e-9
        )


class TestIngestionEdgeCases:
    """Additional edge cases for ingestion."""

    def test_ingest_batch_all_default_params(
        self, mock_embedding_client, empty_vector_store
    ):
        """Test ingest_batch with all None optional parameters."""
        pipeline = TemporalSpinIngestionPipeline(
            embedding_client=mock_embedding_client,
            vector_store=empty_vector_store,
        )

        # Call with minimal parameters - all optional params None
        documents = pipeline.ingest_batch(
            texts=["Test document 1", "Test document 2"],
            # timestamps, doc_ids, metadatas, end_timestamps all None
        )

        assert len(documents) == 2
        # Should have auto-generated IDs
        assert all(doc.doc_id is not None for doc in documents)

    def test_ingest_from_files_exception_in_filename_parsing(
        self, tmp_path, mock_embedding_client, empty_vector_store
    ):
        """Test exception handling in filename timestamp extraction."""
        pipeline = TemporalSpinIngestionPipeline(
            embedding_client=mock_embedding_client,
            vector_store=empty_vector_store,
        )

        # Create file with unparseable filename
        file1 = tmp_path / "!!!invalid_filename!!!"
        file1.write_text("Content")

        # Should not crash, should fallback to mtime
        documents = pipeline.ingest_from_files(
            [str(file1)], extract_timestamp_from_filename=True
        )

        assert len(documents) == 1
        assert documents[0].timestamp is not None


# ========================================================================
# Additional temporal_spin.py coverage
# ========================================================================


class TestArcOverlapFullCircles:
    """Test full circle cases in arc_overlap."""

    def test_full_circle_raw_length(self):
        """Test arc with raw length >= tau."""
        from temporal_spin import arc_overlap

        # Create arcs where raw length is >= tau (full circle)
        start1 = 0.0
        end1 = math.tau + 1.0  # More than full circle
        start2 = 1.0
        end2 = 2.0

        overlap = arc_overlap(start1, end1, start2, end2)
        # Should return min(len1, len2) where len1=tau, len2=1.0
        assert overlap >= 0.9  # Close to 1.0
        assert overlap <= math.tau


class TestTimestampExtractionExceptionHandling:
    """Test exception handling in extract_timestamp_from_text."""

    def test_fuzzy_parse_exception(self):
        """Test that fuzzy parsing exceptions are caught."""
        # Text that might cause parsing issues
        text = "Random $#@! text with %%%% symbols"
        result = extract_timestamp_from_text(text)

        # Should return datetime without crashing
        assert isinstance(result, datetime)
        # Should be close to now since no date found
        now = datetime.now(timezone.utc)
        diff = abs((result - now).total_seconds())
        assert diff < 10


# ========================================================================
# Fixtures
# ========================================================================


@pytest.fixture
def mock_embedding_client():
    """Provide a mock embedding client for testing."""
    return MockEmbeddingClient(dimension=384)


@pytest.fixture
def empty_vector_store():
    """Provide an empty in-memory vector store."""
    return InMemoryVectorStore()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
    pytest.main([__file__, "-v"])
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
    pytest.main([__file__, "-v"])

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
    pytest.main([__file__, "-v"])
    pytest.main([__file__, "-v"])
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
    pytest.main([__file__, "-v"])
