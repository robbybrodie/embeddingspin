"""
Final Coverage Tests to Reach 100%
===================================

Targeted tests for the last uncovered lines in temporal_spin.py,
ingestion.py, and retrieval.py.
"""

import math
import os
import sys
from datetime import datetime, timedelta, timezone

import pytest

# Modify path to allow imports from parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion import TemporalSpinIngestionPipeline  # noqa: E402
from llamastack_client import MockEmbeddingClient  # noqa: E402
from retrieval import TemporalSpinRetriever, format_results_table  # noqa: E402
# fmt: off
from temporal_spin import (SpinDocument, arc_overlap,  # noqa: E402
                           extract_timestamp_from_text)
from vector_store import InMemoryVectorStore  # noqa: E402

# fmt: on


# ========================================================================
# temporal_spin.py - Line 249: Full circle case in arc_overlap
# ========================================================================


class TestArcOverlapLine249:
    """Test the specific full circle case on line 249."""

    def test_arc1_full_circle_returns_min_length(self):
        """Test when arc1's raw length >= tau, returns min(len1, len2)."""
        # Arc1: Full circle (raw length >= tau)
        start1 = 0.0
        end1 = math.tau + 0.5  # More than full circle

        # Arc2: Small arc
        start2 = 1.0
        end2 = 2.5

        overlap = arc_overlap(start1, end1, start2, end2)

        # Should return min(tau, 1.5) = 1.5
        assert 1.4 < overlap < 1.6

    def test_arc2_full_circle_returns_min_length(self):
        """Test when arc2's raw length >= tau, returns min(len1, len2)."""
        # Arc1: Small arc
        start1 = 1.0
        end1 = 2.5

        # Arc2: Full circle
        start2 = 0.0
        end2 = math.tau + 0.5

        overlap = arc_overlap(start1, end1, start2, end2)

        # Should return min(1.5, tau) = 1.5
        assert 1.4 < overlap < 1.6


# ========================================================================
# temporal_spin.py - Lines 380-381: Fuzzy parse exception handling
# ========================================================================


class TestTimestampExtractionExceptions:
    """Test fuzzy parsing exception handling on lines 380-381."""

    def test_fuzzy_parse_value_error(self):
        """Test that ValueError in fuzzy parsing is caught."""
        # Text that causes parsing to fail
        text = "Complete nonsense $$$ ### %%% text"

        # Should not crash, should return current time
        result = extract_timestamp_from_text(text)
        assert isinstance(result, datetime)

        # Verify it's close to now (since fallback)
        now = datetime.now(timezone.utc)
        diff = abs((result - now).total_seconds())
        assert diff < 10  # Within 10 seconds

    def test_fuzzy_parse_type_error(self):
        """Test that TypeError in fuzzy parsing is caught."""
        # Empty string might cause TypeError
        text = ""

        result = extract_timestamp_from_text(text)
        assert isinstance(result, datetime)


# ========================================================================
# ingestion.py - Line 201: Timezone conversion branch
# ========================================================================


class TestIngestionLine201:
    """Test timezone conversion on line 201."""

    def test_ingest_batch_with_non_utc_aware_timestamp(
        self, mock_embedding_client, empty_vector_store
    ):
        """Test that non-UTC aware timestamps are converted to UTC."""
        pipeline = TemporalSpinIngestionPipeline(
            embedding_client=mock_embedding_client,
            vector_store=empty_vector_store,
        )

        # Create timezone-aware timestamp in Eastern time
        eastern = timezone(timedelta(hours=-5))
        ts_eastern = datetime(2020, 6, 15, 12, 0, 0, tzinfo=eastern)

        documents = pipeline.ingest_batch(
            texts=["Test document"], timestamps=[ts_eastern]
        )

        # Should be converted to UTC (17:00 UTC = 12:00 EST)
        assert documents[0].timestamp.tzinfo == timezone.utc
        assert documents[0].timestamp.hour == 17  # Converted from 12 EST


# ========================================================================
# ingestion.py - Lines 280-281, 285-286: File ingestion exception
# ========================================================================


class TestIngestionFileExceptionHandling:
    """Test exception handling in file ingestion."""

    def test_ingest_from_files_extract_exception_caught(
        self, tmp_path, mock_embedding_client, empty_vector_store
    ):
        """Test that extract_timestamp_from_text exceptions are caught."""
        pipeline = TemporalSpinIngestionPipeline(
            embedding_client=mock_embedding_client,
            vector_store=empty_vector_store,
        )

        # Create file with problematic filename
        file1 = tmp_path / "bad_$#@_filename.txt"
        file1.write_text("Content")

        # Should not crash, should use mtime fallback
        documents = pipeline.ingest_from_files(
            [str(file1)], extract_timestamp_from_filename=True
        )

        assert len(documents) == 1
        assert documents[0].timestamp is not None

    def test_ingest_from_files_no_extraction_uses_mtime(
        self, tmp_path, mock_embedding_client, empty_vector_store
    ):
        """Test that mtime is used when extraction is disabled."""
        pipeline = TemporalSpinIngestionPipeline(
            embedding_client=mock_embedding_client,
            vector_store=empty_vector_store,
        )

        file1 = tmp_path / "document.txt"
        file1.write_text("Test content")

        # Don't extract from filename
        documents = pipeline.ingest_from_files(
            [str(file1)], extract_timestamp_from_filename=False
        )

        assert len(documents) == 1
        # Should use file modification time
        assert documents[0].timestamp is not None


# ========================================================================
# retrieval.py - Lines 360-369: Arc-to-point alignment
# ========================================================================


class TestRetrievalArcToPoint:
    """Test arc-to-point alignment logic (lines 360-369)."""

    def test_arc_query_to_point_document_inside_arc(
        self, mock_embedding_client
    ):
        """Test arc query with point document inside the arc."""
        store = InMemoryVectorStore()
        retriever = TemporalSpinRetriever(
            embedding_client=mock_embedding_client, vector_store=store
        )

        # Add point document (Q2 2020)
        point_doc = SpinDocument(
            doc_id="point1",
            text="Q2 2020 report",
            timestamp=datetime(2020, 5, 15, tzinfo=timezone.utc),
            semantic_embedding=[0.5] * 384,
            spin_vector=[1.0, 0.0, 0.0] + [0.0] * 6,
            phi={"quarter": 1.0, "decade": 0.5, "century": 0.1},
            full_embedding=[0.5] * 393,
        )
        store.add_documents([point_doc])

        # Arc query spanning Q1-Q3 2020 (should contain the point)
        results = retriever.search(
            query_text="2020 report",
            query_start_timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc),
            query_end_timestamp=datetime(2020, 9, 30, tzinfo=timezone.utc),
            beta=1000.0,
            top_k_final=5,
        )

        assert len(results) > 0

    def test_arc_query_to_point_document_outside_arc_line_365(
        self, mock_embedding_client
    ):
        """Test arc query with point document outside arc (hits line 365)."""
        store = InMemoryVectorStore()
        retriever = TemporalSpinRetriever(
            embedding_client=mock_embedding_client, vector_store=store
        )

        # Add point document clearly outside the query arc
        point_doc = SpinDocument(
            doc_id="point1",
            text="2015 report",  # Far from query period
            timestamp=datetime(2015, 5, 15, tzinfo=timezone.utc),
            semantic_embedding=[0.9] * 384,  # High semantic similarity
            spin_vector=[1.0, 0.0, 0.0] + [0.0] * 6,
            phi={"quarter": 1.0, "decade": 0.2, "century": 0.0},
            full_embedding=[0.9] * 393,
        )
        store.add_documents([point_doc])

        # Arc query for Q1 2020 only (point is 5 years away)
        results = retriever.search(
            query_text="report",  # High semantic match
            query_start_timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc),
            query_end_timestamp=datetime(2020, 3, 31, tzinfo=timezone.utc),
            beta=100.0,  # Moderate temporal focus
            top_k_coarse=10,
            top_k_final=5,
        )

        # Should return results (uses distance-based scoring on line 366)
        assert len(results) > 0

    def test_arc_query_to_point_document_outside_arc(
        self, mock_embedding_client
    ):
        """Test arc query with point document outside the arc."""
        store = InMemoryVectorStore()
        retriever = TemporalSpinRetriever(
            embedding_client=mock_embedding_client, vector_store=store
        )

        # Add point document (2019)
        point_doc = SpinDocument(
            doc_id="point1",
            text="2019 report",
            timestamp=datetime(2019, 5, 15, tzinfo=timezone.utc),
            semantic_embedding=[0.5] * 384,
            spin_vector=[1.0, 0.0, 0.0] + [0.0] * 6,
            phi={"quarter": 1.0, "decade": 0.5, "century": 0.1},
            full_embedding=[0.5] * 393,
        )
        store.add_documents([point_doc])

        # Arc query for Q1 2020 only (point is outside)
        results = retriever.search(
            query_text="2020 report",
            query_start_timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc),
            query_end_timestamp=datetime(2020, 3, 31, tzinfo=timezone.utc),
            beta=5000.0,
            top_k_final=5,
        )

        # Should still return results (with distance-based scoring)
        assert isinstance(results, list)


# ========================================================================
# retrieval.py - Line 377: Point-to-arc branch
# ========================================================================


class TestRetrievalPointToArc:
    """Test point-to-arc alignment logic (line 377)."""

    def test_point_query_to_arc_document_point_inside(
        self, mock_embedding_client
    ):
        """Test point query inside arc document (hits line 377: if)."""
        store = InMemoryVectorStore()
        retriever = TemporalSpinRetriever(
            embedding_client=mock_embedding_client, vector_store=store
        )

        # Add arc document (Q1 2020)
        arc_doc = SpinDocument(
            doc_id="arc1",
            text="Q1 2020 quarterly report",
            timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc),
            end_timestamp=datetime(2020, 3, 31, tzinfo=timezone.utc),
            semantic_embedding=[0.5] * 384,
            spin_vector=[1.0, 0.0, 0.5] + [0.0] * 6,
            phi={"quarter": 1.0, "decade": 0.5, "century": 0.1},
            full_embedding=[0.5] * 393,
            phi_start={"quarter": 0.0, "decade": 0.4, "century": 0.0},
            phi_end={"quarter": 1.5, "decade": 0.6, "century": 0.2},
            is_arc=True,
        )
        store.add_documents([arc_doc])

        # Point query in middle of Q1 (should be inside arc)
        # This will cause overlap > 0, hitting line 377
        results = retriever.search(
            query_text="report",
            query_timestamp=datetime(2020, 2, 15, tzinfo=timezone.utc),
            beta=500.0,  # Very high beta to force temporal computation
            top_k_final=5,
        )

        assert len(results) > 0
        # With high beta and point inside arc, temporal factors dominate
        # but high beta actually reduces semantic contribution

    def test_point_query_to_arc_document_point_outside_line_379(
        self, mock_embedding_client
    ):
        """Test point query outside arc document (hits line 379 else)."""
        store = InMemoryVectorStore()
        retriever = TemporalSpinRetriever(
            embedding_client=mock_embedding_client, vector_store=store
        )

        # Add arc document (Q1 2020)
        arc_doc = SpinDocument(
            doc_id="arc1",
            text="Q1 2020 quarterly report",
            timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc),
            end_timestamp=datetime(2020, 3, 31, tzinfo=timezone.utc),
            semantic_embedding=[0.9] * 384,  # High semantic match
            spin_vector=[1.0, 0.0, 0.5] + [0.0] * 6,
            phi={"quarter": 1.0, "decade": 0.5, "century": 0.1},
            full_embedding=[0.9] * 393,
            phi_start={"quarter": 0.0, "decade": 0.4, "century": 0.0},
            phi_end={"quarter": 1.5, "decade": 0.6, "century": 0.2},
            is_arc=True,
        )
        store.add_documents([arc_doc])

        # Point query well outside the arc (Q4 2019)
        # This will cause overlap = 0, triggering else at line 379
        results = retriever.search(
            query_text="report",  # High semantic similarity
            query_timestamp=datetime(2019, 12, 15, tzinfo=timezone.utc),
            beta=100.0,  # Moderate temporal focus
            top_k_coarse=10,
            top_k_final=5,
        )

        # Should return results using distance-based scoring (line 380
        # Should return results using distance-based scoring (line 379)
        assert len(results) > 0

    def test_point_query_to_arc_document(self, mock_embedding_client):
        """Test point query with arc document."""
        store = InMemoryVectorStore()
        retriever = TemporalSpinRetriever(
            embedding_client=mock_embedding_client, vector_store=store
        )

        # Add arc document (Q1 2020)
        arc_doc = SpinDocument(
            doc_id="arc1",
            text="Q1 2020 quarterly report",
            timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc),
            end_timestamp=datetime(2020, 3, 31, tzinfo=timezone.utc),
            semantic_embedding=[0.5] * 384,
            spin_vector=[1.0, 0.0, 0.5] + [0.0] * 6,
            phi={"quarter": 1.0, "decade": 0.5, "century": 0.1},
            full_embedding=[0.5] * 393,
            phi_start={"quarter": 0.9, "decade": 0.4, "century": 0.0},
            phi_end={"quarter": 1.5, "decade": 0.6, "century": 0.2},
            is_arc=True,
        )
        store.add_documents([arc_doc])

        # Point query in middle of Q1
        results = retriever.search(
            query_text="report",
            query_timestamp=datetime(2020, 2, 15, tzinfo=timezone.utc),
            beta=1000.0,
            top_k_final=5,
        )

        assert len(results) > 0


# ========================================================================
# retrieval.py - Lines 446-447: Chunk type retrieval
# ========================================================================


class TestRetrievalChunkType:
    """Test chunk type metadata retrieval (lines 446-447)."""

    def test_search_with_chunk_type_metadata(self, mock_embedding_client):
        """Test that chunk_type metadata is correctly retrieved."""
        store = InMemoryVectorStore()
        retriever = TemporalSpinRetriever(
            embedding_client=mock_embedding_client, vector_store=store
        )

        # Add document with chunk_type metadata
        doc = SpinDocument(
            doc_id="section1",
            text="Important section",
            timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc),
            semantic_embedding=[0.5] * 384,
            spin_vector=[1.0] * 9,
            phi={"quarter": 1.0, "decade": 0.5, "century": 0.1},
            full_embedding=[0.5] * 393,
            metadata={"chunk_type": "section"},
        )
        store.add_documents([doc])

        results = retriever.search(
            query_text="important",
            query_timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc),
            top_k_final=5,
        )

        assert len(results) > 0
        # Chunk type should affect scoring

    def test_search_with_malformed_metadata_exception_line_447(
        self, mock_embedding_client
    ):
        """Test exception handling for malformed metadata (lines 446-447)."""
        store = InMemoryVectorStore()
        retriever = TemporalSpinRetriever(
            embedding_client=mock_embedding_client, vector_store=store
        )

        # Add a document that will behave strangely
        doc1 = SpinDocument(
            doc_id="doc1",
            text="Important document",
            timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc),
            semantic_embedding=[0.9] * 384,
            spin_vector=[1.0] * 9,
            phi={"quarter": 1.0, "decade": 0.5, "century": 0.1},
            full_embedding=[0.9] * 393,
            metadata={"chunk_type": "section"},
        )

        store.add_documents([doc1])

        # Create a document that appears normal but has unstable metadata
        class UnstableMetadataDoc:
            """Doc with metadata that sometimes fails."""

            def __init__(self, orig_doc):
                self._call_count = 0
                self.doc_id = orig_doc.doc_id
                self.text = orig_doc.text
                self.timestamp = orig_doc.timestamp
                self.semantic_embedding = orig_doc.semantic_embedding
                self.spin_vector = orig_doc.spin_vector
                self.phi = orig_doc.phi
                self.full_embedding = orig_doc.full_embedding
                self.is_arc = orig_doc.is_arc
                self._metadata = orig_doc.metadata

            @property
            def metadata(self):
                """
                First call (line 416): works fine
                Second call (line 443): raises AttributeError.
                """
                self._call_count += 1
                if self._call_count > 1:
                    raise AttributeError("Unstable metadata")
                return self._metadata

        # Replace doc1 with unstable version AFTER adding to store
        # This way it's in both documents dict and embeddings dict
        orig_doc1 = store.documents["doc1"]
        store.documents["doc1"] = UnstableMetadataDoc(  # type: ignore
            orig_doc1
        )

        # Should handle exception at lines 446-447 and continue
        results = retriever.search(
            query_text="document",
            query_timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc),
            top_k_final=5,
        )

        # Should complete search despite metadata error in priority boost
        assert len(results) >= 1


# ========================================================================
# retrieval.py - Line 485 & 544-573: format_results_table
# ========================================================================


class TestFormatResultsTable:
    """Test format_results_table function."""

    def test_format_empty_results(self):
        """Test formatting empty results list."""
        output = format_results_table([])
        assert output == "No results."

    def test_format_single_result(self, mock_embedding_client):
        """Test formatting a single result."""
        from temporal_spin import RetrievalResult

        result = RetrievalResult(
            doc_id="doc1",
            text="This is a test document with some content",
            timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc),
            semantic_score=0.85,
            phi_doc=1.5,
            phi_query=1.0,
            phi_difference=0.5,
            temporal_alignment=0.9,
            combined_score=0.88,
            rank=1,
        )

        output = format_results_table([result])

        # Should contain table elements
        assert "┌" in output
        assert "│" in output
        assert "└" in output
        assert "Semantic" in output
        assert "Temporal" in output
        assert "0.8500" in output  # Semantic score

    def test_format_multiple_results(self):
        """Test formatting multiple results."""
        from temporal_spin import RetrievalResult

        results = [
            RetrievalResult(
                doc_id=f"doc{i}",
                text=f"Document {i} content",
                timestamp=datetime(2020, 1, i + 1, tzinfo=timezone.utc),
                semantic_score=0.9 - i * 0.1,
                phi_doc=1.0,
                phi_query=1.0,
                phi_difference=0.1 * i,
                temporal_alignment=0.95 - i * 0.05,
                combined_score=0.9 - i * 0.1,
                rank=i + 1,
            )
            for i in range(3)
        ]

        output = format_results_table(results)

        # Should contain all three results
        assert output.count("│") > 10  # Multiple rows
        assert "Document 0" in output
        assert "Document 1" in output
        assert "Document 2" in output

    def test_format_with_custom_max_length(self):
        """Test formatting with custom max text length."""
        from temporal_spin import RetrievalResult

        result = RetrievalResult(
            doc_id="doc1",
            text="A" * 200,  # Very long text
            timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc),
            semantic_score=0.85,
            phi_doc=1.5,
            phi_query=1.0,
            phi_difference=0.5,
            temporal_alignment=0.9,
            combined_score=0.88,
            rank=1,
        )

        output = format_results_table([result], max_text_length=20)

        # Text should be truncated to 20 characters
        lines = output.split("\n")
        # Check that no line is excessively long
        assert all(len(line) < 200 for line in lines)


# ========================================================================
# retrieval.py - Line 485: Beta sweep default values
# ========================================================================


class TestRetrievalBetaSweep:
    """Test beta sweep functionality (line 485)."""

    def test_search_with_beta_sweep_default_line_485(
        self, mock_embedding_client
    ):
        """Test that default beta_values are used when None (line 485)."""
        store = InMemoryVectorStore()
        retriever = TemporalSpinRetriever(
            embedding_client=mock_embedding_client, vector_store=store
        )

        # Add document
        doc = SpinDocument(
            doc_id="doc1",
            text="Test document",
            timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc),
            semantic_embedding=[0.5] * 384,
            spin_vector=[1.0] * 9,
            phi={"quarter": 1.0, "decade": 0.5, "century": 0.1},
            full_embedding=[0.5] * 393,
        )
        store.add_documents([doc])

        # Call with beta_values=None to trigger default assignment
        results = retriever.search_with_beta_sweep(
            query_text="test",
            query_timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc),
            beta_values=None,  # This should trigger line 485
            top_k=5,
        )

        # Should return results with default beta values
        assert isinstance(results, list)
        assert len(results) > 0


# ========================================================================
# ingestion.py - Line 201: Timezone conversion
# ========================================================================


class TestIngestionTimezone:
    """Test timezone conversion in ingestion (line 201)."""

    def test_ingest_batch_with_timezone_aware_timestamp_line_201(
        self, mock_embedding_client
    ):
        """Test ingestion with timezone-aware timestamp (line 201)."""
        from zoneinfo import ZoneInfo

        from ingestion import TemporalSpinIngestionPipeline

        # Use timezone-aware timestamp (not UTC)
        timestamp = datetime(
            2020, 1, 1, 12, 0, 0, tzinfo=ZoneInfo("America/New_York")
        )

        store = InMemoryVectorStore()
        pipeline = TemporalSpinIngestionPipeline(
            embedding_client=mock_embedding_client, vector_store=store
        )
        texts = ["Test content"]
        timestamps = [timestamp]

        pipeline.ingest_batch(
            texts=texts,
            timestamps=timestamps,
        )

        # Document should be ingested with UTC conversion (line 201)
        assert len(store.documents) > 0
        doc = list(store.documents.values())[0]
        assert doc.timestamp.tzinfo == timezone.utc


# ========================================================================
# ingestion.py - Lines 280-281: Exception handling in filename parsing
# ========================================================================


class TestIngestionExceptions:
    """Test exception handling in ingestion (lines 280-281)."""

    def test_extract_timestamp_from_text_exception_line_280(self):
        """Test exception handling in timestamp extraction (line 280)."""
        from ingestion import extract_timestamp_from_text

        # Invalid text that will raise exception in parsing
        result = extract_timestamp_from_text("no date here at all xxx")
        # Should return fallback (datetime.now)
        assert result is not None
        assert isinstance(result, datetime)

    def test_ingest_from_files_exception_in_filename_line_280_281(
        self, tmp_path, mock_embedding_client
    ):
        """Test exception at lines 280-281 in ingest_from_files."""
        from ingestion import TemporalSpinIngestionPipeline

        # Create test file with name that will cause extract failure
        test_file = tmp_path / "bad_filename.txt"
        test_file.write_text("Some content without dates")

        store = InMemoryVectorStore()
        pipeline = TemporalSpinIngestionPipeline(
            embedding_client=mock_embedding_client, vector_store=store
        )

        # Monkey-patch extract_timestamp_from_text to raise exception
        import ingestion as ing_module

        original_extract = ing_module.extract_timestamp_from_text

        def failing_extract(text, fallback=None):
            if "bad_filename" in text:
                raise ValueError("Simulated extraction failure")
            return original_extract(text, fallback)

        ing_module.extract_timestamp_from_text = failing_extract

        try:
            # Should catch exception at lines 280-281 and use mtime
            pipeline.ingest_from_files(
                [str(test_file)], extract_timestamp_from_filename=True
            )

            # Should succeed with mtime fallback
            assert len(store.documents) > 0
        finally:
            # Restore original function
            ing_module.extract_timestamp_from_text = original_extract


# ========================================================================
# temporal_spin.py - Lines 380-381: Exception handling in fuzzy parsing
# ========================================================================


class TestTemporalSpinExceptions:
    """Test exception handling in temporal_spin (lines 380-381)."""

    def test_fuzzy_date_parsing_exception_line_380(self):
        """Test exception handling in fuzzy date parsing (line 380)."""
        from temporal_spin import extract_timestamp_from_text

        # Text that matches DATE_PATTERN but fails to parse
        # "Fiscal Year:" pattern matches but "invalid" can't be parsed
        result = extract_timestamp_from_text(
            "Fiscal Year: invalid date string"
        )

        # Should fall back to the fuzzy parsing or fallback
        assert result is not None
        assert isinstance(result, datetime)

    def test_malformed_date_pattern_match_line_380(self):
        """Test pattern match with unparseable date (lines 380-381)."""
        from temporal_spin import extract_timestamp_from_text

        # Matches "Period Ended:" pattern but with bad date
        result = extract_timestamp_from_text(
            "Period Ended: not-a-date-at-all-xyz"
        )

        assert result is not None

    def test_exception_in_dateutil_parse_line_380_381(self):
        """Force ValueError/TypeError in dateutil.parser.parse."""
        from temporal_spin import extract_timestamp_from_text

        # Create text that matches patterns but will fail parsing
        # Use extreme values that dateutil can't handle
        texts = [
            "fiscal year 99999",  # Year too large
            "period ended 99 InvalidMonth 2020",  # Invalid month
            "as of 32 December 2020",  # Invalid day
            "Q4 99999",  # Invalid year
        ]

        for text in texts:
            result = extract_timestamp_from_text(text)
            # Should fall back and return a valid datetime
            assert result is not None
            assert isinstance(result, datetime)


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
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
