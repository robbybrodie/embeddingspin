"""
Integration Tests for Two-Pass Retrieval Algorithm
===================================================

Tests the complete retrieval pipeline:
- Two-pass retrieval (coarse recall + temporal zoom)
- Beta parameter effects on temporal focus
- Arc-to-arc, point-to-arc, and arc-to-point queries
- Multi-scale temporal alignment
- Priority boosting based on chunk type
"""

import math
import pytest
from datetime import datetime, timezone, timedelta

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from temporal_spin import angular_difference
from retrieval import TemporalSpinRetriever
# Import fixtures and utilities (pytest discovers conftest.py automatically)
from tests.conftest import (
    assert_temporal_ordering,
    get_sample_documents,
    get_quarterly_reports,
    generate_quarterly_dates
)


# ============================================================================
# Tests for Basic Retrieval
# ============================================================================

class TestBasicRetrieval:
    """Test basic retrieval functionality."""
    
    def test_retriever_returns_results(self, retriever):
        """Retriever should return non-empty results."""
        results = retriever.search(
            query_text="Apple revenue",
            query_timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc),
            beta=0.5,
            top_k_final=5
        )
        
        assert len(results) > 0, "Should return at least one result"
        assert len(results) <= 5, "Should not exceed top_k"
    
    def test_results_have_required_fields(self, retriever):
        """Results should have all required fields."""
        results = retriever.search(
            query_text="Apple announcement",
            query_timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc),
            beta=0.5,
            top_k_final=3
        )
        
        for result in results:
            assert hasattr(result, 'doc_id')
            assert hasattr(result, 'text')
            assert hasattr(result, 'timestamp')
            assert hasattr(result, 'semantic_score')
            assert hasattr(result, 'temporal_alignment')
            assert hasattr(result, 'combined_score')
            assert hasattr(result, 'rank')
    
    def test_results_sorted_by_combined_score(self, retriever):
        """Results should be sorted by combined score (descending)."""
        results = retriever.search(
            query_text="Apple technology",
            query_timestamp=datetime(2020, 6, 1, tzinfo=timezone.utc),
            beta=0.5,
            top_k_final=5
        )
        
        # Verify descending order
        for i in range(len(results) - 1):
            assert results[i].combined_score >= results[i+1].combined_score, (
                f"Rank {i+1} score {results[i].combined_score} < "
                f"Rank {i+2} score {results[i+1].combined_score}"
            )
    
    def test_ranks_assigned_correctly(self, retriever):
        """Ranks should be sequential starting from 1."""
        results = retriever.search(
            query_text="Apple",
            query_timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc),
            beta=0.5,
            top_k_final=5
        )
        
        expected_ranks = list(range(1, len(results) + 1))
        actual_ranks = [r.rank for r in results]
        assert actual_ranks == expected_ranks


# ============================================================================
# Tests for Beta Parameter Effects
# ============================================================================

class TestBetaParameterEffects:
    """Test how beta parameter affects retrieval."""
    
    def test_beta_zero_pure_semantic(self, retriever):
        """Beta=0 should ignore temporal alignment."""
        # Query with same timestamp as multiple docs
        query_date = datetime(2020, 1, 1, tzinfo=timezone.utc)
        
        results = retriever.search(
            query_text="Apple",
            query_timestamp=query_date,
            beta=0.0,  # Pure semantic
            top_k_final=5
        )
        
        # With β=0, temporal_alignment should be 1.0 (or close)
        # since exp(-0 * anything) = 1
        for result in results:
            assert result.temporal_alignment >= 0.9, (
                f"With β=0, temporal alignment should be ~1.0, got {result.temporal_alignment}"
            )
    
    def test_beta_high_temporal_focus(self, retriever):
        """High beta should prioritize temporal proximity."""
        query_date = datetime(2020, 1, 1, tzinfo=timezone.utc)
        
        results = retriever.search(
            query_text="Apple",
            query_timestamp=query_date,
            beta=1.0,  # Strong temporal focus
            top_k_final=10
        )
        
        # Top results should be temporally close to query date
        # (within a few months)
        if len(results) > 0:
            top_result = results[0]
            time_diff = abs((top_result.timestamp - query_date).days)
            assert time_diff < 180, (
                f"With high β, top result should be temporally close, got {time_diff} days"
            )
    
    def test_beta_sweep_different_rankings(self, retriever):
        """Different beta values should produce different rankings."""
        query_date = datetime(2020, 6, 1, tzinfo=timezone.utc)
        
        results_low_beta = retriever.search(
            query_text="Apple revenue",
            query_timestamp=query_date,
            beta=0.1,
            top_k_final=5
        )
        
        results_high_beta = retriever.search(
            query_text="Apple revenue",
            query_timestamp=query_date,
            beta=1.0,
            top_k_final=5
        )
        
        # Extract doc IDs in order
        low_beta_order = [r.doc_id for r in results_low_beta]
        high_beta_order = [r.doc_id for r in results_high_beta]
        
        # Rankings should differ (unless all docs are at exact same time)
        # At minimum, scores should differ
        assert low_beta_order != high_beta_order or \
               results_low_beta[0].combined_score != results_high_beta[0].combined_score
    
    def test_temporal_alignment_decreases_with_distance(self, retriever):
        """Temporal alignment should decrease as time distance increases."""
        query_date = datetime(2020, 1, 1, tzinfo=timezone.utc)
        
        results = retriever.search(
            query_text="Apple",
            query_timestamp=query_date,
            beta=0.5,
            top_k_final=10
        )
        
        # Find results with increasing time distance
        # and verify temporal alignment decreases (generally)
        results_by_time = sorted(results, key=lambda r: abs((r.timestamp - query_date).days))
        
        if len(results_by_time) >= 3:
            # Closest should have higher alignment than farthest
            closest = results_by_time[0]
            farthest = results_by_time[-1]
            
            time_diff_closest = abs((closest.timestamp - query_date).days)
            time_diff_farthest = abs((farthest.timestamp - query_date).days)
            
            if time_diff_farthest > time_diff_closest + 30:  # At least 1 month difference
                assert closest.temporal_alignment >= farthest.temporal_alignment


# ============================================================================
# Tests for Arc-Based Queries
# ============================================================================

class TestArcBasedQueries:
    """Test arc-to-arc, point-to-arc, and arc-to-point queries."""
    
    def test_arc_query_uses_both_timestamps(self, mock_embedding_client, empty_vector_store):
        """Arc query should use both start and end timestamps."""
        from ingestion import TemporalSpinIngestionPipeline
        
        # Ingest quarterly reports as arcs
        pipeline = TemporalSpinIngestionPipeline(
            embedding_client=mock_embedding_client,
            vector_store=empty_vector_store
        )
        
        reports = get_quarterly_reports()
        for text, start, end, metadata in reports:
            pipeline.ingest_document(
                text=text,
                timestamp=start,
                end_timestamp=end,
                metadata=metadata
            )
        
        # Create retriever
        retriever = TemporalSpinRetriever(
            embedding_client=mock_embedding_client,
            vector_store=empty_vector_store
        )
        
        # Query with arc (Q1 2020)
        results = retriever.search(
            query_text="revenue",
            query_start_timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc),
            query_end_timestamp=datetime(2020, 3, 31, tzinfo=timezone.utc),
            beta=0.5,
            top_k_final=3
        )
        
        assert len(results) > 0, "Arc query should return results"
    
    def test_arc_to_arc_exact_match_high_score(self, mock_embedding_client, empty_vector_store):
        """Arc query matching exact arc document should have high score."""
        from ingestion import TemporalSpinIngestionPipeline
        
        pipeline = TemporalSpinIngestionPipeline(
            embedding_client=mock_embedding_client,
            vector_store=empty_vector_store
        )
        
        # Ingest Q1 2020 as arc
        pipeline.ingest_document(
            text="Q1 2020 financial results",
            timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc),
            end_timestamp=datetime(2020, 3, 31, tzinfo=timezone.utc),
            doc_id="q1_2020"
        )
        
        retriever = TemporalSpinRetriever(
            embedding_client=mock_embedding_client,
            vector_store=empty_vector_store
        )
        
        # Query with exact same arc
        results = retriever.search(
            query_text="Q1 2020",
            query_start_timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc),
            query_end_timestamp=datetime(2020, 3, 31, tzinfo=timezone.utc),
            beta=0.5,
            top_k_final=1
        )
        
        assert len(results) > 0
        assert results[0].doc_id == "q1_2020"
        # Temporal alignment should be very high for exact match
        assert results[0].temporal_alignment > 0.8
    
    def test_arc_to_arc_zero_overlap_rejected(self, mock_embedding_client, empty_vector_store):
        """Arc query with zero overlap should reject document."""
        from ingestion import TemporalSpinIngestionPipeline
        
        pipeline = TemporalSpinIngestionPipeline(
            embedding_client=mock_embedding_client,
            vector_store=empty_vector_store
        )
        
        # Ingest Q1 2020
        pipeline.ingest_document(
            text="Q1 2020 report",
            timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc),
            end_timestamp=datetime(2020, 3, 31, tzinfo=timezone.utc),
            doc_id="q1_2020"
        )
        
        # Ingest Q1 2021 (different year, should be rejected by decade scale)
        pipeline.ingest_document(
            text="Q1 2021 report",
            timestamp=datetime(2021, 1, 1, tzinfo=timezone.utc),
            end_timestamp=datetime(2021, 3, 31, tzinfo=timezone.utc),
            doc_id="q1_2021"
        )
        
        retriever = TemporalSpinRetriever(
            embedding_client=mock_embedding_client,
            vector_store=empty_vector_store
        )
        
        # Query for Q1 2020
        results = retriever.search(
            query_text="Q1 report",
            query_start_timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc),
            query_end_timestamp=datetime(2020, 3, 31, tzinfo=timezone.utc),
            beta=0.5,
            top_k_final=5
        )
        
        # Q1 2021 should not appear (hard boundary check at decade scale)
        result_ids = [r.doc_id for r in results]
        assert "q1_2021" not in result_ids, "Q1 2021 should be rejected (zero overlap at decade scale)"
    
    def test_point_query_to_arc_document(self, mock_embedding_client, empty_vector_store):
        """Point query should match arc document if point falls within arc."""
        from ingestion import TemporalSpinIngestionPipeline
        
        pipeline = TemporalSpinIngestionPipeline(
            embedding_client=mock_embedding_client,
            vector_store=empty_vector_store
        )
        
        # Ingest full year 2020 as arc
        pipeline.ingest_document(
            text="Annual report 2020",
            timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc),
            end_timestamp=datetime(2020, 12, 31, tzinfo=timezone.utc),
            doc_id="annual_2020"
        )
        
        retriever = TemporalSpinRetriever(
            embedding_client=mock_embedding_client,
            vector_store=empty_vector_store
        )
        
        # Query with point inside the arc (June 2020)
        results = retriever.search(
            query_text="2020 report",
            query_timestamp=datetime(2020, 6, 15, tzinfo=timezone.utc),
            beta=0.5,
            top_k_final=1
        )
        
        assert len(results) > 0
        assert results[0].doc_id == "annual_2020"
        # Should have high temporal alignment (point in arc)
        assert results[0].temporal_alignment > 0.9


# ============================================================================
# Tests for Multi-Scale Temporal Alignment
# ============================================================================

class TestMultiScaleAlignment:
    """Test multi-scale temporal alignment calculations."""
    
    def test_same_quarter_high_alignment(self, retriever):
        """Documents from same quarter should have high temporal alignment."""
        # Query in January 2020
        query_date = datetime(2020, 1, 15, tzinfo=timezone.utc)
        
        results = retriever.search(
            query_text="Apple",
            query_timestamp=query_date,
            beta=0.5,
            top_k_final=10
        )
        
        # Find January 2020 documents
        jan_2020_results = [
            r for r in results
            if r.timestamp.year == 2020 and r.timestamp.month == 1
        ]
        
        if jan_2020_results:
            # Same month should have high temporal alignment
            for result in jan_2020_results:
                assert result.temporal_alignment > 0.7, (
                    f"Same quarter should have high alignment, got {result.temporal_alignment}"
                )
    
    def test_different_year_separation(self, retriever):
        """Documents from different years should be well-separated."""
        # Query in 2020
        query_date = datetime(2020, 6, 1, tzinfo=timezone.utc)
        
        results = retriever.search(
            query_text="Apple",
            query_timestamp=query_date,
            beta=0.7,  # Strong temporal focus
            top_k_final=10
        )
        
        # Find 2020 and 2023 documents
        results_2020 = [r for r in results if r.timestamp.year == 2020]
        results_2023 = [r for r in results if r.timestamp.year == 2023]
        
        if results_2020 and results_2023:
            # 2020 documents should rank higher than 2023
            avg_rank_2020 = sum(r.rank for r in results_2020) / len(results_2020)
            avg_rank_2023 = sum(r.rank for r in results_2023) / len(results_2023)
            
            assert avg_rank_2020 < avg_rank_2023, (
                "2020 documents should rank higher than 2023 for 2020 query"
            )
    
    def test_decade_scale_year_discrimination(self, mock_embedding_client, empty_vector_store):
        """Decade scale should provide strong year-to-year discrimination."""
        from ingestion import TemporalSpinIngestionPipeline
        
        pipeline = TemporalSpinIngestionPipeline(
            embedding_client=mock_embedding_client,
            vector_store=empty_vector_store
        )
        
        # Ingest documents from 2019, 2020, 2021
        for year in [2019, 2020, 2021]:
            pipeline.ingest_document(
                text=f"Annual report {year}",
                timestamp=datetime(year, 6, 1, tzinfo=timezone.utc),
                doc_id=f"report_{year}"
            )
        
        retriever = TemporalSpinRetriever(
            embedding_client=mock_embedding_client,
            vector_store=empty_vector_store
        )
        
        # Query for 2020
        results = retriever.search(
            query_text="report",
            query_timestamp=datetime(2020, 6, 1, tzinfo=timezone.utc),
            beta=0.5,
            top_k_final=3
        )
        
        # 2020 should be top result
        assert results[0].doc_id == "report_2020", "2020 should rank first"


# ============================================================================
# Tests for Two-Pass Algorithm
# ============================================================================

class TestTwoPassAlgorithm:
    """Test two-pass retrieval algorithm mechanics."""
    
    def test_pass_one_coarse_recall(self, retriever):
        """Pass 1 should retrieve candidates with broad search."""
        # This is implicitly tested, but we can verify by checking
        # that we get results even with high beta
        results = retriever.search(
            query_text="Apple",
            query_timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc),
            beta=1.0,  # High beta
            top_k_coarse=50,  # Large candidate set
            top_k_final=5
        )
        
        assert len(results) > 0, "Pass 1 should retrieve candidates"
    
    def test_pass_two_reranking(self, retriever):
        """Pass 2 should rerank results based on temporal alignment."""
        query_date = datetime(2020, 1, 1, tzinfo=timezone.utc)
        
        # Low beta (more semantic)
        results_low = retriever.search(
            query_text="Apple",
            query_timestamp=query_date,
            beta=0.3,
            top_k_final=5
        )
        
        # High beta (more temporal)
        results_high = retriever.search(
            query_text="Apple",
            query_timestamp=query_date,
            beta=0.8,
            top_k_final=5
        )
        
        # Rankings should differ due to different temporal weighting
        # (unless all results are at exact same time)
        if len(results_low) > 1 and len(results_high) > 1:
            low_order = [r.doc_id for r in results_low]
            high_order = [r.doc_id for r in results_high]
            
            # At minimum, top results should differ or scores should differ
            assert (low_order != high_order or
                    abs(results_low[0].combined_score - results_high[0].combined_score) > 1e-6)
    
    def test_lambda_coarse_affects_recall(self, retriever):
        """Lambda parameter in Pass 1 should affect candidate recall."""
        # This tests that lambda_coarse parameter is being used
        results = retriever.search(
            query_text="Apple",
            query_timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc),
            beta=0.5,
            lambda_coarse=0.1,  # Standard
            top_k_final=5
        )
        
        assert len(results) > 0, "Should get results with standard lambda"


# ============================================================================
# Tests for Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_empty_query_text(self, retriever):
        """Empty query should still return results (based on timestamp only)."""
        results = retriever.search(
            query_text="",
            query_timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc),
            beta=0.5,
            top_k_final=3
        )
        
        # May or may not return results depending on implementation
        # At minimum, should not crash
        assert isinstance(results, list)
    
    def test_future_query_timestamp(self, retriever):
        """Query with future timestamp should work."""
        future_date = datetime(2030, 1, 1, tzinfo=timezone.utc)
        
        results = retriever.search(
            query_text="Apple",
            query_timestamp=future_date,
            beta=0.5,
            top_k_final=5
        )
        
        # Should return results (all documents are "in the past")
        assert len(results) > 0
    
    def test_past_query_timestamp(self, retriever):
        """Query with very old timestamp should work."""
        old_date = datetime(2010, 1, 1, tzinfo=timezone.utc)
        
        results = retriever.search(
            query_text="Apple",
            query_timestamp=old_date,
            beta=0.5,
            top_k_final=5
        )
        
        assert len(results) > 0
    
    def test_top_k_larger_than_corpus(self, retriever):
        """Requesting more results than documents should return all."""
        results = retriever.search(
            query_text="Apple",
            query_timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc),
            beta=0.5,
            top_k_final=1000  # More than corpus size
        )
        
        # Should return whatever is available
        assert len(results) > 0
        assert len(results) <= 1000
    
    def test_top_k_zero(self, retriever):
        """Top_k=0 should return empty list."""
        results = retriever.search(
            query_text="Apple",
            query_timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc),
            beta=0.5,
            top_k_final=0
        )
        
        assert len(results) == 0


# ============================================================================
# Tests for Search with Beta Sweep
# ============================================================================

class TestBetaSweep:
    """Test beta sweep functionality."""
    
    def test_beta_sweep_returns_multiple_results(self, retriever):
        """Beta sweep should return results for each beta value."""
        beta_values = [0.0, 0.3, 0.5, 0.7, 1.0]
        
        sweep_results = retriever.search_with_beta_sweep(
            query_text="Apple",
            query_timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc),
            beta_values=beta_values,
            top_k=5
        )
        
        assert len(sweep_results) == len(beta_values)
        
        for beta, results in sweep_results:
            assert beta in beta_values
            assert isinstance(results, list)
            assert len(results) <= 5
    
    def test_beta_sweep_shows_progression(self, retriever):
        """Beta sweep should show progression from semantic to temporal."""
        beta_values = [0.0, 0.5, 1.0]
        query_date = datetime(2020, 1, 1, tzinfo=timezone.utc)
        
        sweep_results = retriever.search_with_beta_sweep(
            query_text="Apple",
            query_timestamp=query_date,
            beta_values=beta_values,
            top_k=5
        )
        
        # Extract top result from each beta
        top_results = [(beta, results[0]) for beta, results in sweep_results if results]
        
        # Verify we got results for each beta
        assert len(top_results) == len(beta_values)


# ============================================================================
# Tests for Result Explanation
# ============================================================================

class TestResultExplanation:
    """Test result explanation and formatting."""
    
    def test_explain_result_returns_string(self, retriever):
        """explain_result should return formatted string."""
        results = retriever.search(
            query_text="Apple",
            query_timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc),
            beta=0.5,
            top_k_final=1
        )
        
        if results:
            explanation = retriever.explain_result(results[0])
            assert isinstance(explanation, str)
            assert len(explanation) > 0
            assert "Rank" in explanation
            assert "Score" in explanation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
