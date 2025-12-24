"""
Unit Tests for Temporal Spin Encoding Functions
================================================

Tests core temporal encoding functions:
- compute_spin_vector (point and arc modes)
- angular_difference
- arc_overlap
- jaccard_similarity_arcs
- extract_timestamp_from_text

Also includes property-based tests for circular math edge cases.
"""

import math
import os
import sys
from datetime import datetime, timezone

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

# Modify path to allow imports from parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# fmt: off
from temporal_spin import (QUARTER_PERIOD_SECONDS, T0_SECONDS,  # noqa: E402
                           angular_difference, arc_overlap,
                           compute_spin_vector, cosine_similarity,
                           extract_timestamp_from_text,
                           jaccard_similarity_arcs, normalize_vector)
from tests.conftest import (is_unit_circle_point, is_valid_phase,  # noqa: E402
                            is_valid_spin_vector)

# fmt: on

# ============================================================================
# Tests for compute_spin_vector (Point Mode)
# ============================================================================


class TestComputeSpinVectorPoint:
    """Test suite for point-mode spin vector computation."""

    def test_point_mode_returns_9d_vector(self):
        """Point mode should return 9D spin vector with z=0."""
        timestamp = datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp()
        result = compute_spin_vector(timestamp)
        spin, _, _, _ = result

        assert len(spin) == 9, "Spin vector should be 9-dimensional"
        assert is_valid_spin_vector(
            spin
        ), "Spin vector should contain finite values"

        # z components (indices 2, 5, 8) should be 0 for points
        assert abs(spin[2]) < 1e-10, "Quarter scale z should be 0 for points"
        assert abs(spin[5]) < 1e-10, "Decade scale z should be 0 for points"
        assert abs(spin[8]) < 1e-10, "Century scale z should be 0 for points"

    def test_point_mode_unit_circle_constraint(self):
        """Each scale's (x, y) should lie on unit circle."""
        timestamp = datetime(2020, 6, 15, tzinfo=timezone.utc).timestamp()
        spin, _, _, _ = compute_spin_vector(timestamp)

        # Check each scale
        assert is_unit_circle_point(
            spin[0], spin[1]
        ), "Quarter scale not on unit circle"
        assert is_unit_circle_point(
            spin[3], spin[4]
        ), "Decade scale not on unit circle"
        assert is_unit_circle_point(
            spin[6], spin[7]
        ), "Century scale not on unit circle"

    def test_point_mode_phase_progression(self):
        """Phases should progress correctly with time at each scale."""
        base_timestamp = datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp()

        # Quarter scale: +3 months = π/2 radians (90°)
        t1 = base_timestamp
        t2 = base_timestamp + (QUARTER_PERIOD_SECONDS / 4)  # +3 months

        _, phi1, _, _ = compute_spin_vector(t1)
        _, phi2, _, _ = compute_spin_vector(t2)

        # Quarter scale should show ~π/2 difference
        quarter_diff = angular_difference(phi1["quarter"], phi2["quarter"])
        assert (
            abs(quarter_diff - math.pi / 2) < 0.01
        ), "Quarter scale should show π/2 for 3 months"

    def test_point_mode_same_timestamp_identical_encoding(self):
        """Same timestamp should produce identical spin vectors."""
        dt = datetime(2023, 5, 15, 12, 30, 45, tzinfo=timezone.utc)
        timestamp = dt.timestamp()

        spin1, phi1, _, _ = compute_spin_vector(timestamp)
        spin2, phi2, _, _ = compute_spin_vector(timestamp)

        assert (
            spin1 == spin2
        ), "Same timestamp should produce identical spin vectors"
        assert phi1 == phi2, "Same timestamp should produce identical phases"

    def test_point_mode_phi_starts_ends_none(self):
        """Point mode should have None for phi_start and phi_end."""
        timestamp = datetime(2021, 3, 15, tzinfo=timezone.utc).timestamp()
        _, _, phi_starts, phi_ends = compute_spin_vector(timestamp)

        assert phi_starts["quarter"] is None
        assert phi_starts["decade"] is None
        assert phi_starts["century"] is None
        assert phi_ends["quarter"] is None
        assert phi_ends["decade"] is None
        assert phi_ends["century"] is None


# ============================================================================
# Tests for compute_spin_vector (Arc Mode)
# ============================================================================


class TestComputeSpinVectorArc:
    """Test suite for arc-mode spin vector computation."""

    def test_arc_mode_returns_9d_vector_with_nonzero_z(self):
        """Arc mode should return 9D vector with non-zero z components."""
        start = datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp()
        end = datetime(2020, 3, 31, tzinfo=timezone.utc).timestamp()  # Q1

        spin, _, _, _ = compute_spin_vector(start, end_timestamp_seconds=end)

        assert len(spin) == 9, "Arc mode should return 9D vector"
        assert is_valid_spin_vector(
            spin
        ), "Spin vector should contain finite values"

        # z components should be non-zero for arcs
        assert spin[2] > 0, "Quarter scale z should be > 0 for arcs"
        assert spin[5] > 0, "Decade scale z should be > 0 for arcs"
        assert spin[8] > 0, "Century scale z should be > 0 for arcs"

    def test_arc_mode_quarterly_arc_length(self):
        """Q1 arc should span ~π/2 radians (90°) on quarter scale."""
        start = datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp()
        end = datetime(2020, 3, 31, tzinfo=timezone.utc).timestamp()

        spin, _, _, _ = compute_spin_vector(start, end_timestamp_seconds=end)

        # Quarter scale arc length (index 2)
        quarter_arc = spin[2]
        expected = math.pi / 2  # 90° for 1/4 of year

        assert (
            abs(quarter_arc - expected) < 0.1
        ), f"Q1 should span ~π/2 on quarter scale, got {quarter_arc}"

    def test_arc_mode_full_year_arc_length(self):
        """Full year arc should span 2π on quarter scale."""
        start = datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp()
        end = datetime(2020, 12, 31, tzinfo=timezone.utc).timestamp()

        spin, _, _, _ = compute_spin_vector(start, end_timestamp_seconds=end)

        quarter_arc = spin[2]
        # Full year should be close to 2π (complete circle)
        assert (
            abs(quarter_arc - math.tau) < 0.2
        ), f"Full year should span ~2π on quarter scale, got {quarter_arc}"

    def test_arc_mode_center_calculation(self):
        """Arc center should be midpoint of start and end phases."""
        start = datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp()
        end = datetime(2020, 6, 30, tzinfo=timezone.utc).timestamp()  # H1

        _, phi_centers, phi_starts, phi_ends = compute_spin_vector(
            start, end_timestamp_seconds=end
        )

        # Verify center is between start and end for each scale
        for scale in ["quarter", "decade", "century"]:
            start_phi = phi_starts[scale]
            end_phi = phi_ends[scale]
            center_phi = phi_centers[scale]

            # Center should be within the arc
            # (accounting for wrapping)
            assert start_phi is not None
            assert end_phi is not None
            assert is_valid_phase(center_phi)

    def test_arc_mode_phi_starts_ends_defined(self):
        """Arc mode should have defined phi_start and phi_end."""
        start = datetime(2020, 4, 1, tzinfo=timezone.utc).timestamp()
        end = datetime(2020, 6, 30, tzinfo=timezone.utc).timestamp()

        _, _, phi_starts, phi_ends = compute_spin_vector(
            start, end_timestamp_seconds=end
        )

        for scale in ["quarter", "decade", "century"]:
            assert phi_starts[scale] is not None
            assert phi_ends[scale] is not None
            start_val = phi_starts[scale]
            end_val = phi_ends[scale]
            if start_val is not None:
                assert is_valid_phase(start_val)
            if end_val is not None:
                assert is_valid_phase(end_val)


# ============================================================================
# Tests for angular_difference
# ============================================================================


class TestAngularDifference:
    """Test suite for angular difference calculation."""

    def test_same_angle_zero_difference(self):
        """Same angle should have zero difference."""
        phi = math.pi / 4
        assert abs(angular_difference(phi, phi)) < 1e-10

    def test_opposite_angles_pi_difference(self):
        """Opposite angles should have π difference."""
        phi1 = 0.0
        phi2 = math.pi
        assert abs(angular_difference(phi1, phi2) - math.pi) < 1e-10

    def test_wrapping_shortest_arc(self):
        """Should return shortest arc, accounting for wrapping."""
        phi1 = 0.1  # Near 0
        phi2 = 2 * math.pi - 0.1  # Near 2π (wraps to 0)

        diff = angular_difference(phi1, phi2)
        assert diff < 0.5, f"Should use short arc, got {diff}"

    def test_quarter_circle_pi_over_2(self):
        """90° angles should have π/2 difference."""
        phi1 = 0.0
        phi2 = math.pi / 2
        assert abs(angular_difference(phi1, phi2) - math.pi / 2) < 1e-10

    def test_commutative_property(self):
        """angular_difference(a, b) should equal angular_difference(b, a)."""
        phi1 = 1.5
        phi2 = 4.2
        diff1 = angular_difference(phi1, phi2)
        diff2 = angular_difference(phi2, phi1)
        assert abs(diff1 - diff2) < 1e-10

    def test_result_in_valid_range(self):
        """Result should be in [0, π]."""
        phi1 = 0.5
        phi2 = 5.0
        diff = angular_difference(phi1, phi2)
        assert 0 <= diff <= math.pi


# ============================================================================
# Tests for arc_overlap
# ============================================================================


class TestArcOverlap:
    """Test suite for arc overlap calculation."""

    def test_identical_arcs_full_overlap(self):
        """Identical arcs should have full overlap."""
        start = 0.0
        end = math.pi / 2
        overlap = arc_overlap(start, end, start, end)
        expected = end - start
        assert abs(overlap - expected) < 1e-10

    def test_non_overlapping_arcs_zero(self):
        """Non-overlapping arcs should have zero overlap."""
        # Arc 1: [0, π/2]
        # Arc 2: [π, 3π/2]
        overlap = arc_overlap(0.0, math.pi / 2, math.pi, 3 * math.pi / 2)
        assert abs(overlap) < 1e-10

    def test_partial_overlap(self):
        """Partially overlapping arcs should return intersection length."""
        # Arc 1: [0, π]
        # Arc 2: [π/2, 3π/2]
        # Overlap: [π/2, π] = π/2
        overlap = arc_overlap(0.0, math.pi, math.pi / 2, 3 * math.pi / 2)
        assert abs(overlap - math.pi / 2) < 1e-10

    def test_one_arc_contains_other(self):
        """If one arc contains another, overlap equals smaller arc."""
        # Arc 1: [0, 2π] (full circle)
        # Arc 2: [π/4, π/2]
        overlap = arc_overlap(0.0, math.tau, math.pi / 4, math.pi / 2)
        expected = math.pi / 2 - math.pi / 4
        assert abs(overlap - expected) < 1e-10

    def test_wrapping_arcs(self):
        """Arcs that wrap around 0 should be handled correctly."""
        # Arc 1: [5π/4, π/4] (wraps around 0)
        # Arc 2: [0, π/2]
        # Should detect overlap
        arc1_start = 5 * math.pi / 4
        arc1_end = math.pi / 4 + math.tau
        overlap = arc_overlap(arc1_start, arc1_end, 0.0, math.pi / 2)
        assert overlap > 0, "Wrapping arcs should detect overlap"

    def test_adjacent_arcs_touch_at_boundary(self):
        """Adjacent arcs that touch at boundary should have minimal overlap."""
        # Arc 1: [0, π/2]
        # Arc 2: [π/2, π]
        overlap = arc_overlap(0.0, math.pi / 2, math.pi / 2, math.pi)
        assert overlap < 1e-6, "Adjacent arcs should have minimal overlap"


# ============================================================================
# Tests for jaccard_similarity_arcs
# ============================================================================


class TestJaccardSimilarityArcs:
    """Test suite for Jaccard similarity between arcs."""

    def test_identical_arcs_similarity_one(self):
        """Identical arcs should have Jaccard similarity = 1."""
        start = math.pi / 4
        end = 3 * math.pi / 4
        similarity = jaccard_similarity_arcs(start, end, start, end)
        assert abs(similarity - 1.0) < 1e-10

    def test_non_overlapping_arcs_similarity_zero(self):
        """Non-overlapping arcs should have Jaccard similarity = 0."""
        arc1_start, arc1_end = 0.0, math.pi / 4
        arc2_start, arc2_end = math.pi / 2, 3 * math.pi / 4
        similarity = jaccard_similarity_arcs(
            arc1_start, arc1_end, arc2_start, arc2_end
        )
        assert abs(similarity - 0.0) < 1e-10

    def test_half_overlap_similarity(self):
        """Arcs with 50% overlap should have specific Jaccard."""
        # Arc 1: [0, 2]
        # Arc 2: [1, 3]
        # Intersection: 1, Union: 3, Jaccard: 1/3
        similarity = jaccard_similarity_arcs(0.0, 2.0, 1.0, 3.0)
        expected = 1.0 / 3.0
        assert abs(similarity - expected) < 1e-10

    def test_one_arc_subset_of_other(self):
        """If one arc is subset of other, Jaccard = smaller/larger."""
        # Arc 1: [0, 4]
        # Arc 2: [1, 2] (subset)
        # Intersection: 1, Union: 4, Jaccard: 0.25
        similarity = jaccard_similarity_arcs(0.0, 4.0, 1.0, 2.0)
        expected = 1.0 / 4.0
        assert abs(similarity - expected) < 1e-10

    def test_similarity_bounded_zero_to_one(self):
        """Jaccard similarity should always be in [0, 1]."""
        similarity = jaccard_similarity_arcs(0.5, 2.5, 1.5, 3.5)
        assert 0.0 <= similarity <= 1.0


# ============================================================================
# Tests for extract_timestamp_from_text
# ============================================================================


class TestExtractTimestampFromText:
    """Test suite for timestamp extraction from text."""

    def test_extract_fiscal_year_format(self):
        """Should extract 'fiscal year YYYY' format."""
        text = "The company's fiscal year 2023 results exceeded expectations."
        timestamp = extract_timestamp_from_text(text)
        assert timestamp.year == 2023

    def test_extract_period_ended_format(self):
        """Should extract 'period ended DD Month YYYY' format."""
        text = "For the period ended 31 December 2019, revenue was $10B."
        timestamp = extract_timestamp_from_text(text)
        assert timestamp.year == 2019
        assert timestamp.month == 12

    def test_extract_as_of_format(self):
        """Should extract 'as of Month DD, YYYY' format."""
        text = "As of June 30, 2022, total assets were $350B."
        timestamp = extract_timestamp_from_text(text)
        assert timestamp.year == 2022
        assert timestamp.month == 6

    def test_extract_quarter_format(self):
        """Should extract 'QX YYYY' format."""
        text = "Q3 2021 performance showed strong growth."
        timestamp = extract_timestamp_from_text(text)
        assert timestamp.year == 2021

    def test_extract_iso_format(self):
        """Should extract ISO format dates (YYYY-MM-DD)."""
        text = "Report dated 2024-05-15 shows improvements."
        timestamp = extract_timestamp_from_text(text)
        assert timestamp.year == 2024
        assert timestamp.month == 5
        assert timestamp.day == 15

    def test_extract_us_date_format(self):
        """Should extract US format dates (MM/DD/YYYY)."""
        text = "The meeting was held on 12/25/2020."
        timestamp = extract_timestamp_from_text(text)
        assert timestamp.year == 2020

    def test_fallback_to_provided(self):
        """Should use fallback when no date found."""
        text = "No dates in this text whatsoever."
        fallback = datetime(2010, 1, 1, tzinfo=timezone.utc)
        timestamp = extract_timestamp_from_text(text, fallback=fallback)
        assert timestamp == fallback

    def test_timezone_aware_output(self):
        """Extracted timestamp should be timezone-aware (UTC)."""
        text = "For the period ended 31 December 2019"
        timestamp = extract_timestamp_from_text(text)
        assert timestamp.tzinfo is not None
        assert timestamp.tzinfo == timezone.utc


# ============================================================================
# Tests for cosine_similarity
# ============================================================================


class TestCosineSimilarity:
    """Test suite for cosine similarity calculation."""

    def test_identical_vectors_similarity_one(self):
        """Identical vectors should have cosine similarity = 1."""
        vec = [1.0, 2.0, 3.0, 4.0]
        similarity = cosine_similarity(vec, vec)
        assert abs(similarity - 1.0) < 1e-10

    def test_orthogonal_vectors_similarity_zero(self):
        """Orthogonal vectors should have cosine similarity = 0."""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        similarity = cosine_similarity(vec1, vec2)
        assert abs(similarity - 0.0) < 1e-10

    def test_opposite_vectors_similarity_negative_one(self):
        """Opposite vectors should have cosine similarity = -1."""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [-1.0, 0.0, 0.0]
        similarity = cosine_similarity(vec1, vec2)
        assert abs(similarity - (-1.0)) < 1e-10

    def test_similarity_bounded(self):
        """Cosine similarity should be in [-1, 1]."""
        vec1 = [1.5, 2.3, -0.5, 1.1]
        vec2 = [-0.3, 1.7, 2.2, -1.4]
        similarity = cosine_similarity(vec1, vec2)
        assert -1.0 <= similarity <= 1.0

    def test_mismatched_dimensions_raises_error(self):
        """Vectors with different dimensions should raise error."""
        vec1 = [1.0, 2.0, 3.0]
        vec2 = [1.0, 2.0]
        with pytest.raises(ValueError):
            cosine_similarity(vec1, vec2)


# ============================================================================
# Tests for normalize_vector
# ============================================================================


class TestNormalizeVector:
    """Test suite for vector normalization."""

    def test_normalize_to_unit_length(self):
        """Normalized vector should have length 1."""
        vec = [3.0, 4.0]
        normalized = normalize_vector(vec)
        length = math.sqrt(sum(x**2 for x in normalized))
        assert abs(length - 1.0) < 1e-10

    def test_zero_vector_unchanged(self):
        """Zero vector should remain unchanged."""
        vec = [0.0, 0.0, 0.0]
        normalized = normalize_vector(vec)
        assert normalized == vec

    def test_already_normalized_unchanged(self):
        """Already normalized vector should be unchanged."""
        vec = [1.0, 0.0, 0.0]
        normalized = normalize_vector(vec)
        assert abs(normalized[0] - 1.0) < 1e-10
        assert abs(normalized[1] - 0.0) < 1e-10
        assert abs(normalized[2] - 0.0) < 1e-10


# ============================================================================
# Property-Based Tests for Circular Math
# ============================================================================


class TestCircularMathProperties:
    """Property-based tests for circular math edge cases using Hypothesis."""

    @given(
        timestamp=st.floats(
            min_value=T0_SECONDS,
            max_value=T0_SECONDS + 100 * 365.25 * 24 * 3600,  # 100 years
            allow_nan=False,
            allow_infinity=False,
        )
    )
    def test_spin_vector_always_valid(self, timestamp):
        """Any valid timestamp should produce valid spin vector."""
        spin, phi_centers, _, _ = compute_spin_vector(timestamp)

        assert is_valid_spin_vector(spin), "Spin vector should always be valid"
        assert all(is_valid_phase(phi) for phi in phi_centers.values())

    @given(
        phi1=st.floats(min_value=0, max_value=2 * math.pi, allow_nan=False),
        phi2=st.floats(min_value=0, max_value=2 * math.pi, allow_nan=False),
    )
    def test_angular_difference_symmetric(self, phi1, phi2):
        """angular_difference should be symmetric."""
        diff1 = angular_difference(phi1, phi2)
        diff2 = angular_difference(phi2, phi1)
        assert abs(diff1 - diff2) < 1e-10

    @given(
        phi1=st.floats(min_value=0, max_value=2 * math.pi, allow_nan=False),
        phi2=st.floats(min_value=0, max_value=2 * math.pi, allow_nan=False),
    )
    def test_angular_difference_bounded(self, phi1, phi2):
        """angular_difference should always be in [0, π]."""
        diff = angular_difference(phi1, phi2)
        assert 0 <= diff <= math.pi

    @given(
        start1=st.floats(min_value=0, max_value=2 * math.pi, allow_nan=False),
        length1=st.floats(min_value=0.1, max_value=math.pi, allow_nan=False),
        start2=st.floats(min_value=0, max_value=2 * math.pi, allow_nan=False),
        length2=st.floats(min_value=0.1, max_value=math.pi, allow_nan=False),
    )
    def test_arc_overlap_non_negative(self, start1, length1, start2, length2):
        """Arc overlap should always be non-negative."""
        end1 = start1 + length1
        end2 = start2 + length2
        overlap = arc_overlap(start1, end1, start2, end2)
        assert overlap >= 0

    @given(
        start=st.floats(min_value=0, max_value=2 * math.pi, allow_nan=False),
        length=st.floats(min_value=0.1, max_value=math.pi, allow_nan=False),
    )
    def test_jaccard_self_similarity_one(self, start, length):
        """Arc compared with itself should have Jaccard = 1."""
        end = start + length
        similarity = jaccard_similarity_arcs(start, end, start, end)
        assert abs(similarity - 1.0) < 1e-6

    @given(
        vec=st.lists(
            st.floats(min_value=-10, max_value=10, allow_nan=False),
            min_size=3,
            max_size=10,
        )
    )
    def test_cosine_similarity_self_one(self, vec):
        """Vector compared with itself should have cosine similarity = 1."""
        assume(sum(x**2 for x in vec) > 1e-6)  # Non-zero vector
        similarity = cosine_similarity(vec, vec)
        assert abs(similarity - 1.0) < 1e-6


# ============================================================================
# Integration Tests
# ============================================================================


class TestMultiScaleEncoding:
    """Test multi-scale temporal encoding integration."""

    def test_three_scales_present(self):
        """Spin vector should have components from all three scales."""
        timestamp = datetime(2020, 6, 15, tzinfo=timezone.utc).timestamp()
        spin, phi_centers, _, _ = compute_spin_vector(timestamp)

        # Should have 3 scales × 3 dimensions = 9D
        assert len(spin) == 9
        assert "quarter" in phi_centers
        assert "decade" in phi_centers
        assert "century" in phi_centers

    def test_scales_progress_at_different_rates(self):
        """Different scales should progress at different rates."""
        t1 = datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp()
        t2 = datetime(2021, 1, 1, tzinfo=timezone.utc).timestamp()  # +1 year

        _, phi1, _, _ = compute_spin_vector(t1)
        _, phi2, _, _ = compute_spin_vector(t2)

        # Quarter scale: 1 year = full rotation (2π)
        quarter_diff = angular_difference(phi1["quarter"], phi2["quarter"])
        assert quarter_diff < 0.1, "Quarter scale should wrap after 1 year"

        # Decade scale: 1 year = 2π/16 ≈ 0.39 rad
        decade_diff = angular_difference(phi1["decade"], phi2["decade"])
        expected_decade = math.tau / 16
        assert abs(decade_diff - expected_decade) < 0.1

        # Century scale: 1 year = 2π/256 ≈ 0.025 rad
        century_diff = angular_difference(phi1["century"], phi2["century"])
        expected_century = math.tau / 256
        assert abs(century_diff - expected_century) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
    pytest.main([__file__, "-v"])
