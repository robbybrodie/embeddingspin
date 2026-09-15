"""
Intervals, phase, arc algebra, encoding, splitting, padding and lazy traversal.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from temporal_config import (
    CENTURY_SCALE,
    DECADE_SCALE,
    DEFAULT_HIERARCHY,
    MILLENNIUM_SCALE,
    QUARTER_SCALE,
    TAU,
    TemporalHierarchy,
)
from temporal_encoding import (
    EPS,
    MAX_REPRESENTATIONS,
    TemporalEncoding,
    TemporalInterval,
    angular_difference,
    arc_contains_point,
    arc_overlap,
    arc_overlap3,
    boundary_moments,
    encode,
    encode_single,
    encode_tuple,
    evaluate_scale,
    jaccard_arcs,
    moment_from_year_position,
    neutral_tuple,
    pad_to_hierarchy,
    phase_of,
    split_at_boundaries,
    temporal_alignment,
    traversal_plan,
)


def utc(*args) -> datetime:
    return datetime(*args, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Intervals
# ---------------------------------------------------------------------------


class TestTemporalInterval:
    def test_naive_datetimes_are_assumed_utc(self):
        interval = TemporalInterval(datetime(2026, 1, 1))
        assert interval.start.tzinfo == timezone.utc

    def test_a_zero_length_interval_collapses_to_a_point(self):
        moment = utc(2026, 1, 1)
        assert TemporalInterval(moment, moment).is_point

    def test_end_before_start_is_rejected(self):
        with pytest.raises(ValueError, match="precedes"):
            TemporalInterval(utc(2026, 4, 1), utc(2026, 1, 1))

    def test_point_has_zero_duration(self):
        point = TemporalInterval.point(utc(2026, 2, 15))
        assert point.is_point
        assert point.duration_days == 0.0
        assert point.midpoint == point.start

    @pytest.mark.parametrize(
        "year,quarter,days",
        [(2026, 1, 90), (2026, 2, 91), (2026, 3, 92), (2026, 4, 92),
         (2024, 1, 91)],  # leap year
    )
    def test_calendar_quarters_are_unequal(self, year, quarter, days):
        assert TemporalInterval.of_quarter(year, quarter).duration_days == days

    def test_quarters_are_half_open_and_tile_the_year(self):
        quarters = [TemporalInterval.of_quarter(2026, q) for q in (1, 2, 3, 4)]
        for earlier, later in zip(quarters, quarters[1:]):
            assert earlier.end == later.start
        assert quarters[0].start == TemporalInterval.of_year(2026).start
        assert quarters[-1].end == TemporalInterval.of_year(2026).end
        assert sum(q.duration_days for q in quarters) == 365

    def test_quarter_index_is_validated(self):
        with pytest.raises(ValueError, match="1-4"):
            TemporalInterval.of_quarter(2026, 5)

    def test_of_year_is_half_open(self):
        year = TemporalInterval.of_year(2024)
        assert year.start == utc(2024, 1, 1)
        assert year.end == utc(2025, 1, 1)
        assert year.duration_days == 366

    def test_of_month_wraps_december_into_the_next_year(self):
        december = TemporalInterval.of_month(2026, 12)
        assert december.end == utc(2027, 1, 1)

    def test_spanning_is_inclusive_of_the_last_year(self):
        span = TemporalInterval.spanning(2017, 2022)
        assert span.start == utc(2017, 1, 1)
        assert span.end == utc(2023, 1, 1)
        assert span.years_span(DEFAULT_HIERARCHY) == pytest.approx(6.0)

    def test_midpoint_of_q1_2026(self):
        """90 days, so the midpoint is 45.0 elapsed days in: 15 February, not 14."""
        interval = TemporalInterval.of_quarter(2026, 1)
        assert interval.duration_days == 90.0
        assert interval.midpoint == utc(2026, 2, 15)


# ---------------------------------------------------------------------------
# Phase (Formula 1)
# ---------------------------------------------------------------------------


class TestPhase:
    @pytest.mark.parametrize("year", [1900, 1916, 1999, 2000, 2023, 2024, 2100, 2155])
    def test_january_first_is_phase_zero_on_the_one_year_circle(self, year):
        """
        No leap-year drift: the calendar convention measures position as a fraction
        of the actual year, so New Year always lands exactly on zero degrees.
        """
        phi = phase_of(utc(year, 1, 1), QUARTER_SCALE, DEFAULT_HIERARCHY)
        assert phi == pytest.approx(0.0, abs=1e-12)

    def test_phase_is_always_in_range(self):
        for year in range(1900, 2150, 7):
            for month in (1, 5, 9):
                for scale in DEFAULT_HIERARCHY.scales:
                    phi = phase_of(utc(year, month, 14), scale, DEFAULT_HIERARCHY)
                    assert 0.0 <= phi < TAU

    def test_the_epoch_itself_is_phase_zero_everywhere(self):
        for scale in DEFAULT_HIERARCHY.scales:
            assert phase_of(DEFAULT_HIERARCHY.epoch, scale, DEFAULT_HIERARCHY) == 0.0

    def test_moments_before_the_epoch_still_yield_a_positive_phase(self):
        """
        Pre-epoch instants wrap to the far side of the circle rather than producing a
        negative phase, and land on the same phase as the matching date in any other
        common year.
        """
        phi = phase_of(utc(1899, 7, 1), QUARTER_SCALE, DEFAULT_HIERARCHY)
        assert 0.0 <= phi < TAU
        assert phi == pytest.approx(
            phase_of(utc(1901, 7, 1), QUARTER_SCALE, DEFAULT_HIERARCHY), abs=1e-12
        )

    def test_the_decade_circle_advances_one_sixteenth_per_year(self):
        a = phase_of(utc(1901, 1, 1), DECADE_SCALE, DEFAULT_HIERARCHY)
        assert a == pytest.approx(TAU / 16, abs=1e-12)

    def test_the_decade_circle_wraps_every_sixteen_years(self):
        assert phase_of(utc(1916, 1, 1), DECADE_SCALE, DEFAULT_HIERARCHY) == pytest.approx(
            0.0, abs=1e-12
        )

    def test_moment_from_year_position_inverts_the_convention(self):
        for moment in (utc(2024, 3, 17), utc(1955, 11, 2), utc(2100, 1, 1)):
            from temporal_config import calendar_year_position

            recovered = moment_from_year_position(calendar_year_position(moment))
            assert abs((recovered - moment).total_seconds()) < 1e-3


# ---------------------------------------------------------------------------
# Arc algebra
# ---------------------------------------------------------------------------


class TestArcAlgebra:
    def test_angular_difference_is_symmetric_and_bounded(self):
        assert angular_difference(0.0, math.pi) == pytest.approx(math.pi)
        assert angular_difference(0.1, TAU - 0.1) == pytest.approx(0.2)
        assert angular_difference(TAU - 0.1, 0.1) == pytest.approx(0.2)
        for a in (0.0, 1.0, 3.0, 6.0):
            for b in (0.0, 2.0, 4.0, 6.2):
                assert 0.0 <= angular_difference(a, b) <= math.pi + EPS

    def test_overlap_of_disjoint_arcs_is_zero(self):
        assert arc_overlap(0.0, 1.0, 2.0, 1.0) == pytest.approx(0.0)

    def test_overlap_of_identical_arcs_is_their_length(self):
        assert arc_overlap(1.0, 2.0, 1.0, 2.0) == pytest.approx(2.0)

    def test_overlap_handles_the_wrap(self):
        """An arc from 350° to 10° meets an arc from 0° to 20°."""
        start = math.radians(350)
        assert arc_overlap(start, math.radians(20), 0.0, math.radians(20)) == pytest.approx(
            math.radians(10)
        )

    def test_a_full_circle_overlaps_everything(self):
        assert arc_overlap(0.0, TAU, 3.0, 0.5) == pytest.approx(0.5)

    def test_three_way_overlap(self):
        assert arc_overlap3(0.0, 2.0, 1.0, 2.0, 1.5, 2.0) == pytest.approx(0.5)
        assert arc_overlap3(0.0, 1.0, 2.0, 1.0, 4.0, 1.0) == pytest.approx(0.0)

    def test_arc_contains_point(self):
        assert arc_contains_point(1.0, 1.0, 1.5)
        assert not arc_contains_point(1.0, 1.0, 2.5)
        # Wrapping arc.
        assert arc_contains_point(TAU - 0.5, 1.0, 0.2)

    def test_jaccard_of_identical_arcs_is_one(self):
        assert jaccard_arcs(1.0, 2.0, 1.0, 2.0) == pytest.approx(1.0)

    def test_jaccard_of_disjoint_arcs_is_zero(self):
        assert jaccard_arcs(0.0, 1.0, 3.0, 1.0) == pytest.approx(0.0)

    def test_jaccard_of_a_quarter_inside_a_year(self):
        """A 90° arc inside a 360° arc: intersection 90, union 360."""
        assert jaccard_arcs(0.0, TAU, 0.0, TAU / 4) == pytest.approx(0.25)

    def test_temporal_alignment_kernel(self):
        assert temporal_alignment(0.0, 5.0) == pytest.approx(1.0)
        assert temporal_alignment(1.0, 0.0) == pytest.approx(1.0)  # beta=0 flattens it
        assert temporal_alignment(1.0, 5.0) < temporal_alignment(0.5, 5.0)


# ---------------------------------------------------------------------------
# Tuple encoding
# ---------------------------------------------------------------------------


class TestEncodeTuple:
    def test_a_point_has_zero_arc_length(self):
        t = encode_tuple(
            TemporalInterval.point(utc(2026, 2, 15)), QUARTER_SCALE, DEFAULT_HIERARCHY
        )
        assert t.z == 0.0
        assert t.is_point
        assert t.phi_start == t.phi_center

    def test_cos_and_sin_describe_the_arc_centre(self):
        t = encode_tuple(
            TemporalInterval.of_quarter(2026, 1), QUARTER_SCALE, DEFAULT_HIERARCHY
        )
        assert t.cos == pytest.approx(math.cos(t.phi_center))
        assert t.sin == pytest.approx(math.sin(t.phi_center))
        assert t.cos ** 2 + t.sin ** 2 == pytest.approx(1.0)

    def test_a_full_year_saturates_the_one_year_circle(self):
        t = encode_tuple(
            TemporalInterval.of_year(2021), QUARTER_SCALE, DEFAULT_HIERARCHY
        )
        assert t.z == pytest.approx(TAU)
        assert t.is_full_circle

    def test_an_arc_longer_than_the_period_is_capped_at_one_revolution(self):
        t = encode_tuple(
            TemporalInterval.spanning(2000, 2099), QUARTER_SCALE, DEFAULT_HIERARCHY
        )
        assert t.z == pytest.approx(TAU)

    def test_arc_length_scales_with_the_period(self):
        interval = TemporalInterval.of_year(2021)
        quarter = encode_tuple(interval, QUARTER_SCALE, DEFAULT_HIERARCHY)
        decade = encode_tuple(interval, DECADE_SCALE, DEFAULT_HIERARCHY)
        century = encode_tuple(interval, CENTURY_SCALE, DEFAULT_HIERARCHY)
        assert quarter.z == pytest.approx(TAU)
        assert decade.z == pytest.approx(TAU / 16)
        assert century.z == pytest.approx(TAU / 256)

    def test_phi_end_is_unwrapped(self):
        t = encode_tuple(
            TemporalInterval.of_quarter(2026, 4), QUARTER_SCALE, DEFAULT_HIERARCHY
        )
        assert t.phi_end == pytest.approx(t.phi_start + t.z)
        assert t.phi_end > TAU - 1e-6

    def test_tuple_round_trips_through_a_dict(self):
        from temporal_encoding import ScaleTuple

        t = encode_tuple(
            TemporalInterval.of_quarter(2026, 2), QUARTER_SCALE, DEFAULT_HIERARCHY
        )
        assert ScaleTuple.from_dict(t.to_dict()) == t


# ---------------------------------------------------------------------------
# Encoding and splitting
# ---------------------------------------------------------------------------


class TestEncoding:
    def test_the_temporal_block_is_nine_dimensions(self):
        encoding = encode_single(TemporalInterval.of_quarter(2026, 1))
        assert encoding.dimensions == 9
        assert len(encoding.to_vector()) == 9

    def test_the_vector_is_cos_sin_z_per_scale_finest_first(self):
        encoding = encode_single(TemporalInterval.of_quarter(2026, 1))
        vector = encoding.to_vector()
        assert encoding.scale_names == ("quarter", "decade", "century")
        for index, name in enumerate(encoding.scale_names):
            t = encoding.tuple_for(name)
            assert vector[3 * index: 3 * index + 3] == [t.cos, t.sin, t.z]

    def test_tuple_for_an_unknown_scale_raises(self):
        encoding = encode_single(TemporalInterval.of_year(2021))
        with pytest.raises(KeyError):
            encoding.tuple_for("millennium")

    def test_a_within_year_interval_is_not_split(self):
        reps = encode(TemporalInterval.of_quarter(2021, 2))
        assert len(reps) == 1
        assert not reps[0].is_split

    def test_a_six_year_span_yields_six_representations(self):
        """
        2017 through 2022 crosses five one-year boundaries, so it is indexed six
        times. Splitting happens at the finest scale only.
        """
        reps = encode(TemporalInterval.spanning(2017, 2022))
        assert len(reps) == 6
        assert all(r.representation_count == 6 for r in reps)
        assert [r.representation_index for r in reps] == [0, 1, 2, 3, 4, 5]

    def test_every_component_covers_exactly_one_calendar_year(self):
        reps = encode(TemporalInterval.spanning(2017, 2022))
        years = [r.interval.start.year for r in reps]
        assert years == [2017, 2018, 2019, 2020, 2021, 2022]
        for rep in reps:
            assert rep.interval.start.month == 1 and rep.interval.start.day == 1
            assert rep.interval.end == utc(rep.interval.start.year + 1, 1, 1)

    def test_split_components_share_one_group_id(self):
        reps = encode(TemporalInterval.spanning(2017, 2022), group_id="review")
        assert {r.group_id for r in reps} == {"review"}

    def test_every_component_carries_the_full_source_interval(self):
        source = TemporalInterval.spanning(2017, 2022)
        for rep in encode(source):
            assert rep.source_interval == source

    def test_each_component_saturates_the_one_year_circle(self):
        for rep in encode(TemporalInterval.spanning(2017, 2022)):
            assert rep.tuple_for("quarter").z == pytest.approx(TAU)

    def test_components_are_distinguished_on_the_decade_circle(self):
        reps = encode(TemporalInterval.spanning(2017, 2022))
        phases = [rep.tuple_for("decade").phi_start for rep in reps]
        assert len(set(round(p, 9) for p in phases)) == 6

    def test_a_partial_year_at_each_end_is_preserved(self):
        interval = TemporalInterval(utc(2020, 7, 1), utc(2022, 4, 1))
        components = split_at_boundaries(interval)
        assert len(components) == 3
        assert components[0].start == utc(2020, 7, 1)
        assert components[0].end == utc(2021, 1, 1)
        assert components[-1].end == utc(2022, 4, 1)

    def test_split_false_forces_a_single_representation(self):
        reps = encode(TemporalInterval.spanning(2017, 2022), split=False)
        assert len(reps) == 1
        assert reps[0].tuple_for("quarter").z == pytest.approx(TAU)

    def test_splitting_is_capped(self):
        """
        A document spanning more years than the cap allows is left whole rather than
        exploding the index; its finest arc saturates instead.
        """
        reps = encode(TemporalInterval.spanning(1900, 1900 + MAX_REPRESENTATIONS + 5))
        assert len(reps) == 1

    def test_the_cap_is_configurable(self):
        reps = encode(TemporalInterval.spanning(2017, 2022), max_representations=3)
        assert len(reps) == 1

    def test_a_point_is_never_split(self):
        reps = encode(TemporalInterval.point(utc(2021, 6, 1)))
        assert len(reps) == 1
        assert reps[0].is_point

    def test_boundary_moments_are_the_new_years_strictly_inside(self):
        cuts = boundary_moments(
            TemporalInterval.spanning(2017, 2019), QUARTER_SCALE, DEFAULT_HIERARCHY
        )
        assert cuts == [utc(2018, 1, 1), utc(2019, 1, 1)]

    def test_a_span_aligned_to_the_boundary_has_no_interior_cut(self):
        assert boundary_moments(
            TemporalInterval.of_year(2021), QUARTER_SCALE, DEFAULT_HIERARCHY
        ) == []

    def test_encode_single_never_splits(self):
        encoding = encode_single(TemporalInterval.spanning(2017, 2022))
        assert encoding.representation_count == 1

    def test_encoding_round_trips_through_a_dict(self):
        original = encode(TemporalInterval.spanning(2020, 2021), group_id="g")[1]
        restored = TemporalEncoding.from_dict(original.to_dict())
        assert restored.to_vector() == pytest.approx(original.to_vector())
        assert restored.group_id == "g"
        assert restored.representation_index == 1
        assert restored.representation_count == 2
        assert restored.hierarchy.fingerprint() == original.hierarchy.fingerprint()
        assert restored.interval.start == original.interval.start
        assert restored.source_interval.end == original.source_interval.end

    def test_the_header_declares_how_many_tuples_follow(self):
        """
        The header is what makes the vector self-describing: a reader strips exactly
        ``3 * tuple_count`` trailing dimensions rather than assuming nine.
        """
        encoding = encode_single(TemporalInterval.of_year(2021))
        header = encoding.header()
        assert header["tuple_count"] == len(encoding.tuples)
        assert len(header["scales"]) == header["tuple_count"]
        assert header["epoch"].startswith("1900-01-01")
        assert header["year_convention"] == "calendar"


# ---------------------------------------------------------------------------
# Neutral padding
# ---------------------------------------------------------------------------


class TestPadding:
    def test_a_neutral_tuple_is_a_full_circle_at_phase_zero(self):
        t = neutral_tuple(MILLENNIUM_SCALE)
        assert (t.cos, t.sin) == (1.0, 0.0)
        assert t.z == pytest.approx(TAU)
        assert t.is_full_circle
        assert t.segments == tuple(range(MILLENNIUM_SCALE.segments))

    def test_padding_widens_an_old_vector_to_a_longer_hierarchy(self):
        extended = DEFAULT_HIERARCHY.extended(MILLENNIUM_SCALE)
        old = encode_single(TemporalInterval.of_year(2021), DEFAULT_HIERARCHY)
        padded = pad_to_hierarchy(old, extended)
        assert padded.dimensions == 12
        assert padded.scale_names == ("quarter", "decade", "century", "millennium")

    def test_padding_leaves_the_existing_tuples_untouched(self):
        extended = DEFAULT_HIERARCHY.extended(MILLENNIUM_SCALE)
        old = encode_single(TemporalInterval.of_year(2021), DEFAULT_HIERARCHY)
        padded = pad_to_hierarchy(old, extended)
        assert padded.to_vector()[:9] == pytest.approx(old.to_vector())

    def test_a_padded_vector_neither_matches_nor_blocks_on_the_new_scale(self):
        extended = DEFAULT_HIERARCHY.extended(MILLENNIUM_SCALE)
        old = pad_to_hierarchy(
            encode_single(TemporalInterval.of_year(2021), DEFAULT_HIERARCHY), extended
        )
        fresh = encode_single(TemporalInterval.of_year(2021), extended)
        match = evaluate_scale(
            fresh.tuple_for("millennium"),
            old.tuple_for("millennium"),
            MILLENNIUM_SCALE,
        )
        assert match.overlaps  # a full circle overlaps everything

    def test_padding_across_an_incompatible_hierarchy_raises(self):
        other = TemporalHierarchy(epoch=utc(2010, 1, 1))
        old = encode_single(TemporalInterval.of_year(2021), DEFAULT_HIERARCHY)
        with pytest.raises(ValueError, match="incompatible"):
            pad_to_hierarchy(old, other.extended(MILLENNIUM_SCALE))


# ---------------------------------------------------------------------------
# Lazy traversal
# ---------------------------------------------------------------------------


class TestTraversalPlan:
    def test_a_point_query_traverses_every_circle(self):
        plan = traversal_plan(encode_single(TemporalInterval.point(utc(2021, 5, 17))))
        assert plan.scale_names == ("century", "decade", "quarter")
        assert plan.skipped == ()

    def test_the_plan_is_ordered_coarsest_first(self):
        """The widest, cheapest rejection should happen before any fine work."""
        plan = traversal_plan(encode_single(TemporalInterval.of_quarter(2021, 2)))
        assert plan.scale_names == ("century", "decade", "quarter")
        assert plan.depth == 3

    def test_a_full_year_query_skips_the_one_year_circle(self):
        """Every document in any year saturates it, so the check cannot reject."""
        plan = traversal_plan(encode_single(TemporalInterval.of_year(2021)))
        assert plan.scale_names == ("century", "decade")
        assert [name for name, _ in plan.skipped] == ["quarter"]
        assert "saturates" in dict(plan.skipped)["quarter"]

    def test_a_multi_year_query_also_skips_the_one_year_circle(self):
        plan = traversal_plan(encode_single(TemporalInterval.spanning(2017, 2022)))
        assert plan.scale_names == ("century", "decade")

    def test_an_era_wide_query_keeps_only_the_outermost_circle(self):
        plan = traversal_plan(encode_single(TemporalInterval.spanning(2000, 2100)))
        assert plan.scale_names == ("century",)
        assert {name for name, _ in plan.skipped} == {"quarter", "decade"}

    def test_a_query_saturating_everything_still_keeps_one_axis(self):
        plan = traversal_plan(encode_single(TemporalInterval.spanning(1900, 2400)))
        assert plan.scale_names == ("century",)
        assert "century" not in dict(plan.skipped)

    def test_membership_and_iteration(self):
        plan = traversal_plan(encode_single(TemporalInterval.of_year(2021)))
        assert "decade" in plan
        assert "quarter" not in plan
        assert list(plan) == ["century", "decade"]

    def test_traversal_depth_is_independent_of_encoded_precision(self):
        """
        Both queries are stored at full nine-dimensional precision; only the number
        of circles retrieval descends through differs.
        """
        fine = encode_single(TemporalInterval.of_quarter(2021, 2))
        coarse = encode_single(TemporalInterval.of_year(2021))
        assert fine.dimensions == coarse.dimensions == 9
        assert traversal_plan(fine).depth > traversal_plan(coarse).depth

    def test_a_scale_missing_from_the_query_is_skipped_with_a_reason(self):
        extended = DEFAULT_HIERARCHY.extended(MILLENNIUM_SCALE)
        narrow = encode_single(TemporalInterval.of_quarter(2021, 2), DEFAULT_HIERARCHY)
        plan = traversal_plan(narrow, extended)
        assert "millennium" not in plan
        assert dict(plan.skipped)["millennium"] == "absent from query encoding"


# ---------------------------------------------------------------------------
# Scale matching
# ---------------------------------------------------------------------------


class TestEvaluateScale:
    def _quarter(self, year, q):
        return encode_single(TemporalInterval.of_quarter(year, q)).tuple_for("quarter")

    def test_identical_arcs_match_perfectly(self):
        t = self._quarter(2023, 2)
        match = evaluate_scale(t, t, QUARTER_SCALE)
        assert match.overlaps
        assert match.jaccard == pytest.approx(1.0)
        assert match.delta_phi == pytest.approx(0.0)

    def test_sibling_quarters_do_not_overlap(self):
        match = evaluate_scale(self._quarter(2023, 1), self._quarter(2023, 3), QUARTER_SCALE)
        assert not match.overlaps
        assert match.jaccard == pytest.approx(0.0)

    def test_the_same_quarter_in_different_years_collides_on_the_one_year_circle(self):
        """
        Q2 is Q2 in every year. Separation is the 16-year circle's job — this is the
        reason a single timestamp dimension cannot answer "which year".
        """
        match = evaluate_scale(self._quarter(2021, 2), self._quarter(2024, 2), QUARTER_SCALE)
        assert match.overlaps
        assert match.jaccard == pytest.approx(1.0, abs=0.02)

    def test_and_is_separated_on_the_sixteen_year_circle(self):
        a = encode_single(TemporalInterval.of_quarter(2021, 2)).tuple_for("decade")
        b = encode_single(TemporalInterval.of_quarter(2024, 2)).tuple_for("decade")
        assert not evaluate_scale(a, b, DECADE_SCALE).overlaps

    def test_a_quarter_sits_inside_its_year(self):
        year = encode_single(TemporalInterval.of_year(2023)).tuple_for("quarter")
        quarter = self._quarter(2023, 2)
        match = evaluate_scale(year, quarter, QUARTER_SCALE)
        assert match.overlaps
        # Whole-arc Jaccard is only 91/365 — the quarter is a small slice of the year.
        assert jaccard_arcs(
            year.phi_start, year.z, quarter.phi_start, quarter.z
        ) == pytest.approx(91 / 365, abs=0.01)
        # Restricted to the Q2 segment they very nearly coincide, and a match on any
        # shared segment is what qualifies the document. Not exactly 1.0: the segment
        # window is the even [90°, 180°) arc, while Q2 2023 actually runs from 88.77°
        # to 178.53°, so it fills 88.53 of the window's 90 degrees.
        assert match.shared_segments == (1,)
        assert match.segment_jaccard[1] == pytest.approx(0.9836, abs=1e-4)
        assert match.jaccard == pytest.approx(0.9836, abs=1e-4)

    def test_a_point_inside_an_arc_overlaps(self):
        point = encode_single(
            TemporalInterval.point(utc(2023, 5, 10))
        ).tuple_for("quarter")
        quarter = self._quarter(2023, 2)
        assert evaluate_scale(quarter, point, QUARTER_SCALE).overlaps
        assert evaluate_scale(point, quarter, QUARTER_SCALE).overlaps

    def test_a_point_outside_an_arc_does_not(self):
        point = encode_single(
            TemporalInterval.point(utc(2023, 11, 5))
        ).tuple_for("quarter")
        assert not evaluate_scale(self._quarter(2023, 2), point, QUARTER_SCALE).overlaps

    def test_two_distinct_points_do_not_overlap(self):
        a = encode_single(TemporalInterval.point(utc(2023, 5, 10))).tuple_for("quarter")
        b = encode_single(TemporalInterval.point(utc(2023, 5, 11))).tuple_for("quarter")
        assert not evaluate_scale(a, b, QUARTER_SCALE).overlaps

    def test_two_identical_points_do_overlap(self):
        a = encode_single(TemporalInterval.point(utc(2023, 5, 10))).tuple_for("quarter")
        assert evaluate_scale(a, a, QUARTER_SCALE).overlaps
