"""
Segments: even geometry, calendar identity.

Two separate things wear the word "segment" here, and the split is the point.

The **geometry** is even, always. Every circle is divided into equal arcs — four of
90 degrees on the 1-year circle, sixteen of 22.5 on the others — because periods and
segment counts are powers of two and an index is then ``int(fraction * segments)``:
one multiply, no divider table, no search, no data-dependent branch. That is what
makes the arithmetic vectorise across a batch of candidates.

The **calendar** is not even. Quarters run 90, 91, 92 and 92 days, and a leap day
shifts everything after February. None of that is permitted into the geometry. Which
calendar segment an interval occupies is resolved from the real dates, once, at
encoding time, and stored on the tuple — integer arithmetic on months and years,
with no divider to approximate and so no tolerance to tune.

So ``segment_of_phase`` and ``calendar_segment`` deliberately disagree near a
divider: 1 April sits at 88.8 degrees, which is geometrically Q1 and calendrically
Q2. The encoder uses the calendar one; scoring, which holds only phases, uses the
even windows.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from temporal_config import (
    CENTURY_SCALE,
    DECADE_SCALE,
    DEFAULT_HIERARCHY,
    QUARTER_SCALE,
    TAU,
)
from temporal_encoding import (
    TemporalInterval,
    calendar_segment,
    calendar_segments_touched,
    encode_single,
    evaluate_scale,
    phase_of,
    segment_bounds,
    segment_label,
    segment_of_phase,
    segments_intersected,
)


def utc(*args) -> datetime:
    return datetime(*args, tzinfo=timezone.utc)


def quarter_segments(interval: TemporalInterval):
    return encode_single(interval, DEFAULT_HIERARCHY).tuple_for("quarter").segments


# ---------------------------------------------------------------------------
# Divider geometry
# ---------------------------------------------------------------------------


class TestDividerGeometry:
    def test_every_circle_is_divided_evenly(self):
        for scale in DEFAULT_HIERARCHY.scales:
            widths = {
                round(segment_bounds(scale, i)[1], 12) for i in range(scale.segments)
            }
            assert widths == {round(TAU / scale.segments, 12)}, scale.name

    def test_the_quarter_circle_is_four_ninety_degree_arcs(self):
        widths = [segment_bounds(QUARTER_SCALE, i)[1] for i in range(4)]
        assert [round(math.degrees(w), 6) for w in widths] == [90.0] * 4

    def test_segment_counts_are_powers_of_two(self):
        """
        What makes an index a shift rather than a search. With the power-of-two
        periods, this is the performance argument for the hierarchy: the calendar is
        kept out so the arithmetic stays exact and branch-free.
        """
        for scale in DEFAULT_HIERARCHY.scales:
            assert scale.segments & (scale.segments - 1) == 0, scale.name

    def test_dividers_are_derived_not_tabulated(self):
        for scale in DEFAULT_HIERARCHY.scales:
            assert scale.boundary_phases == tuple(
                i * TAU / scale.segments for i in range(scale.segments)
            )

    def test_the_calendar_does_not_line_up_with_the_geometry(self):
        """
        1 April is at 88.8 degrees, not 90. That is left alone rather than corrected
        in the geometry; ``calendar_segment`` is where it is answered.
        """
        phi = phase_of(utc(2023, 4, 1), QUARTER_SCALE, DEFAULT_HIERARCHY)
        assert math.degrees(phi) == pytest.approx(88.77, abs=0.01)
        assert segment_of_phase(QUARTER_SCALE, phi) == 0               # geometric
        assert calendar_segment(utc(2023, 4, 1), QUARTER_SCALE) == 1   # calendar

    def test_segments_tile_each_circle_exactly(self):
        for scale in DEFAULT_HIERARCHY.scales:
            spans = [segment_bounds(scale, i) for i in range(scale.segments)]
            assert spans[0][0] == 0.0
            for (start, length), (next_start, _) in zip(spans, spans[1:]):
                assert start + length == pytest.approx(next_start, abs=1e-12)
            assert sum(length for _, length in spans) == pytest.approx(TAU, abs=1e-12)

    def test_segment_index_wraps(self):
        assert segment_bounds(QUARTER_SCALE, 4) == segment_bounds(QUARTER_SCALE, 0)


# ---------------------------------------------------------------------------
# Locating a single instant
# ---------------------------------------------------------------------------


class TestGeometricIndex:
    """
    ``segment_of_phase`` answers "which quarter of the circle", arithmetically.

    No table, no search: ``int(fraction * segments)``. Away from a divider it agrees
    with the calendar; near one it does not, and that is left alone rather than
    patched, because patching it is what costs the branch-free property.
    """

    @pytest.mark.parametrize(
        "month,expected", [(1, 0), (2, 0), (3, 0), (4, 1), (5, 1), (6, 1),
                           (7, 2), (8, 2), (9, 2), (10, 3), (11, 3), (12, 3)]
    )
    def test_a_mid_month_date_agrees_with_the_calendar(self, month, expected):
        phi = phase_of(utc(2023, month, 15), QUARTER_SCALE, DEFAULT_HIERARCHY)
        assert segment_of_phase(QUARTER_SCALE, phi) == expected

    def test_the_index_is_a_multiply_and_a_truncation(self):
        for fraction, expected in ((0.0, 0), (0.24, 0), (0.25, 1), (0.6, 2),
                                   (0.75, 3), (0.999, 3)):
            assert segment_of_phase(QUARTER_SCALE, fraction * TAU) == expected

    def test_a_phase_at_the_very_top_of_the_circle_does_not_overflow(self):
        assert segment_of_phase(QUARTER_SCALE, TAU - 1e-15) == 3
        assert segment_of_phase(DECADE_SCALE, TAU - 1e-15) == 15

    def test_it_disagrees_with_the_calendar_near_a_divider_by_design(self):
        """
        1 April is geometrically still in the first quarter of the circle, because
        the first calendar quarter is only 90 of 365 days. The encoder does not
        consult this function, so nothing stored is wrong.
        """
        phi = phase_of(utc(2023, 4, 1), QUARTER_SCALE, DEFAULT_HIERARCHY)
        assert segment_of_phase(QUARTER_SCALE, phi) == 0
        assert calendar_segment(utc(2023, 4, 1), QUARTER_SCALE) == 1

    def test_whole_year_segments_agree_exactly(self):
        """
        On the 16-year and 256-year circles there is no disagreement at all: a
        segment is a whole number of years and the year count is an integer, so the
        even divider and the calendar divider are the same place.
        """
        for offset in range(16):
            moment = utc(1900 + offset, 6, 1)
            phi = phase_of(moment, DECADE_SCALE, DEFAULT_HIERARCHY)
            assert segment_of_phase(DECADE_SCALE, phi) == offset
            assert calendar_segment(moment, DECADE_SCALE) == offset


# ---------------------------------------------------------------------------
# Calendar identity — resolved from the dates, not from a phase
# ---------------------------------------------------------------------------


class TestCalendarIdentity:
    """
    ``calendar_segment`` is integer arithmetic on months and years.

    There is no divider to approximate, so the unevenness of the calendar costs
    nothing: quarters of 90/91/92/92 days and the leap day need no special case, no
    tolerance and no second table. This is the correction applied *after* the
    geometry, at the one point where the real dates are still available.
    """

    @pytest.mark.parametrize("year", [1900, 2000, 2023, 2024, 2048, 2052, 2100, 2155])
    def test_every_hour_of_a_year_lands_in_its_calendar_quarter(self, year):
        moment = utc(year, 1, 1)
        end = utc(year + 1, 1, 1)
        while moment < end:
            assert calendar_segment(moment, QUARTER_SCALE) == (
                moment.month - 1
            ) // 3, moment
            moment += timedelta(hours=1)

    def test_the_first_instant_of_each_quarter_opens_it(self):
        for year in (2023, 2024):
            for quarter, month in enumerate((1, 4, 7, 10)):
                assert calendar_segment(utc(year, month, 1), QUARTER_SCALE) == quarter

    def test_the_last_instant_of_each_quarter_stays_in_it(self):
        """The leap-year instants that a phase-space divider used to push forward."""
        for moment, expected in (
            (utc(2024, 3, 31, 23, 59), 0),
            (utc(2024, 6, 30, 23, 59), 1),
            (utc(2024, 9, 30, 23, 59), 2),
            (utc(2024, 12, 31, 23, 59), 3),
        ):
            assert calendar_segment(moment, QUARTER_SCALE) == expected, moment

    def test_a_year_maps_to_its_slot_on_the_coarser_circles(self):
        for year in range(1900, 2157):
            moment = utc(year, 6, 1)
            assert calendar_segment(moment, DECADE_SCALE) == (year - 1900) % 16
            assert calendar_segment(moment, CENTURY_SCALE) == ((year - 1900) // 16) % 16

    def test_the_encoder_stores_the_calendar_answer(self):
        point = TemporalInterval.point(utc(2024, 3, 31, 18))
        assert quarter_segments(point) == (0,)
        assert segment_label(QUARTER_SCALE, quarter_segments(point)[0]) == "Q1"

    def test_every_month_and_quarter_across_the_coverage_window(self):
        for year in range(1900, 2157):
            for month in range(1, 13):
                assert quarter_segments(TemporalInterval.of_month(year, month)) == (
                    (month - 1) // 3,
                ), (year, month)
            for quarter in range(1, 5):
                assert quarter_segments(
                    TemporalInterval.of_quarter(year, quarter)
                ) == (quarter - 1,), (year, quarter)

    def test_a_span_ending_on_a_divider_does_not_touch_the_next_segment(self):
        """Half-open, like everything else here."""
        interval = TemporalInterval(utc(2024, 1, 1), utc(2024, 4, 1))
        assert calendar_segments_touched(interval, QUARTER_SCALE) == (0,)

    def test_a_span_is_reported_in_circle_order_through_the_wrap(self):
        interval = TemporalInterval(utc(2023, 11, 1), utc(2024, 2, 1))
        assert calendar_segments_touched(interval, QUARTER_SCALE) == (3, 0)

    def test_a_full_period_touches_every_segment(self):
        assert calendar_segments_touched(
            TemporalInterval.of_year(2024), QUARTER_SCALE
        ) == (0, 1, 2, 3)
        assert calendar_segments_touched(
            TemporalInterval.spanning(2017, 2022), DECADE_SCALE
        ) == (5, 6, 7, 8, 9, 10)

    def test_a_sub_year_segmentation_that_is_whole_months_generalises(self):
        """Quarters are the case that matters, but the expression is not special."""
        from temporal_config import ScaleSpec

        months = ScaleSpec(name="m", period_years=1, segments=12, weight=1.0)
        halves = ScaleSpec(name="h", period_years=1, segments=2, weight=1.0)
        for month in range(1, 13):
            moment = utc(2024, month, 20)
            assert calendar_segment(moment, months) == month - 1
            assert calendar_segment(moment, halves) == (month - 1) // 6


# ---------------------------------------------------------------------------
# Which segments an arc touches
# ---------------------------------------------------------------------------


class TestSegmentsIntersected:
    def test_a_quarter_touches_exactly_its_own_segment(self):
        for quarter in (1, 2, 3, 4):
            assert quarter_segments(TemporalInterval.of_quarter(2023, quarter)) == (
                quarter - 1,
            )

    def test_this_holds_in_leap_years_too(self):
        """
        Q3 2024 ends one day further round the circle than Q3 2023 does. Without the
        divider tolerance it would report itself as spilling into Q4.
        """
        for quarter in (1, 2, 3, 4):
            assert quarter_segments(TemporalInterval.of_quarter(2024, quarter)) == (
                quarter - 1,
            )

    def test_quarters_stay_in_one_segment_across_seventy_years(self):
        for year in range(1990, 2061):
            for quarter in (1, 2, 3, 4):
                assert quarter_segments(
                    TemporalInterval.of_quarter(year, quarter)
                ) == (quarter - 1,), (year, quarter)

    def test_a_full_year_touches_every_segment(self):
        for year in range(1990, 2061):
            assert quarter_segments(TemporalInterval.of_year(year)) == (0, 1, 2, 3)

    def test_an_interval_straddling_a_divider_touches_both_sides(self):
        interval = TemporalInterval(utc(2023, 2, 1), utc(2023, 5, 1))
        assert quarter_segments(interval) == (0, 1)

    def test_a_half_year_touches_two_quarters(self):
        first = TemporalInterval(utc(2024, 1, 1), utc(2024, 7, 1))
        second = TemporalInterval(utc(2024, 7, 1), utc(2025, 1, 1))
        assert quarter_segments(first) == (0, 1)
        assert quarter_segments(second) == (2, 3)

    def test_a_wrapping_interval_is_reported_in_circle_order(self):
        """November to February crosses zero degrees, so Q4 comes before Q1."""
        interval = TemporalInterval(utc(2023, 11, 1), utc(2024, 2, 1))
        assert quarter_segments(interval) == (3, 0)

    def test_a_ten_month_interval_touches_all_four(self):
        interval = TemporalInterval(utc(2024, 2, 1), utc(2024, 11, 1))
        assert quarter_segments(interval) == (0, 1, 2, 3)

    def test_a_point_touches_exactly_one_segment(self):
        segments = quarter_segments(TemporalInterval.point(utc(2023, 5, 10)))
        assert segments == (1,)

    def test_a_saturating_arc_touches_everything(self):
        assert segments_intersected(QUARTER_SCALE, 0.0, TAU) == (0, 1, 2, 3)
        assert segments_intersected(QUARTER_SCALE, 2.0, TAU * 3) == (0, 1, 2, 3)

    def test_a_multi_year_document_covers_every_year_on_the_decade_circle(self):
        """
        Unsplit, a 2017-2022 span saturates the 1-year circle and sweeps six of the
        sixteen slots on the 16-year circle.
        """
        tuple_ = encode_single(TemporalInterval.spanning(2017, 2022)).tuple_for("decade")
        assert len(tuple_.segments) == 6


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------


class TestSegmentLabels:
    def test_quarter_labels(self):
        labels = [segment_label(QUARTER_SCALE, i, DEFAULT_HIERARCHY) for i in range(4)]
        assert labels == ["Q1", "Q2", "Q3", "Q4"]

    def test_decade_labels_are_relative_to_the_epoch(self):
        """
        Segment 0 on the 16-year circle is 1900, but also 1916, 1932 and so on. The
        label names the role and gives an example; disambiguating the repeats is the
        256-year circle's job.
        """
        assert segment_label(DECADE_SCALE, 0, DEFAULT_HIERARCHY) == "year +0y (e.g. 1900)"
        assert segment_label(DECADE_SCALE, 15, DEFAULT_HIERARCHY) == "year +15y (e.g. 1915)"

    def test_century_labels_name_sixteen_year_blocks(self):
        assert segment_label(CENTURY_SCALE, 0, DEFAULT_HIERARCHY) == "16-year block +0y"
        assert segment_label(CENTURY_SCALE, 2, DEFAULT_HIERARCHY) == "16-year block +32y"


# ---------------------------------------------------------------------------
# Per-segment scoring
# ---------------------------------------------------------------------------


class TestPerSegmentScoring:
    def test_a_straddling_document_is_kept_by_the_segment_it_shares(self):
        """
        February to May is two thirds outside Q2. Scored as one arc it looks weak;
        scored per segment the Q2 portion is what counts, so the document survives.
        """
        query = encode_single(TemporalInterval.of_quarter(2023, 2)).tuple_for("quarter")
        doc = encode_single(
            TemporalInterval(utc(2023, 2, 1), utc(2023, 5, 1))
        ).tuple_for("quarter")

        match = evaluate_scale(query, doc, QUARTER_SCALE)
        assert match.overlaps
        assert match.shared_segments == (1,)
        assert 0.0 < match.segment_jaccard[1] <= 1.0

    def test_a_document_sharing_no_segment_and_no_overlap_is_rejected(self):
        query = encode_single(TemporalInterval.of_quarter(2023, 1)).tuple_for("quarter")
        doc = encode_single(TemporalInterval.of_quarter(2023, 3)).tuple_for("quarter")
        match = evaluate_scale(query, doc, QUARTER_SCALE)
        assert not match.overlaps
        assert match.shared_segments == ()
        assert match.segment_jaccard == {}

    def test_the_scale_score_is_the_best_of_the_shared_segments(self):
        query = encode_single(TemporalInterval.of_year(2023)).tuple_for("quarter")
        doc = encode_single(
            TemporalInterval(utc(2023, 2, 1), utc(2023, 5, 1))
        ).tuple_for("quarter")
        match = evaluate_scale(query, doc, QUARTER_SCALE)
        assert match.jaccard == pytest.approx(max(match.segment_jaccard.values()))

    def test_a_shared_segment_with_a_point_scores_on_whether_they_meet(self):
        query = encode_single(TemporalInterval.of_quarter(2023, 2)).tuple_for("quarter")
        inside = encode_single(
            TemporalInterval.point(utc(2023, 5, 10))
        ).tuple_for("quarter")
        match = evaluate_scale(query, inside, QUARTER_SCALE)
        assert match.overlaps
        assert match.segment_jaccard[1] == 1.0
