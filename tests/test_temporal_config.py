"""
Epoch, the floored modulo, scale definitions and epoch migration.

These cover the parts of the specification that are stated as guarantees rather
than as behaviour: that the remainder is never negative, that a period boundary
lands exactly on zero, that each period divides the next, and that an epoch change
is only free when it is congruent with every period.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest

from temporal_config import (
    CENTURY_SCALE,
    DECADE_SCALE,
    DEFAULT_HIERARCHY,
    DEFAULT_SCALES,
    EPOCH_1900,
    EPOCH_2010_LEGACY,
    MILLENNIUM_SCALE,
    PATENT_EXAMPLE_HIERARCHY,
    QUARTER_SCALE,
    SCHEMA_VERSION,
    TAU,
    ScaleSpec,
    TemporalHierarchy,
    calendar_year_position,
    describe_epoch_migration,
    epoch_shift_is_congruent,
    floored_mod,
    years_since_epoch,
)


# ---------------------------------------------------------------------------
# Floored modulo
# ---------------------------------------------------------------------------


class TestFlooredMod:
    """The remainder must satisfy ``0 <= r < modulus``, with no exceptions."""

    @pytest.mark.parametrize(
        "value,expected",
        [
            (0.0, 0.0),
            (0.25, 0.25),
            (1.0, 0.0),        # exact boundary -> zero, not one
            (2.5, 0.5),
            (-0.25, 0.75),     # negative input -> positive remainder
            (-1.0, 0.0),
            (-1.25, 0.75),
            (-3.0, 0.0),
        ],
    )
    def test_known_values(self, value, expected):
        assert floored_mod(value, 1.0) == pytest.approx(expected, abs=1e-12)

    def test_never_negative(self):
        for i in range(-500, 500):
            value = i / 37.0
            assert 0.0 <= floored_mod(value, 1.0) < 1.0

    def test_zero_on_every_period_boundary(self):
        for k in range(-20, 21):
            assert floored_mod(k * 16.0, 16.0) == pytest.approx(0.0, abs=1e-12)

    def test_tiny_negative_snaps_to_zero_not_to_the_modulus(self):
        """
        A value a hair below a boundary must not report a remainder of nearly the
        full modulus. Otherwise a timestamp one nanosecond before New Year would
        encode as phase 2π- instead of 0.
        """
        assert floored_mod(-1e-18, 1.0) == pytest.approx(0.0, abs=1e-12)

    def test_non_unit_modulus(self):
        assert floored_mod(-1.0, 256.0) == pytest.approx(255.0)
        assert floored_mod(257.0, 256.0) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Scales
# ---------------------------------------------------------------------------


class TestScaleSpec:
    def test_default_periods_are_powers_of_two(self):
        for spec in DEFAULT_SCALES:
            exponent = math.log2(spec.period_years)
            assert exponent == pytest.approx(round(exponent))

    def test_each_period_divides_the_next(self):
        """
        This is what makes splitting at the finest scale sufficient: a boundary on a
        coarser circle is always also a boundary on a finer one.
        """
        periods = [s.period_years for s in DEFAULT_SCALES]
        for inner, outer in zip(periods, periods[1:]):
            assert outer % inner == pytest.approx(0.0)

    def test_weights_sum_to_one(self):
        assert sum(s.weight for s in DEFAULT_SCALES) == pytest.approx(1.0)

    def test_every_scale_is_divided_evenly(self):
        """
        The calendar is uneven; the geometry is not allowed to be. Segment identity
        is resolved from the dates at encoding time instead — see
        ``temporal_encoding.calendar_segment``.
        """
        for spec in DEFAULT_SCALES:
            widths = {round(spec.segment_span(i)[1], 12) for i in range(spec.segments)}
            assert widths == {round(TAU / spec.segments, 12)}, spec.name

    def test_segment_counts_are_powers_of_two(self):
        """So a segment index is a multiply and a truncation, not a table search."""
        assert QUARTER_SCALE.segments == 4
        assert DECADE_SCALE.segments == 16
        assert CENTURY_SCALE.segments == 16
        for spec in DEFAULT_SCALES:
            assert spec.segments & (spec.segments - 1) == 0, spec.name

    def test_a_scale_carries_no_divider_table(self):
        """
        Dividers are derived from the segment count. There is deliberately nowhere to
        put a calendar exception, because that is what would cost the branch-free
        index.
        """
        for field in ("boundaries", "leap_boundaries", "boundary_tolerance_days"):
            assert not hasattr(QUARTER_SCALE, field), field

    def test_boundary_phases_start_at_zero_and_increase(self):
        for spec in DEFAULT_SCALES:
            phases = spec.boundary_phases
            assert len(phases) == spec.segments
            assert phases[0] == 0.0
            assert list(phases) == sorted(phases)
            assert phases[-1] < TAU

    def test_segment_spans_tile_the_whole_circle(self):
        for spec in DEFAULT_SCALES:
            total = sum(spec.segment_span(i)[1] for i in range(spec.segments))
            assert total == pytest.approx(TAU, abs=1e-12)

    def test_rejects_bad_configuration(self):
        with pytest.raises(ValueError):
            ScaleSpec(name="bad", period_years=0, segments=4, weight=0.5)
        with pytest.raises(ValueError):
            ScaleSpec(name="bad", period_years=1, segments=0, weight=0.5)
        with pytest.raises(ValueError):
            ScaleSpec(name="bad", period_years=1, segments=4, weight=-0.1)
        with pytest.raises(TypeError):
            # There is no divider table to misconfigure any more.
            ScaleSpec(
                name="bad", period_years=1, segments=4, weight=0.5,
                boundaries=(0.0, 0.5),
            )


# ---------------------------------------------------------------------------
# Year conventions
# ---------------------------------------------------------------------------


class TestYearConventions:
    def test_calendar_position_is_integral_on_january_first(self):
        for year in (1900, 1999, 2000, 2023, 2024, 2100):
            position = calendar_year_position(datetime(year, 1, 1, tzinfo=timezone.utc))
            assert position == pytest.approx(float(year), abs=1e-12)

    def test_calendar_position_uses_the_actual_year_length(self):
        """
        Mid-year in a leap year and mid-year in a common year both sit at 0.5, which
        is the point of the convention: the fraction is of *that* year, so 1 January
        never drifts.
        """
        common = calendar_year_position(datetime(2023, 7, 2, 12, tzinfo=timezone.utc))
        leap = calendar_year_position(datetime(2024, 7, 2, tzinfo=timezone.utc))
        assert common - 2023 == pytest.approx(0.5, abs=1e-3)
        assert leap - 2024 == pytest.approx(0.5, abs=1e-3)

    def test_years_since_epoch_is_a_whole_number_across_whole_years(self):
        assert years_since_epoch(
            datetime(2024, 1, 1, tzinfo=timezone.utc), EPOCH_1900, "calendar"
        ) == pytest.approx(124.0, abs=1e-12)

    def test_linear_convention_drifts_where_calendar_does_not(self):
        moment = datetime(2024, 1, 1, tzinfo=timezone.utc)
        calendar = years_since_epoch(moment, EPOCH_1900, "calendar")
        linear = years_since_epoch(moment, EPOCH_1900, "linear")
        assert calendar == pytest.approx(124.0, abs=1e-12)
        assert linear != pytest.approx(124.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Hierarchy
# ---------------------------------------------------------------------------


class TestTemporalHierarchy:
    def test_defaults(self):
        assert DEFAULT_HIERARCHY.epoch == EPOCH_1900
        assert DEFAULT_HIERARCHY.year_convention == "calendar"
        assert DEFAULT_HIERARCHY.names == ("quarter", "decade", "century")
        assert DEFAULT_HIERARCHY.periods == (1, 16, 256)

    def test_nine_dimensions_for_three_scales(self):
        assert DEFAULT_HIERARCHY.dimensions == 9

    def test_finest_and_coarsest(self):
        assert DEFAULT_HIERARCHY.finest.name == "quarter"
        assert DEFAULT_HIERARCHY.coarsest.name == "century"

    def test_coverage_is_an_integer_year(self):
        end = DEFAULT_HIERARCHY.coverage_end_year
        assert isinstance(end, int)
        assert end == 1900 + 256

    def test_covers(self):
        assert DEFAULT_HIERARCHY.covers(datetime(2026, 1, 1, tzinfo=timezone.utc))
        assert not DEFAULT_HIERARCHY.covers(datetime(1899, 1, 1, tzinfo=timezone.utc))
        assert not DEFAULT_HIERARCHY.covers(datetime(2200, 1, 1, tzinfo=timezone.utc))

    def test_scales_must_be_ordered_finest_first(self):
        with pytest.raises(ValueError, match="finest"):
            TemporalHierarchy(scales=(CENTURY_SCALE, QUARTER_SCALE))

    def test_scale_names_must_be_unique(self):
        with pytest.raises(ValueError, match="unique"):
            TemporalHierarchy(scales=(QUARTER_SCALE, QUARTER_SCALE))

    def test_normalized_weights_sum_to_one_over_any_subset(self):
        full = DEFAULT_HIERARCHY.normalized_weights()
        assert sum(full.values()) == pytest.approx(1.0)
        subset = DEFAULT_HIERARCHY.normalized_weights(["decade", "century"])
        assert set(subset) == {"decade", "century"}
        assert sum(subset.values()) == pytest.approx(1.0)

    def test_header_round_trip(self):
        header = DEFAULT_HIERARCHY.header()
        assert header["tuple_count"] == 3
        assert header["schema_version"] == SCHEMA_VERSION
        restored = TemporalHierarchy.from_header(header)
        assert restored.fingerprint() == DEFAULT_HIERARCHY.fingerprint()

    def test_fingerprint_shape(self):
        assert DEFAULT_HIERARCHY.fingerprint() == (
            "v2|1900-01-01|calendar|quarter:1:4+decade:16:16+century:256:16"
        )

    def test_extension_appends_an_outer_scale(self):
        extended = DEFAULT_HIERARCHY.extended(MILLENNIUM_SCALE)
        assert extended.names == ("quarter", "decade", "century", "millennium")
        assert extended.dimensions == 12
        # Every pre-existing tuple keeps its period, so phases are untouched.
        assert extended.periods[:3] == DEFAULT_HIERARCHY.periods

    def test_extension_must_be_coarser(self):
        with pytest.raises(ValueError, match="coarser"):
            DEFAULT_HIERARCHY.extended(QUARTER_SCALE)

    def test_extension_is_compatible_both_ways(self):
        extended = DEFAULT_HIERARCHY.extended(MILLENNIUM_SCALE)
        assert DEFAULT_HIERARCHY.is_compatible_with(extended)
        assert extended.is_compatible_with(DEFAULT_HIERARCHY)

    def test_different_epoch_is_not_compatible(self):
        assert not DEFAULT_HIERARCHY.is_compatible_with(PATENT_EXAMPLE_HIERARCHY)

    def test_different_year_convention_is_not_compatible(self):
        linear = TemporalHierarchy(year_convention="linear")
        assert not DEFAULT_HIERARCHY.is_compatible_with(linear)


# ---------------------------------------------------------------------------
# Epoch migration
# ---------------------------------------------------------------------------


class TestEpochMigration:
    def test_2010_to_1900_requires_a_reindex(self):
        """
        110 years is a multiple of the 1-year period but not of 16 or 256, so every
        decade and century phase in the corpus would move.
        """
        assert not epoch_shift_is_congruent(
            EPOCH_2010_LEGACY, EPOCH_1900, DEFAULT_SCALES, "calendar"
        )

    def test_a_multiple_of_every_period_is_congruent(self):
        new_epoch = datetime(1900 - 256, 1, 1, tzinfo=timezone.utc)
        assert epoch_shift_is_congruent(
            EPOCH_1900, new_epoch, DEFAULT_SCALES, "calendar"
        )

    def test_a_one_year_shift_is_congruent_only_for_the_one_year_circle(self):
        shifted = datetime(1901, 1, 1, tzinfo=timezone.utc)
        assert epoch_shift_is_congruent(EPOCH_1900, shifted, [QUARTER_SCALE], "calendar")
        assert not epoch_shift_is_congruent(EPOCH_1900, shifted, [DECADE_SCALE], "calendar")

    def test_identity_shift_is_congruent(self):
        assert epoch_shift_is_congruent(
            EPOCH_1900, EPOCH_1900, DEFAULT_SCALES, "calendar"
        )

    def test_describe_flags_the_reindex_and_explains_why(self):
        report = describe_epoch_migration(PATENT_EXAMPLE_HIERARCHY, DEFAULT_HIERARCHY)
        assert report["requires_reindex"] is True
        assert report["reasons"]
        assert any("epoch" in reason.lower() for reason in report["reasons"])

    def test_describe_reports_no_reindex_for_a_pure_extension(self):
        report = describe_epoch_migration(
            DEFAULT_HIERARCHY, DEFAULT_HIERARCHY.extended(MILLENNIUM_SCALE)
        )
        assert report["requires_reindex"] is False
