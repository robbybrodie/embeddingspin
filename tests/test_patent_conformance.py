"""
Conformance with the specification's worked examples and stated guarantees.

Everything here is pinned to a number or a property written down in the patent
application, so a failure means the implementation has drifted from the filing
rather than merely from a previous release.

The worked example was computed against a 2010-01-01 epoch. The shipped default is
1900-01-01, which gives coverage to 2156 instead of 2266 but starts the corpus
inside the representable window rather than ahead of it. Both hierarchies are
exercised: ``PATENT_EXAMPLE_HIERARCHY`` reproduces the published figures,
``DEFAULT_HIERARCHY`` is what the system actually runs on.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest

from temporal_config import (
    DEFAULT_HIERARCHY,
    EPOCH_2010_LEGACY,
    PATENT_EXAMPLE_HIERARCHY,
    TAU,
    floored_mod,
)
from temporal_encoding import (
    TemporalInterval,
    encode,
    encode_single,
    evaluate_scale,
    traversal_plan,
)


def utc(*args) -> datetime:
    return datetime(*args, tzinfo=timezone.utc)


Q1_2026 = TemporalInterval.of_quarter(2026, 1)


# ---------------------------------------------------------------------------
# Formula 1
# ---------------------------------------------------------------------------


class TestFormula1:
    """``phi = 2 * pi * fmod_floored((t - t0) / T, 1.0)``."""

    def test_the_modulo_is_applied_before_multiplying_by_two_pi(self):
        """
        The reduction is dimensionless: it acts on the period ratio, not on an
        already-scaled angle. Checked by reproducing the formula longhand.
        """
        from temporal_config import years_since_epoch
        from temporal_encoding import phase_of

        moment = utc(2026, 2, 15)
        for scale in DEFAULT_HIERARCHY.scales:
            elapsed = years_since_epoch(moment, DEFAULT_HIERARCHY.epoch, "calendar")
            expected = TAU * floored_mod(elapsed / scale.period_years, 1.0)
            assert phase_of(moment, scale, DEFAULT_HIERARCHY) == pytest.approx(
                expected, abs=1e-12
            )

    def test_the_remainder_is_non_negative_and_below_the_modulus(self):
        """
        Corrects the filing's "always positive" phrasing: the remainder is
        non-negative, and it is exactly zero — not the full modulus — on a boundary.
        """
        assert floored_mod(0.0, 1.0) == 0.0
        assert floored_mod(1.0, 1.0) == 0.0
        assert floored_mod(-1.0, 1.0) == 0.0
        for i in range(-1000, 1000):
            r = floored_mod(i / 13.0, 1.0)
            assert 0.0 <= r < 1.0


# ---------------------------------------------------------------------------
# The Q1 2026 worked example
# ---------------------------------------------------------------------------


class TestQ1_2026WorkedExample:
    def test_the_interval_is_half_open_and_ninety_days_long(self):
        """
        [1 Jan, 1 Apr) — 90 days, not 91. The half-open convention is applied without
        exception, which is what makes the midpoint 15 February.
        """
        assert Q1_2026.start == utc(2026, 1, 1)
        assert Q1_2026.end == utc(2026, 4, 1)
        assert Q1_2026.duration_days == 90.0

    def test_the_midpoint_is_february_fifteenth(self):
        assert Q1_2026.midpoint == utc(2026, 2, 15)

    def test_forty_five_elapsed_days_reach_the_midpoint(self):
        assert (Q1_2026.midpoint - Q1_2026.start).days == 45

    @pytest.mark.parametrize(
        "scale_name,cos,sin,z",
        [
            ("quarter", 0.7147, 0.6995, 1.5493),
            ("decade", 0.9988, 0.0484, 0.0968),
            ("century", 0.9227, 0.3855, 0.0061),
        ],
    )
    def test_published_tuple_values(self, scale_name, cos, sin, z):
        encoding = encode_single(Q1_2026, PATENT_EXAMPLE_HIERARCHY)
        t = encoding.tuple_for(scale_name)
        assert t.cos == pytest.approx(cos, abs=5e-5)
        assert t.sin == pytest.approx(sin, abs=5e-5)
        assert t.z == pytest.approx(z, abs=5e-5)

    def test_the_arc_length_is_the_fraction_of_the_period(self):
        encoding = encode_single(Q1_2026, PATENT_EXAMPLE_HIERARCHY)
        for scale in PATENT_EXAMPLE_HIERARCHY.scales:
            expected = TAU * (90 / 365) / scale.period_years
            assert encoding.tuple_for(scale.name).z == pytest.approx(expected, abs=1e-9)

    def test_the_example_epoch_is_2010(self):
        assert PATENT_EXAMPLE_HIERARCHY.epoch == EPOCH_2010_LEGACY

    def test_the_one_year_tuple_is_epoch_independent(self):
        """
        Both epochs fall on 1 January, so a within-year phase is the same under
        either. Only the 16-year and 256-year tuples move.
        """
        published = encode_single(Q1_2026, PATENT_EXAMPLE_HIERARCHY).tuple_for("quarter")
        shipped = encode_single(Q1_2026, DEFAULT_HIERARCHY).tuple_for("quarter")
        assert shipped.cos == pytest.approx(published.cos)
        assert shipped.sin == pytest.approx(published.sin)
        assert shipped.z == pytest.approx(published.z)

    def test_the_coarser_tuples_do_move_with_the_epoch(self):
        published = encode_single(Q1_2026, PATENT_EXAMPLE_HIERARCHY).tuple_for("decade")
        shipped = encode_single(Q1_2026, DEFAULT_HIERARCHY).tuple_for("decade")
        assert shipped.phi_center != pytest.approx(published.phi_center, abs=1e-6)


# ---------------------------------------------------------------------------
# Structure of the temporal vector
# ---------------------------------------------------------------------------


class TestVectorStructure:
    def test_three_circles_give_nine_dimensions(self):
        assert DEFAULT_HIERARCHY.dimensions == 9
        assert len(encode_single(Q1_2026).to_vector()) == 9

    def test_periods_are_one_sixteen_and_two_hundred_fifty_six_years(self):
        assert DEFAULT_HIERARCHY.periods == (1, 16, 256)

    def test_z_is_zero_for_a_point_and_positive_for_a_duration(self):
        point = encode_single(TemporalInterval.point(utc(2026, 2, 15)))
        arc = encode_single(Q1_2026)
        assert all(t.z == 0.0 for t in point.tuples)
        assert all(t.z > 0.0 for t in arc.tuples)

    def test_an_arc_is_capped_at_one_full_revolution(self):
        for t in encode_single(TemporalInterval.spanning(1950, 2050)).tuples:
            assert t.z <= TAU + 1e-12
        assert encode_single(
            TemporalInterval.spanning(1950, 2050)
        ).tuple_for("quarter").z == pytest.approx(TAU)

    def test_coverage_runs_to_2156_under_the_shipped_epoch(self):
        assert DEFAULT_HIERARCHY.coverage_end_year == 2156

    def test_coverage_runs_to_2266_under_the_example_epoch(self):
        assert PATENT_EXAMPLE_HIERARCHY.coverage_end_year == 2266


# ---------------------------------------------------------------------------
# Zero-degree boundary splitting
# ---------------------------------------------------------------------------


class TestBoundarySplitting:
    def test_2017_to_2022_yields_six_representations(self):
        assert len(encode(TemporalInterval.spanning(2017, 2022))) == 6

    def test_all_six_share_one_group_id(self):
        reps = encode(TemporalInterval.spanning(2017, 2022), group_id="strategic-review")
        assert {r.group_id for r in reps} == {"strategic-review"}
        assert {r.representation_count for r in reps} == {6}

    def test_splitting_happens_at_the_finest_scale_only(self):
        """
        Each period divides the next, so the 1-year boundaries are a superset of the
        16-year and 256-year ones. One pass catches every crossing.
        """
        reps = encode(TemporalInterval.spanning(2017, 2022))
        assert [r.interval.start.year for r in reps] == list(range(2017, 2023))

    def test_a_span_ending_mid_year_keeps_its_partial_arc(self):
        reps = encode(TemporalInterval(utc(2021, 1, 1), utc(2022, 4, 1)))
        assert len(reps) == 2
        assert reps[0].tuple_for("quarter").z == pytest.approx(TAU)
        assert reps[1].tuple_for("quarter").z < TAU


# ---------------------------------------------------------------------------
# Lazy resolution
# ---------------------------------------------------------------------------


class TestLazyResolution:
    def test_a_saturating_query_skips_that_circle(self):
        plan = traversal_plan(encode_single(TemporalInterval.of_year(2021)))
        assert "quarter" not in plan

    def test_traversal_is_coarsest_first(self):
        plan = traversal_plan(encode_single(TemporalInterval.of_quarter(2021, 2)))
        periods = [DEFAULT_HIERARCHY.scale(n).period_years for n in plan.scale_names]
        assert periods == sorted(periods, reverse=True)

    def test_encoding_precision_and_traversal_depth_are_separate(self):
        """
        The filing distinguishes the number of scales encoded (fixed at ingestion)
        from the number traversed at query time. Both queries below are stored at the
        same precision.
        """
        fine = encode_single(TemporalInterval.of_quarter(2021, 2))
        coarse = encode_single(TemporalInterval.spanning(2000, 2100))
        assert fine.dimensions == coarse.dimensions
        assert traversal_plan(fine).depth == 3
        assert traversal_plan(coarse).depth == 1


# ---------------------------------------------------------------------------
# Extensibility
# ---------------------------------------------------------------------------


class TestExtensibility:
    def test_appending_an_outer_scale_preserves_every_existing_phase(self):
        from temporal_config import MILLENNIUM_SCALE

        extended = DEFAULT_HIERARCHY.extended(MILLENNIUM_SCALE)
        before = encode_single(Q1_2026, DEFAULT_HIERARCHY)
        after = encode_single(Q1_2026, extended)
        assert after.to_vector()[:9] == pytest.approx(before.to_vector())

    def test_extension_is_not_free_if_the_epoch_moves(self):
        """
        The filing's extensibility claim is conditional: phases survive only because
        the epoch is held fixed. Changing it is a re-index.
        """
        from temporal_config import describe_epoch_migration

        report = describe_epoch_migration(PATENT_EXAMPLE_HIERARCHY, DEFAULT_HIERARCHY)
        assert report["requires_reindex"] is True


# ---------------------------------------------------------------------------
# Arbitrary scale count
# ---------------------------------------------------------------------------


class TestArbitraryScaleCount:
    """
    The claims recite "a first" and "a second" periodic scale, and claim 2 adds a
    third. Nothing caps the count, so three circles are the shipped default rather
    than the method. Everything downstream has to read the count from the hierarchy
    instead of assuming it: the vector width, the split point, the traversal order,
    the weighting and the coverage horizon.
    """

    @staticmethod
    def _hierarchy(n):
        from temporal_config import (
            CENTURY_SCALE, DECADE_SCALE, MILLENNIUM_SCALE, QUARTER_SCALE, ScaleSpec,
            TemporalHierarchy,
        )

        eon = ScaleSpec(name="eon", period_years=65536, segments=16, weight=0.05,
                        segment_label="4096-year block")
        available = (QUARTER_SCALE, DECADE_SCALE, CENTURY_SCALE, MILLENNIUM_SCALE, eon)
        return TemporalHierarchy(scales=available[:n])

    def test_a_hierarchy_needs_at_least_one_scale(self):
        from temporal_config import TemporalHierarchy

        with pytest.raises(ValueError, match="at least one scale"):
            TemporalHierarchy(scales=())

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
    def test_the_vector_is_three_dimensions_per_scale(self, n):
        hierarchy = self._hierarchy(n)
        encoding = encode_single(Q1_2026, hierarchy)
        assert hierarchy.dimensions == 3 * n
        assert len(encoding.to_vector()) == 3 * n

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
    def test_the_header_declares_the_count_rather_than_assuming_it(self, n):
        header = encode_single(Q1_2026, self._hierarchy(n)).header()
        assert header["tuple_count"] == n
        assert len(header["scales"]) == n

    @pytest.mark.parametrize("n, coverage", [(2, 1916), (3, 2156), (4, 5996)])
    def test_coverage_is_set_by_the_outermost_circle(self, n, coverage):
        """Adding a circle is how you buy range; it is not a free precision gain."""
        assert self._hierarchy(n).coverage_end_year == coverage

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
    def test_traversal_runs_coarsest_to_finest_whatever_the_depth(self, n):
        hierarchy = self._hierarchy(n)
        plan = traversal_plan(encode_single(Q1_2026, hierarchy), hierarchy)
        expected = tuple(s.name for s in reversed(hierarchy.scales))
        assert plan.scale_names == expected

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
    def test_splitting_follows_the_finest_scale_whatever_it_is(self, n):
        hierarchy = self._hierarchy(n)
        reps = encode(TemporalInterval.spanning(2017, 2022), hierarchy)
        # The finest circle is the 1-year one in every configuration above, so the
        # six-year span splits into six regardless of how many circles sit above it.
        assert len(reps) == 6
        assert len({r.group_id for r in reps}) == 1

    def test_a_four_circle_corpus_is_searchable_end_to_end(self):
        """
        The generalisation has to survive the whole stack, not just the encoder —
        storage splits the vector on the header, and the retriever weights over
        whichever scales it traversed.
        """
        from ingestion import TemporalSpinIngestionPipeline
        from llamastack_client import MockEmbeddingClient
        from retrieval import TemporalSpinRetriever
        from vector_store import InMemoryVectorStore

        hierarchy = self._hierarchy(4)
        client = MockEmbeddingClient(dimension=32)
        store = InMemoryVectorStore(hierarchy=hierarchy)
        pipeline = TemporalSpinIngestionPipeline(
            embedding_client=client, vector_store=store, hierarchy=hierarchy
        )
        for year in (2021, 2022, 2023):
            pipeline.ingest_document(
                f"IBM annual report {year}.",
                interval=TemporalInterval.of_year(year),
                doc_id=f"annual-{year}",
            )

        stored = store.get_document("annual-2022")
        assert len(stored.full_embedding) == 32 + 12

        retriever = TemporalSpinRetriever(
            client, store, hierarchy=hierarchy, default_beta=0.5
        )
        results = retriever.search("report", interval=TemporalInterval.of_year(2022))
        assert [r.doc_id for r in results] == ["annual-2022"]
        assert results[0].traversed_scales == ("millennium", "century", "decade")

    def test_a_shorter_hierarchy_can_read_a_longer_corpus(self):
        """
        Prefix compatibility: appending an outer circle leaves every existing phase
        untouched, so the two configurations can be compared without re-encoding.
        A different epoch cannot.
        """
        assert self._hierarchy(3).is_compatible_with(self._hierarchy(4))
        assert not PATENT_EXAMPLE_HIERARCHY.is_compatible_with(DEFAULT_HIERARCHY)


# ---------------------------------------------------------------------------
# Retrieval semantics
# ---------------------------------------------------------------------------


class TestRetrievalSemantics:
    def test_beta_is_bounded_to_the_unit_interval(self):
        """
        β blends two scores that are both in [0, 1]; values outside that range are a
        configuration error, not a stronger preference.
        """
        from llamastack_client import MockEmbeddingClient
        from retrieval import TemporalSpinRetriever
        from vector_store import InMemoryVectorStore

        client = MockEmbeddingClient(dimension=32)
        store = InMemoryVectorStore(hierarchy=DEFAULT_HIERARCHY)
        for bad in (-0.1, 1.5, 5000.0):
            with pytest.raises(ValueError, match=r"\[0, 1\]"):
                TemporalSpinRetriever(client, store, default_beta=bad)

    def test_the_same_quarter_in_two_years_is_separated_by_the_coarser_circle(self):
        a = encode_single(TemporalInterval.of_quarter(2021, 1))
        b = encode_single(TemporalInterval.of_quarter(2024, 1))
        quarter = evaluate_scale(
            a.tuple_for("quarter"), b.tuple_for("quarter"), DEFAULT_HIERARCHY.scale("quarter")
        )
        decade = evaluate_scale(
            a.tuple_for("decade"), b.tuple_for("decade"), DEFAULT_HIERARCHY.scale("decade")
        )
        assert quarter.overlaps        # indistinguishable on the 1-year circle
        assert not decade.overlaps     # resolved on the 16-year circle

    def test_a_multi_year_document_matches_through_its_component_year(self):
        review = encode(TemporalInterval.spanning(2017, 2022))
        query = encode_single(TemporalInterval.of_year(2021))
        matching = [
            rep for rep in review
            if evaluate_scale(
                query.tuple_for("decade"),
                rep.tuple_for("decade"),
                DEFAULT_HIERARCHY.scale("decade"),
            ).overlaps
        ]
        assert len(matching) == 1
        assert matching[0].interval.start.year == 2021
