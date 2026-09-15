"""
Natural-language query decomposition.

One question can carry several independent temporal constraints. "What was the Q1
impact on the full year for 2021, 2022 and 2023?" has three anchor years and two
granularities, which is six retrievals, not one. The sub-queries run in parallel
and their results are merged and deduplicated on ``group_id``.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from query_decomposition import MAX_SUBQUERIES, decompose, search_decomposed
from temporal_encoding import TemporalInterval


def utc(*args) -> datetime:
    return datetime(*args, tzinfo=timezone.utc)


REFERENCE = utc(2026, 9, 15)


def labels(text, **kw):
    return [sq.label for sq in decompose(text, reference=REFERENCE, **kw)]


def spans(text, **kw):
    return [
        (sq.interval.start.date().isoformat(),
         sq.interval.end.date().isoformat() if sq.interval and sq.interval.end else None)
        for sq in decompose(text, reference=REFERENCE, **kw)
    ]


# ---------------------------------------------------------------------------
# The headline case
# ---------------------------------------------------------------------------


class TestHeadlineCase:
    QUERY = "What was the Q1 impact on the full year for 2021, 2022 and 2023?"

    def test_three_years_times_two_granularities_is_six_sub_queries(self):
        assert len(decompose(self.QUERY, reference=REFERENCE)) == 6

    def test_the_sub_queries_pair_each_quarter_with_its_year(self):
        assert labels(self.QUERY) == [
            "Q1 2021", "FY2021", "Q1 2022", "FY2022", "Q1 2023", "FY2023",
        ]

    def test_the_intervals_are_half_open_and_correct(self):
        assert spans(self.QUERY)[:2] == [
            ("2021-01-01", "2021-04-01"),
            ("2021-01-01", "2022-01-01"),
        ]

    def test_every_sub_query_keeps_the_original_text(self):
        for sq in decompose(self.QUERY, reference=REFERENCE):
            assert sq.text == self.QUERY

    def test_the_decomposition_reports_itself_as_temporal(self):
        assert decompose(self.QUERY, reference=REFERENCE).is_temporal

    def test_describe_names_every_sub_query(self):
        described = decompose(self.QUERY, reference=REFERENCE).describe()
        for label in ("Q1 2021", "FY2023"):
            assert label in described


# ---------------------------------------------------------------------------
# Lists versus ranges
# ---------------------------------------------------------------------------


class TestListsVersusRanges:
    def test_a_bare_and_joins_a_list_not_a_range(self):
        """
        "2022 and 2023" is two years. Reading it as a span would silently widen the
        question, and would also swallow the preceding items of a comma list.
        """
        assert labels("results for 2022 and 2023") == ["FY2022", "FY2023"]

    def test_between_makes_and_a_range_connector(self):
        assert labels("results between 2017 and 2022") == ["2017-2022"]

    def test_from_x_to_y_is_a_range(self):
        assert labels("revenue from 2019 to 2021") == ["2019-2021"]

    def test_a_dash_is_a_range(self):
        assert labels("revenue 2019-2021") == ["2019-2021"]

    def test_an_en_dash_is_a_range(self):
        assert labels("revenue 2019–2021") == ["2019-2021"]

    def test_through_is_a_range(self):
        assert labels("revenue 2019 through 2021") == ["2019-2021"]

    def test_a_range_label_names_the_inclusive_last_year(self):
        """
        The interval is half-open and ends on 1 Jan 2023, but the range is 2017 to
        2022 — the label must not read off ``end.year``.
        """
        assert labels("between 2017 and 2022") == ["2017-2022"]
        assert spans("between 2017 and 2022") == [("2017-01-01", "2023-01-01")]

    def test_a_range_can_be_expanded_year_by_year(self):
        assert labels("revenue between 2017 and 2022 year by year") == [
            "FY2017", "FY2018", "FY2019", "FY2020", "FY2021", "FY2022",
        ]

    def test_annually_also_expands_a_range(self):
        assert labels("revenue from 2019 to 2021 annually") == [
            "FY2019", "FY2020", "FY2021",
        ]


# ---------------------------------------------------------------------------
# Anchors
# ---------------------------------------------------------------------------


class TestAnchors:
    def test_a_bare_year(self):
        assert labels("IBM revenue in 2021") == ["FY2021"]
        assert spans("IBM revenue in 2021") == [("2021-01-01", "2022-01-01")]

    def test_a_quarter_and_year(self):
        assert labels("IBM Q3 2022 results") == ["Q3 2022"]
        assert spans("IBM Q3 2022 results") == [("2022-07-01", "2022-10-01")]

    def test_a_fiscal_year_quarter(self):
        assert labels("Q4 FY2020 performance") == ["Q4 2020"]

    def test_an_ordinal_quarter(self):
        assert labels("second quarter 2023 revenue") == ["Q2 2023"]

    def test_a_month_and_year(self):
        assert spans("what happened in March 2021") == [("2021-03-01", "2021-04-01")]

    def test_since_runs_to_the_reference_date(self):
        start, end = spans("cloud revenue since 2020")[0]
        assert start == "2020-01-01"
        assert end.startswith("2027")  # through the end of the reference year

    def test_before_runs_up_to_the_named_year(self):
        start, end = spans("cloud revenue before 2015")[0]
        assert end == "2015-01-01"

    def test_last_n_years_is_relative_to_the_reference(self):
        start, end = spans("revenue over the last three years")[0]
        assert start.startswith("2023")
        assert end.startswith("2026")

    def test_a_query_with_no_temporal_language_yields_one_unconstrained_sub_query(self):
        decomposition = decompose("what is IBM's cloud strategy", reference=REFERENCE)
        assert len(decomposition) == 1
        assert decomposition.subqueries[0].interval is None
        assert not decomposition.is_temporal

    def test_the_reference_defaults_to_now(self):
        assert len(decompose("revenue last two years")) == 1


# ---------------------------------------------------------------------------
# Granularities
# ---------------------------------------------------------------------------


class TestGranularities:
    def test_a_bare_quarter_applies_to_every_anchor_year(self):
        assert labels("Q2 for 2021 and 2022") == ["Q2 2021", "Q2 2022"]

    def test_a_bare_ordinal_quarter_applies_too(self):
        assert labels("third quarter for 2021 and 2022") == ["Q3 2021", "Q3 2022"]

    def test_full_year_adds_the_annual_granularity(self):
        assert labels("Q2 and the full year for 2022") == ["Q2 2022", "FY2022"]

    def test_several_quarters_expand_against_each_year(self):
        assert labels("Q1 and Q3 for 2022 and 2023") == [
            "Q1 2022", "Q3 2022", "Q1 2023", "Q3 2023",
        ]


# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------


class TestLimits:
    def test_the_sub_query_count_is_capped(self):
        decomposition = decompose(
            "revenue between 1950 and 2050 year by year", reference=REFERENCE
        )
        assert len(decomposition) <= MAX_SUBQUERIES
        assert decomposition.truncated

    def test_a_custom_cap_is_honoured(self):
        decomposition = decompose(
            "revenue for 2019, 2020, 2021 and 2022", reference=REFERENCE, max_subqueries=2
        )
        assert len(decomposition) == 2
        assert decomposition.truncated

    def test_an_uncapped_decomposition_is_not_flagged(self):
        assert not decompose("revenue in 2021", reference=REFERENCE).truncated


# ---------------------------------------------------------------------------
# Data shape
# ---------------------------------------------------------------------------


class TestDataShape:
    def test_as_pairs_is_what_search_many_consumes(self):
        pairs = decompose("Q1 2021 and Q1 2022", reference=REFERENCE).as_pairs()
        assert all(isinstance(text, str) for text, _ in pairs)
        assert all(
            interval is None or isinstance(interval, TemporalInterval)
            for _, interval in pairs
        )

    def test_a_decomposition_is_iterable_and_sized(self):
        decomposition = decompose("Q1 2021 and Q1 2022", reference=REFERENCE)
        assert len(decomposition) == len(list(decomposition)) == 2

    def test_the_original_text_is_retained(self):
        text = "IBM Q1 2021 results"
        assert decompose(text, reference=REFERENCE).query_text == text


# ---------------------------------------------------------------------------
# Decomposed search
# ---------------------------------------------------------------------------


class TestDecomposedSearch:
    @pytest.fixture
    def corpus(self, pipeline):
        for year in (2021, 2022, 2023):
            pipeline.ingest_document(
                f"IBM annual report {year}: revenue and cloud growth.",
                interval=TemporalInterval.of_year(year),
                doc_id=f"annual-{year}",
            )
            pipeline.ingest_document(
                f"IBM Q1 {year}: first quarter revenue.",
                interval=TemporalInterval.of_quarter(year, 1),
                doc_id=f"q1-{year}",
            )
        pipeline.ingest_document(
            "IBM strategic review 2017-2022.",
            interval=TemporalInterval.spanning(2017, 2022),
            doc_id="review",
        )
        return pipeline

    QUERY = "What was the Q1 impact on the full year for 2021, 2022 and 2023?"

    def test_the_merged_results_cover_every_sub_query(self, corpus, retriever):
        results, decomposition = search_decomposed(
            retriever, self.QUERY, beta=0.5, top_k_final=10, reference=REFERENCE
        )
        assert len(decomposition) == 6
        ids = {r.doc_id for r in results}
        assert {"q1-2021", "q1-2022", "q1-2023"} <= ids
        assert {"annual-2021", "annual-2022", "annual-2023"} <= ids

    def test_each_result_records_which_sub_queries_it_answered(self, corpus, retriever):
        results, _ = search_decomposed(
            retriever, self.QUERY, beta=0.5, top_k_final=10, reference=REFERENCE
        )
        by_id = {r.doc_id: r for r in results}
        assert by_id["q1-2021"].metadata["matched_subqueries"] == ["Q1 2021", "FY2021"]

    def test_a_document_matching_several_sub_queries_appears_once(self, corpus, retriever):
        results, _ = search_decomposed(
            retriever, self.QUERY, beta=0.5, top_k_final=10, reference=REFERENCE
        )
        assert [r.group_id for r in results].count("review") <= 1

    def test_the_review_records_every_year_it_answered(self, corpus, retriever):
        results, _ = search_decomposed(
            retriever, self.QUERY, beta=0.5, top_k_final=10, reference=REFERENCE
        )
        matched = {
            r.group_id: r.metadata.get("matched_subqueries", []) for r in results
        }
        assert len(matched.get("review", [])) > 1

    def test_results_are_reranked_after_merging(self, corpus, retriever):
        results, _ = search_decomposed(
            retriever, self.QUERY, beta=0.5, top_k_final=10, reference=REFERENCE
        )
        scores = [r.combined_score for r in results]
        assert scores == sorted(scores, reverse=True)
        assert [r.rank for r in results] == list(range(1, len(results) + 1))

    def test_top_k_final_bounds_the_merged_output(self, corpus, retriever):
        results, _ = search_decomposed(
            retriever, self.QUERY, top_k_final=3, reference=REFERENCE
        )
        assert len(results) <= 3

    def test_a_non_temporal_query_falls_back_to_one_search(self, corpus, retriever):
        results, decomposition = search_decomposed(
            retriever, "IBM cloud strategy", top_k_final=5, reference=REFERENCE
        )
        assert len(decomposition) == 1
        assert results
