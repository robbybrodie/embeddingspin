"""
End-to-end retrieval: the hard overlap gate, β, deduplication and lazy traversal.

These run against the in-memory store with deterministic mock embeddings, so
semantic scores are stable and any change in ordering is a change in the temporal
logic rather than in the embedding model.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from temporal_config import DEFAULT_HIERARCHY, TemporalHierarchy
from temporal_encoding import TemporalInterval
from temporal_spin import RetrievalResult, deduplicate_by_group
from vector_store import InMemoryVectorStore


def utc(*args) -> datetime:
    return datetime(*args, tzinfo=timezone.utc)


@pytest.fixture
def corpus(pipeline):
    """Annual reports 2019-2023, the four 2023 quarters, and a 2017-2022 review."""
    for year in range(2019, 2024):
        pipeline.ingest_document(
            text=f"IBM annual report {year}: revenue, cloud growth, AI investment.",
            interval=TemporalInterval.of_year(year),
            doc_id=f"annual-{year}",
            metadata={"type": "10-K", "year": year},
        )
    for quarter in (1, 2, 3, 4):
        pipeline.ingest_document(
            text=f"IBM Q{quarter} 2023: quarterly revenue and cloud growth.",
            interval=TemporalInterval.of_quarter(2023, quarter),
            doc_id=f"q{quarter}-2023",
            metadata={"type": "10-Q", "year": 2023, "quarter": quarter},
        )
    pipeline.ingest_document(
        text="IBM news: a major AI partnership is announced.",
        interval=TemporalInterval.point(utc(2023, 5, 10)),
        doc_id="news-2023-05-10",
        metadata={"type": "news"},
    )
    pipeline.ingest_document(
        text="IBM strategic review 2017-2022: the hybrid cloud transformation.",
        interval=TemporalInterval.spanning(2017, 2022),
        doc_id="review-2017-2022",
        metadata={"type": "review"},
    )
    return pipeline


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


class TestIngestion:
    def test_a_within_year_document_writes_one_row(self, pipeline, store):
        written = pipeline.ingest_document(
            "quarterly results", TemporalInterval.of_quarter(2021, 2), doc_id="q"
        )
        assert len(written) == 1
        assert store.count() == 1

    def test_a_six_year_document_writes_six_rows_under_one_group(self, pipeline, store):
        written = pipeline.ingest_document(
            "strategic review", TemporalInterval.spanning(2017, 2022), doc_id="review"
        )
        assert len(written) == 6
        assert store.count() == 6
        assert store.count_groups() == 1
        assert {d.group_id for d in written} == {"review"}

    def test_split_rows_get_distinct_ids(self, pipeline):
        written = pipeline.ingest_document(
            "strategic review", TemporalInterval.spanning(2017, 2022), doc_id="review"
        )
        assert len({d.doc_id for d in written}) == 6

    def test_split_rows_share_the_semantic_embedding(self, pipeline):
        written = pipeline.ingest_document(
            "strategic review", TemporalInterval.spanning(2017, 2022), doc_id="review"
        )
        first = written[0].semantic_embedding
        assert all(d.semantic_embedding == first for d in written)

    def test_batch_ingestion_counts_representations_not_chunks(self, pipeline, store):
        written = pipeline.ingest_batch(
            texts=["a", "b", "c"],
            intervals=[
                TemporalInterval.of_year(2021),
                TemporalInterval.spanning(2017, 2019),
                TemporalInterval.point(utc(2022, 3, 1)),
            ],
            doc_ids=["a", "b", "c"],
        )
        assert len(written) == 1 + 3 + 1
        assert store.count() == 5
        assert store.count_groups() == 3


# ---------------------------------------------------------------------------
# The hard gate
# ---------------------------------------------------------------------------


class TestOverlapGate:
    def test_a_quarterly_query_excludes_sibling_quarters(self, corpus, retriever):
        results = retriever.search(
            "IBM quarterly revenue",
            interval=TemporalInterval.of_quarter(2023, 2),
            top_k_final=20,
        )
        ids = {r.doc_id for r in results}
        assert "q2-2023" in ids
        assert not {"q1-2023", "q3-2023", "q4-2023"} & ids

    def test_a_quarterly_query_still_reaches_the_containing_year(self, corpus, retriever):
        results = retriever.search(
            "IBM quarterly revenue",
            interval=TemporalInterval.of_quarter(2023, 2),
            top_k_final=20,
        )
        assert "annual-2023" in {r.doc_id for r in results}

    def test_an_annual_query_reaches_every_quarter_inside_it(self, corpus, retriever):
        results = retriever.search(
            "IBM fiscal year performance",
            interval=TemporalInterval.of_year(2023),
            top_k_final=20,
        )
        ids = {r.doc_id for r in results}
        assert {"q1-2023", "q2-2023", "q3-2023", "q4-2023"} <= ids

    def test_an_annual_query_excludes_other_years(self, corpus, retriever):
        results = retriever.search(
            "IBM fiscal year performance",
            interval=TemporalInterval.of_year(2023),
            top_k_final=20,
        )
        ids = {r.doc_id for r in results}
        assert "annual-2023" in ids
        assert "annual-2019" not in ids
        assert "annual-2021" not in ids

    def test_a_point_query_reaches_the_arcs_containing_it(self, corpus, retriever):
        results = retriever.search(
            "AI partnership announcement",
            interval=TemporalInterval.point(utc(2023, 5, 10)),
            top_k_final=20,
        )
        ids = {r.doc_id for r in results}
        assert {"news-2023-05-10", "q2-2023", "annual-2023"} <= ids

    def test_a_query_outside_the_corpus_returns_nothing(self, corpus, retriever):
        results = retriever.search(
            "IBM revenue", interval=TemporalInterval.of_year(2005), top_k_final=20
        )
        assert results == []

    def test_the_gate_is_independent_of_beta(self, corpus, retriever):
        """β weights the survivors; it never admits a non-overlapping document."""
        for beta in (0.0, 0.5, 1.0):
            results = retriever.search(
                "IBM revenue",
                interval=TemporalInterval.of_quarter(2023, 2),
                beta=beta,
                top_k_final=20,
            )
            assert "q1-2023" not in {r.doc_id for r in results}


# ---------------------------------------------------------------------------
# Multi-year documents
# ---------------------------------------------------------------------------


class TestMultiYearDocuments:
    def test_the_review_is_found_through_a_year_it_covers(self, corpus, retriever):
        results = retriever.search(
            "hybrid cloud transformation",
            interval=TemporalInterval.of_year(2021),
            top_k_final=20,
        )
        assert "review-2017-2022" in {r.group_id for r in results}

    def test_the_review_is_not_found_outside_its_span(self, corpus, retriever):
        results = retriever.search(
            "hybrid cloud transformation",
            interval=TemporalInterval.of_year(2023),
            top_k_final=20,
        )
        assert "review-2017-2022" not in {r.group_id for r in results}

    def test_the_review_appears_once_despite_six_representations(self, corpus, retriever):
        results = retriever.search(
            "hybrid cloud transformation",
            interval=TemporalInterval.spanning(2018, 2021),
            top_k_final=20,
        )
        matching = [r for r in results if r.group_id == "review-2017-2022"]
        assert len(matching) == 1

    def test_without_deduplication_the_components_surface_separately(self, corpus, retriever):
        results = retriever.search(
            "hybrid cloud transformation",
            interval=TemporalInterval.spanning(2018, 2021),
            top_k_final=20,
            deduplicate=False,
        )
        matching = [r for r in results if r.group_id == "review-2017-2022"]
        assert len(matching) > 1


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


class TestDeduplication:
    def _result(self, doc_id, group_id, score):
        return RetrievalResult(
            doc_id=doc_id,
            group_id=group_id,
            text="t",
            interval=TemporalInterval.of_year(2021),
            semantic_score=0.0,
            temporal_alignment=0.0,
            combined_score=score,
        )

    def test_the_best_scoring_representation_wins(self):
        results = [
            self._result("a#0", "a", 0.3),
            self._result("a#1", "a", 0.9),
            self._result("a#2", "a", 0.5),
        ]
        deduped = deduplicate_by_group(results)
        assert len(deduped) == 1
        assert deduped[0].combined_score == 0.9

    def test_distinct_groups_are_preserved(self):
        deduped = deduplicate_by_group(
            [self._result("a#0", "a", 0.3), self._result("b#0", "b", 0.2)]
        )
        assert [r.group_id for r in deduped] == ["a", "b"]

    def test_input_order_is_preserved(self):
        deduped = deduplicate_by_group([
            self._result("b#0", "b", 0.1),
            self._result("a#0", "a", 0.2),
            self._result("b#1", "b", 0.9),
        ])
        assert [r.group_id for r in deduped] == ["b", "a"]

    def test_a_missing_group_id_falls_back_to_the_doc_id(self):
        deduped = deduplicate_by_group(
            [self._result("x", "", 0.1), self._result("y", "", 0.2)]
        )
        assert len(deduped) == 2


# ---------------------------------------------------------------------------
# Beta
# ---------------------------------------------------------------------------


class TestBeta:
    def test_beta_is_validated_at_construction(self, embedding_client, store):
        from retrieval import TemporalSpinRetriever

        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            TemporalSpinRetriever(embedding_client, store, default_beta=1.5)

    def test_beta_zero_is_pure_semantic(self, corpus, retriever):
        results = retriever.search(
            "IBM annual report 2023 revenue",
            interval=TemporalInterval.of_year(2023),
            beta=0.0,
            top_k_final=5,
        )
        for r in results:
            assert r.combined_score == pytest.approx(r.semantic_score)

    def test_beta_one_is_pure_temporal(self, corpus, retriever):
        results = retriever.search(
            "IBM annual report 2023 revenue",
            interval=TemporalInterval.of_year(2023),
            beta=1.0,
            top_k_final=5,
        )
        for r in results:
            assert r.combined_score == pytest.approx(r.temporal_alignment)

    def test_beta_interpolates_linearly(self, corpus, retriever):
        results = retriever.search(
            "IBM annual report 2023 revenue",
            interval=TemporalInterval.of_year(2023),
            beta=0.4,
            top_k_final=5,
        )
        for r in results:
            expected = 0.6 * r.semantic_score + 0.4 * r.temporal_alignment
            assert r.combined_score == pytest.approx(expected)

    def test_the_sweep_returns_one_ranking_per_value(self, corpus, retriever):
        sweep = retriever.search_with_beta_sweep(
            "IBM cloud revenue",
            interval=TemporalInterval.of_year(2023),
            beta_values=[0.0, 0.5, 1.0],
            top_k=3,
        )
        assert [beta for beta, _ in sweep] == [0.0, 0.5, 1.0]

    def test_the_sweep_does_not_touch_the_index(self, corpus, retriever, store):
        """β is a runtime knob: nothing is re-embedded or re-indexed between sweeps."""
        before = store.count()
        retriever.search_with_beta_sweep(
            "IBM cloud revenue", interval=TemporalInterval.of_year(2023)
        )
        assert store.count() == before


# ---------------------------------------------------------------------------
# Result shape
# ---------------------------------------------------------------------------


class TestResults:
    def test_results_are_ranked_from_one(self, corpus, retriever):
        results = retriever.search(
            "IBM revenue", interval=TemporalInterval.of_year(2023), top_k_final=5
        )
        assert [r.rank for r in results] == list(range(1, len(results) + 1))

    def test_results_are_ordered_by_combined_score(self, corpus, retriever):
        results = retriever.search(
            "IBM revenue", interval=TemporalInterval.of_year(2023), top_k_final=10
        )
        scores = [r.combined_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_final_is_respected(self, corpus, retriever):
        results = retriever.search(
            "IBM revenue", interval=TemporalInterval.of_year(2023), top_k_final=2
        )
        assert len(results) == 2

    def test_a_result_records_which_scales_were_traversed(self, corpus, retriever):
        results = retriever.search(
            "IBM revenue", interval=TemporalInterval.of_year(2023), top_k_final=1
        )
        assert results[0].traversed_scales == ("century", "decade")

    def test_a_result_explains_itself(self, corpus, retriever):
        results = retriever.search(
            "IBM revenue", interval=TemporalInterval.of_quarter(2023, 2), top_k_final=1
        )
        explanation = results[0].explain()
        assert results[0].doc_id in explanation
        assert "quarter" in explanation

    def test_metadata_filtering_restricts_the_candidate_pool(self, corpus, retriever):
        results = retriever.search(
            "IBM revenue",
            interval=TemporalInterval.of_year(2023),
            top_k_final=20,
        )
        assert any(r.metadata.get("type") == "10-Q" for r in results)


# ---------------------------------------------------------------------------
# Hierarchy guarding
# ---------------------------------------------------------------------------


class TestRetrieverHierarchyGuard:
    def test_a_retriever_refuses_a_store_on_a_different_epoch(self, embedding_client):
        from retrieval import TemporalSpinRetriever

        store = InMemoryVectorStore(hierarchy=TemporalHierarchy(epoch=utc(2010, 1, 1)))
        with pytest.raises(ValueError, match="incompatible"):
            TemporalSpinRetriever(embedding_client, store, hierarchy=DEFAULT_HIERARCHY)


class TestUnconstrainedQuery:
    """
    "I did not say when" is not "I mean right now". An absent interval must not be
    read as a point query against the present moment, which would reject the entire
    historical corpus.
    """

    def test_the_full_span_arc_saturates_every_circle(self):
        from temporal_encoding import encode_single, full_span_interval

        encoding = encode_single(full_span_interval(DEFAULT_HIERARCHY))
        assert all(t.is_full_circle for t in encoding.tuples)

    def test_an_unconstrained_query_returns_results(self, corpus, retriever):
        results = retriever.search("IBM revenue", interval=None, top_k_final=5)
        assert results

    def test_an_unconstrained_query_can_reach_every_group(self, corpus, retriever, store):
        results = retriever.search("IBM revenue", interval=None, top_k_final=50)
        assert len({r.group_id for r in results}) == store.count_groups()

    def test_the_gate_rejects_nothing(self, corpus, retriever):
        """
        A saturating arc overlaps every document, so pass 2 cannot reject. β still
        weights the surviving candidates, but only the soft signal is in play.
        """
        results = retriever.search("IBM revenue", interval=None, beta=1.0, top_k_final=50)
        assert all(r.rejected_at is None for r in results)
        assert all(r.temporal_alignment > 0.0 for r in results)

    def test_an_unconstrained_query_descends_only_the_outermost_circle(self, corpus, retriever):
        results = retriever.search("IBM revenue", interval=None, top_k_final=1)
        assert results[0].traversed_scales == ("century",)


# ---------------------------------------------------------------------------
# Parallel sub-queries
# ---------------------------------------------------------------------------


class TestSearchMany:
    def test_input_order_is_preserved(self, corpus, retriever):
        pairs = [
            ("IBM revenue", TemporalInterval.of_year(2019)),
            ("IBM revenue", TemporalInterval.of_year(2023)),
            ("IBM revenue", TemporalInterval.of_quarter(2023, 1)),
        ]
        batches = retriever.search_many(pairs, top_k_final=5)
        assert len(batches) == 3
        assert "annual-2019" in {r.doc_id for r in batches[0]}
        assert "annual-2023" in {r.doc_id for r in batches[1]}
        assert "q1-2023" in {r.doc_id for r in batches[2]}

    def test_a_single_sub_query_skips_the_thread_pool(self, corpus, retriever):
        batches = retriever.search_many(
            [("IBM revenue", TemporalInterval.of_year(2023))], top_k_final=3
        )
        assert len(batches) == 1
