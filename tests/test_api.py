"""
The FastAPI service.

Exercised against the in-memory store with mock embeddings and the demo corpus, so
these are integration tests over the whole stack: request validation, ingestion,
the two-pass search, decomposition and the statistics the service reports about its
own hierarchy.
"""

from __future__ import annotations

import os

import pytest

os.environ["USE_OPENAI_EMBEDDINGS"] = "false"
os.environ["USE_MOCK_EMBEDDINGS"] = "true"
os.environ["VECTOR_STORE"] = "memory"
os.environ["LOAD_DEMO_DATA"] = "true"

from fastapi.testclient import TestClient  # noqa: E402

import api  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(api.app) as test_client:
        yield test_client


# ---------------------------------------------------------------------------
# Service description
# ---------------------------------------------------------------------------


class TestHealthAndStats:
    def test_health(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_stats_reports_the_active_hierarchy(self, client):
        body = client.get("/stats").json()
        assert body["epoch"].startswith("1900-01-01")
        assert body["year_convention"] == "calendar"
        assert body["temporal_dimensions"] == 9
        assert body["coverage_end_year"] == 2156
        assert body["schema_version"] == 2

    def test_stats_lists_every_scale(self, client):
        scales = client.get("/stats").json()["scales"]
        assert [s["name"] for s in scales] == ["quarter", "decade", "century"]
        assert [s["period_years"] for s in scales] == [1, 16, 256]

    def test_stats_separates_representations_from_documents(self, client):
        """
        The demo corpus contains a multi-year review that is indexed once per year it
        spans, so the row count exceeds the document count.
        """
        body = client.get("/stats").json()
        assert body["total_representations"] > body["total_documents"] > 0

    def test_stats_reports_the_fingerprint(self, client):
        assert client.get("/stats").json()["fingerprint"] == (
            "v2|1900-01-01|calendar|quarter:1:4+decade:16:16+century:256:16"
        )


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


class TestTemporalSearch:
    def test_a_quarterly_search_returns_results(self, client):
        response = client.post("/temporal_search", json={
            "query": "IBM revenue", "start": "2021-01-01", "end": "2021-04-01",
        })
        assert response.status_code == 200
        assert response.json()["results"]

    def test_results_carry_the_group_id_and_interval(self, client):
        body = client.post("/temporal_search", json={
            "query": "IBM revenue", "start": "2021-01-01", "end": "2022-01-01",
        }).json()
        first = body["results"][0]
        assert first["group_id"]
        assert first["interval"]["start"]
        assert first["traversed_scales"]

    def test_results_carry_the_per_scale_breakdown(self, client):
        body = client.post("/temporal_search", json={
            "query": "IBM revenue", "start": "2021-01-01", "end": "2021-04-01",
        }).json()
        matches = body["results"][0]["scale_matches"]
        assert {m["scale"] for m in matches} == {"century", "decade", "quarter"}
        assert all(m["overlaps"] for m in matches)

    def test_beta_is_bounded_to_the_unit_interval(self, client):
        """
        β blends two scores that are each in [0, 1]. The old API accepted values up
        to 10000, which was a different parameter wearing the same name.
        """
        for bad in (-0.1, 1.5, 5000.0):
            response = client.post("/temporal_search", json={
                "query": "IBM revenue", "start": "2021-01-01", "beta": bad,
            })
            assert response.status_code == 422

    def test_beta_at_both_ends_is_accepted(self, client):
        for beta in (0.0, 1.0):
            response = client.post("/temporal_search", json={
                "query": "IBM revenue", "start": "2021-01-01", "beta": beta,
            })
            assert response.status_code == 200

    def test_an_end_at_or_before_the_start_is_rejected(self, client):
        response = client.post("/temporal_search", json={
            "query": "IBM revenue", "start": "2021-04-01", "end": "2021-01-01",
        })
        assert response.status_code == 400
        assert "half-open" in response.json()["detail"]

    def test_a_zero_length_interval_is_rejected(self, client):
        response = client.post("/temporal_search", json={
            "query": "IBM revenue", "start": "2021-01-01", "end": "2021-01-01",
        })
        assert response.status_code == 400

    def test_a_malformed_date_is_rejected(self, client):
        response = client.post("/temporal_search", json={
            "query": "IBM revenue", "start": "not-a-date",
        })
        assert response.status_code == 400

    def test_the_response_echoes_the_query_parameters(self, client):
        body = client.post("/temporal_search", json={
            "query": "IBM revenue", "start": "2021-01-01", "end": "2022-01-01", "beta": 0.75,
        }).json()
        assert body["query"] == "IBM revenue"
        assert body["beta"] == 0.75

    def test_a_search_outside_the_corpus_returns_no_results(self, client):
        body = client.post("/temporal_search", json={
            "query": "IBM revenue", "start": "1950-01-01", "end": "1951-01-01",
        }).json()
        assert body["results"] == []


# ---------------------------------------------------------------------------
# Decomposition
# ---------------------------------------------------------------------------


class TestDecompose:
    QUERY = "What was the Q1 impact on the full year for 2021, 2022 and 2023?"

    def test_the_headline_query_yields_six_sub_queries(self, client):
        body = client.post("/decompose", json={"query": self.QUERY}).json()
        assert len(body["subqueries"]) == 6
        assert [s["label"] for s in body["subqueries"]][:2] == ["Q1 2021", "FY2021"]

    def test_the_decomposition_reports_its_anchors_and_granularities(self, client):
        body = client.post("/decompose", json={"query": self.QUERY}).json()
        assert body["subquery_count"] == 6
        assert body["anchors"] == [2021, 2022, 2023]
        assert body["truncated"] is False

    def test_a_non_temporal_query_yields_one_unconstrained_sub_query(self, client):
        body = client.post("/decompose", json={"query": "IBM cloud strategy"}).json()
        assert body["subquery_count"] == 1
        assert body["subqueries"][0]["interval"] is None

    def test_a_reference_date_resolves_relative_expressions(self, client):
        body = client.post("/decompose", json={
            "query": "revenue over the last three years", "reference": "2026-09-15",
        }).json()
        assert body["subqueries"][0]["interval"]["start"].startswith("2023")

    def test_decomposed_search_merges_and_deduplicates(self, client):
        body = client.post("/decomposed_search", json={
            "query": self.QUERY, "beta": 0.5, "top_k_final": 10,
        }).json()
        group_ids = [r["group_id"] for r in body["results"]]
        assert len(group_ids) == len(set(group_ids))
        assert body["decomposition"]["subquery_count"] == 6


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


class TestIngest:
    def test_a_multi_year_document_reports_both_counts(self, client):
        body = client.post("/ingest", json={"documents": [{
            "text": "A five-year strategic outlook.",
            "start": "2030-01-01",
            "end": "2035-01-01",
            "doc_id": "outlook-2030-2034",
        }]}).json()
        assert body["ingested_count"] == 1
        assert body["representation_count"] == 5
        assert body["group_ids"] == ["outlook-2030-2034"]

    def test_an_ingested_document_becomes_searchable(self, client):
        client.post("/ingest", json={"documents": [{
            "text": "Quantum roadmap milestone for the year.",
            "start": "2031-01-01",
            "end": "2032-01-01",
            "doc_id": "quantum-2031",
        }]})
        body = client.post("/temporal_search", json={
            "query": "quantum roadmap milestone",
            "start": "2031-01-01",
            "end": "2032-01-01",
        }).json()
        assert "quantum-2031" in {r["group_id"] for r in body["results"]}

    def test_a_point_document_is_accepted(self, client):
        body = client.post("/ingest", json={"documents": [{
            "text": "An announcement on a single day.",
            "start": "2033-06-01",
            "doc_id": "announcement-2033",
        }]}).json()
        assert body["representation_count"] == 1

    def test_an_inverted_interval_is_rejected(self, client):
        response = client.post("/ingest", json={"documents": [{
            "text": "Backwards.", "start": "2031-01-01", "end": "2030-01-01",
        }]})
        assert response.status_code == 400


# ---------------------------------------------------------------------------
# Beta sweep
# ---------------------------------------------------------------------------


class TestBetaSweep:
    def test_the_sweep_returns_one_ranking_per_value(self, client):
        body = client.post("/beta_sweep", json={
            "query": "IBM revenue", "start": "2021-01-01", "end": "2022-01-01",
        }).json()
        sweep = body["results_by_beta"]
        assert body["beta_values"] == [0.0, 0.25, 0.5, 0.75, 1.0]
        assert set(sweep) == {f"beta_{beta}" for beta in body["beta_values"]}

    def test_the_sweep_does_not_change_the_index(self, client):
        before = client.get("/stats").json()["total_representations"]
        client.post("/beta_sweep", json={
            "query": "IBM revenue", "start": "2021-01-01", "end": "2022-01-01",
        })
        assert client.get("/stats").json()["total_representations"] == before
