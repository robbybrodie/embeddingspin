#!/usr/bin/env python3
"""
Arcs, Points and Segments
=========================

The half of the system ``demo.py`` does not cover: what an arc actually is, how
points and arcs coexist in one index, and what the segment subdivision buys.

1. **Point mode** (``z = 0``) — an instant. A news announcement.
2. **Arc mode** (``z > 0``) — a duration. A 10-Q covers a quarter; a 10-K covers a
   year. Containment falls out of the geometry: the Q2 arc lies inside the annual
   arc on the 1-year circle, so an annual query overlaps every quarter in it.
3. **Segments** — each circle is divided evenly: the 1-year circle into four
   quarters, the 16-year circle into sixteen years. The geometry stays even so the
   index is one multiply; which *calendar* quarter an interval occupies is resolved
   from the dates at encoding time and stored. Scoring each intersected segment
   independently keeps a document on the far side of a divider from being lost when
   it genuinely overlaps.

Run: ``python arc_demo.py``
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

from ingestion import TemporalSpinIngestionPipeline
from llamastack_client import MockEmbeddingClient
from retrieval import TemporalSpinRetriever
from temporal_config import DEFAULT_HIERARCHY, QUARTER_SCALE
from temporal_encoding import (
    TemporalInterval,
    encode_single,
    evaluate_scale,
    segment_bounds,
    segment_label,
)
from temporal_spin import SpinDocument
from vector_store import InMemoryVectorStore


def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def describe(doc: SpinDocument) -> None:
    mode = "ARC  " if doc.is_arc else "POINT"
    print(f"  [{mode}] {doc.doc_id:<22}", end="")
    if doc.is_arc:
        quarter = doc.encoding.tuple_for("quarter")
        print(
            f" {doc.interval.start.date()} -> {doc.interval.end.date()}"
            f"   {doc.interval.duration_days:6.1f}d"
            f"   z={math.degrees(quarter.z):6.1f}° on the 1-year circle"
        )
    else:
        print(f" {doc.interval.start.date()}              instant, z=0")


def print_results(results, description: str) -> None:
    print(f"\n{'─' * 80}")
    print(f"{description} — {len(results)} result(s)\n")
    for r in results:
        kind = "ARC  " if r.interval.end else "POINT"
        print(
            f"  {r.rank}. [{kind}] {r.doc_id:<22} combined {r.combined_score:.4f}"
            f"  (semantic {r.semantic_score:+.4f}, temporal {r.temporal_alignment:.4f})"
        )


def main() -> None:
    print_header("Arcs, Points and Segments")

    embedding_client = MockEmbeddingClient(dimension=384)
    vector_store = InMemoryVectorStore(hierarchy=DEFAULT_HIERARCHY)
    pipeline = TemporalSpinIngestionPipeline(
        embedding_client, vector_store, hierarchy=DEFAULT_HIERARCHY
    )
    retriever = TemporalSpinRetriever(
        embedding_client, vector_store, hierarchy=DEFAULT_HIERARCHY, default_beta=0.5
    )

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    print_header("INGESTION: annual arcs, quarterly arcs, point events")

    documents = []

    print("Annual reports (10-K) — one full-year arc each:")
    for year in (2022, 2023, 2024):
        documents += pipeline.ingest_document(
            text=(
                f"IBM Annual Report {year}: total revenue, cloud growth, AI investment. "
                f"Strategic focus on hybrid cloud and quantum computing. "
                f"Fiscal year ended December 31, {year}."
            ),
            interval=TemporalInterval.of_year(year),
            doc_id=f"IBM-10K-{year}",
            metadata={"type": "10-K", "year": year},
        )

    print("Quarterly reports (10-Q) for 2023 — one quarter arc each:")
    for quarter in (1, 2, 3, 4):
        documents += pipeline.ingest_document(
            text=(
                f"IBM Q{quarter} 2023 Report: revenue, cloud revenue growth, "
                f"quantum computing milestone."
            ),
            interval=TemporalInterval.of_quarter(2023, quarter),
            doc_id=f"IBM-10Q-2023-Q{quarter}",
            metadata={"type": "10-Q", "year": 2023, "quarter": quarter},
        )

    print("Announcements — points, no duration:")
    events = [
        ("2023-02-15", "IBM announces a major AI partnership to accelerate adoption."),
        ("2023-05-10", "IBM unveils a 1000-qubit quantum processor at a tech conference."),
        ("2023-08-22", "IBM cloud revenue exceeds expectations; hybrid cloud growth continues."),
        ("2023-11-05", "IBM Q3 earnings beat estimates across all business segments."),
    ]
    for date_str, text in events:
        moment = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        documents += pipeline.ingest_document(
            text=text,
            interval=TemporalInterval.point(moment),
            doc_id=f"IBM-NEWS-{date_str}",
            metadata={"type": "news"},
        )

    print()
    for doc in documents:
        describe(doc)

    arcs = [d for d in documents if d.is_arc]
    points = [d for d in documents if not d.is_arc]
    print(f"\n  {len(documents)} representations: {len(arcs)} arcs, {len(points)} points.")
    print("  Both live in one index and one coordinate space; a point is just an arc")
    print("  of zero length, so no separate code path is needed to compare them.")

    # ------------------------------------------------------------------
    # Containment
    # ------------------------------------------------------------------

    print_header("CONTAINMENT: an annual query reaches every quarter inside it")

    results = retriever.search(
        "IBM fiscal year 2023 performance",
        interval=TemporalInterval.of_year(2023),
        beta=0.7,
        top_k_final=10,
    )
    print_results(results, "Query: FY2023 (a full-year arc)")
    print()
    print("  The four 2023 quarters and the four 2023 announcements all sit inside the")
    print("  query arc, so all of them overlap. The 2022 and 2024 annual reports occupy")
    print("  the same phase on the 1-year circle — every year does — and are rejected")
    print("  on the 16-year circle instead.")

    print_header("CONTAINMENT: a quarterly query, and what it excludes")

    results = retriever.search(
        "Q2 2023 revenue cloud growth",
        interval=TemporalInterval.of_quarter(2023, 2),
        beta=0.7,
        top_k_final=10,
    )
    print_results(results, "Query: Q2 2023")
    print()
    print("  Q2 2023, the 2023 annual report (which contains Q2), and the 10 May")
    print("  announcement (a point inside the arc) survive. Q1, Q3 and Q4 do not")
    print("  overlap on the 1-year circle and are gated out.")

    print_header("A POINT QUERY")

    results = retriever.search(
        "quantum computing breakthrough",
        interval=TemporalInterval.point(datetime(2023, 5, 10, tzinfo=timezone.utc)),
        beta=0.7,
        top_k_final=6,
    )
    print_results(results, "Query: the instant 2023-05-10")
    print()
    print("  A zero-length query arc still overlaps any arc containing it, so the")
    print("  enclosing quarter and year come back alongside the exact-date event.")

    # ------------------------------------------------------------------
    # Segments
    # ------------------------------------------------------------------

    print_header("SEGMENTS: even geometry, calendar identity")

    for scale in DEFAULT_HIERARCHY.scales:
        width = math.degrees(segment_bounds(scale, 0)[1])
        print(
            f"  {scale.name:<9} {scale.period_years:>5.0f}y divided into "
            f"{scale.segments:>2} x {scale.segment_label:<14} ({width:.1f}° each)"
        )

    print()
    print("  The circle is divided evenly, always. Periods and segment counts are")
    print("  powers of two, so a segment index is int(fraction * segments) — one")
    print("  multiply, no divider table, no branch, and it vectorises across a batch.")
    print()
    print("  The calendar is not even: quarters run 90, 91, 92 and 92 days, and a leap")
    print("  day shifts every divider after February. None of that is allowed into the")
    print("  geometry. Segment identity is resolved from the real dates once, at")
    print("  encoding time, and stored. Even geometry first, calendar correction after:")
    print()

    samples = [
        ("Q1 2023", TemporalInterval.of_quarter(2023, 1)),
        ("Q2 2023", TemporalInterval.of_quarter(2023, 2)),
        ("Feb-May 2023", TemporalInterval(
            datetime(2023, 2, 1, tzinfo=timezone.utc),
            datetime(2023, 5, 1, tzinfo=timezone.utc),
        )),
        ("FY2023", TemporalInterval.of_year(2023)),
    ]
    for label, interval in samples:
        tuple_ = encode_single(interval, DEFAULT_HIERARCHY).tuple_for("quarter")
        names = ", ".join(
            segment_label(QUARTER_SCALE, i, DEFAULT_HIERARCHY) for i in tuple_.segments
        )
        print(f"  {label:<14} touches {len(tuple_.segments)} segment(s): {names}")

    print()
    print("  Why it matters. A document running February to May straddles the 1 April")
    print("  divider. Scored as one arc against a Q2 query it looks like a weak match,")
    print("  because two thirds of it falls outside Q2. Scoring each intersected")
    print("  segment on its own keeps the part that genuinely does overlap:")
    print()

    query_tuple = encode_single(
        TemporalInterval.of_quarter(2023, 2), DEFAULT_HIERARCHY
    ).tuple_for("quarter")
    doc_tuple = encode_single(
        TemporalInterval(
            datetime(2023, 2, 1, tzinfo=timezone.utc),
            datetime(2023, 5, 1, tzinfo=timezone.utc),
        ),
        DEFAULT_HIERARCHY,
    ).tuple_for("quarter")

    match = evaluate_scale(query_tuple, doc_tuple, QUARTER_SCALE)
    print(f"    query  : Q2 2023")
    print(f"    document: 2023-02-01 -> 2023-05-01")
    print(f"    overlaps           : {match.overlaps}")
    print(f"    shared segments    : {list(match.shared_segments)}")
    for index, score in sorted(match.segment_jaccard.items()):
        name = segment_label(QUARTER_SCALE, index, DEFAULT_HIERARCHY)
        print(f"      {name:<12} per-segment Jaccard {score:.4f}")
    print(f"    scale Jaccard      : {match.jaccard:.4f}")
    print()
    print("  A match in any one segment qualifies the document. Without that, a hard")
    print("  divider would silently discard material that overlaps the query.")

    # ------------------------------------------------------------------
    # Beta
    # ------------------------------------------------------------------

    print_header("β ACROSS A MIXED COLLECTION")

    print("Query: 'IBM cloud revenue', August 2023\n")
    interval = TemporalInterval.of_month(2023, 8)
    for beta in (0.0, 0.25, 0.5, 0.75, 1.0):
        results = retriever.search(
            "IBM cloud revenue", interval=interval, beta=beta, top_k_final=3
        )
        print(f"  β = {beta:.2f}:")
        for r in results:
            kind = "ARC  " if r.interval.end else "POINT"
            print(f"    {r.rank}. [{kind}] {r.doc_id:<22} {r.combined_score:+.4f}")
        print()

    print("  At β = 0 the ordering is whatever semantic similarity says; at β = 1 it is")
    print("  temporal overlap alone. The gate applies throughout — β weights the")
    print("  surviving candidates, it does not admit non-overlapping ones.")

    print_header("Done")
    print("  • Points and arcs share one index and one comparison rule.")
    print("  • Containment is geometric: no interval arithmetic at query time.")
    print("  • Segments stay even in the geometry; calendar identity is fixed at ingestion.")
    print("  • β weights the survivors; the arc-overlap gate decides who survives.")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
