#!/usr/bin/env python3
"""
Temporal-Phase Spin Retrieval — Demo CLI
========================================

Walks through the whole system on a small IBM corpus: interval encoding across
three circles, boundary splitting, the hard overlap gate, lazy traversal, β as a
runtime knob, and natural-language query decomposition.

Usage::

    python demo.py                      # full walkthrough
    python demo.py --query "..."        # custom query
    python demo.py --start 2016-01-01 --end 2016-04-01
    python demo.py --beta 0.75
    python demo.py --beta-sweep
    python demo.py --decompose "Q1 impact on the full year for 2021 and 2022"
"""

from __future__ import annotations

import argparse
import math
from datetime import datetime, timezone

from demo_data import generate_ibm_report_intervals, generate_query_examples
from ingestion import TemporalSpinIngestionPipeline
from llamastack_client import MockEmbeddingClient
from query_decomposition import decompose, search_decomposed
from retrieval import TemporalSpinRetriever, format_results_table
from temporal_config import DEFAULT_HIERARCHY
from temporal_encoding import TemporalInterval, traversal_plan
from vector_store import InMemoryVectorStore


def print_header(title: str) -> None:
    print()
    print("=" * 80)
    print(title.center(80))
    print("=" * 80)
    print()


def print_section(title: str) -> None:
    print()
    print("-" * 80)
    print(title)
    print("-" * 80)


def _doc_id(interval: TemporalInterval) -> str:
    """Distinguish the FY2017 report from the 2017-2022 review; both start in 2017."""
    first = interval.start.year
    last = (interval.end.year - 1) if interval.end else first
    return f"ibm-report-{first}" if last <= first else f"ibm-report-{first}-{last}"


# ============================================================================
# Steps
# ============================================================================


def demo_ingestion(pipeline: TemporalSpinIngestionPipeline):
    print_header("STEP 1: INGESTION — INTERVAL ENCODING ACROSS THREE CIRCLES")

    hierarchy = pipeline.hierarchy
    print(f"Hierarchy : {hierarchy.fingerprint()}")
    print(f"Epoch     : {hierarchy.epoch.date()}   ({hierarchy.year_convention} years)")
    print(f"Coverage  : through {hierarchy.coverage_end_year}")
    print()
    for scale in hierarchy.scales:
        print(
            f"  {scale.name:<9} period {scale.period_years:>5.0f}y   "
            f"{scale.segments:>2} x {scale.segment_label:<14} weight {scale.weight:.1f}"
        )
    print()
    print("Each circle contributes [cos, sin, z]; z is the arc length in radians.")
    print(f"Temporal block: {hierarchy.dimensions} dimensions appended to the semantic vector.")
    print()

    reports = generate_ibm_report_intervals()
    docs = pipeline.ingest_batch(
        texts=[t for t, _ in reports],
        intervals=[i for _, i in reports],
        doc_ids=[_doc_id(i) for _, i in reports],
    )

    print(f"✓ {len(reports)} documents indexed as {len(docs)} representations")
    print()
    print("Sample encodings (φ = arc start, z = arc length, both in degrees):")
    print()
    print("  Period            │ quarter φ/z      │ decade φ/z       │ century φ/z")
    print("  ──────────────────┼──────────────────┼──────────────────┼─────────────────")
    for doc in docs:
        if doc.encoding.representation_index != 0:
            continue
        label = f"{doc.interval.start.date()}"
        cells = []
        for name in ("quarter", "decade", "century"):
            t = doc.encoding.tuple_for(name)
            cells.append(f"{math.degrees(t.phi_start):6.1f}°/{math.degrees(t.z):6.1f}°")
        print(f"  {label:<17} │ {cells[0]:<16} │ {cells[1]:<16} │ {cells[2]}")

    split = [d for d in docs if d.is_split]
    if split:
        print()
        print_section("Zero-degree boundary splitting")
        group = split[0].group_id
        members = [d for d in docs if d.group_id == group]
        print(f"'{group}' covers {members[0].source_interval}")
        print(
            f"It crosses {len(members) - 1} one-year boundaries, so it is indexed "
            f"{len(members)} times:"
        )
        print()
        for d in members:
            t = d.encoding.tuple_for("decade")
            print(
                f"  {d.doc_id:<28} {d.interval.start.date()} -> "
                f"{d.interval.end.date()}   decade φ={math.degrees(t.phi_start):6.1f}°"
            )
        print()
        print("All six share one group_id. Retrieval deduplicates on it, so the")
        print("consumer sees the document once no matter which arc matched.")

    return docs


def demo_basic_search(retriever: TemporalSpinRetriever):
    print_header("STEP 2: TWO-PASS SEARCH WITH THE HARD OVERLAP GATE")

    query_text = "IBM revenue and financial performance"
    interval = TemporalInterval.of_quarter(2016, 2)
    beta = 0.5

    print(f'Query   : "{query_text}"')
    print(f"Period  : {interval}")
    print(f"β       : {beta}  (0 = pure semantic, 1 = pure temporal)")
    print()
    print("Pass 1: coarse recall on 0.9 x semantic + 0.1 x temporal")
    print("Pass 2: arc-overlap gate per circle, then weighted Jaccard")
    print()

    results = retriever.search(query_text, interval=interval, beta=beta, top_k_final=5)
    print(format_results_table(results, max_text_length=40))

    if results:
        print()
        print("Why the top hit scored as it did:")
        print()
        print(results[0].explain())

    print()
    print("The 2015 and 2017 reports cover the same phase on the 1-year circle as")
    print("2016 does — Q2 is Q2 in every year. They are separated on the 16-year")
    print("circle, which is what the gate rejects them on.")


def demo_lazy_traversal(retriever: TemporalSpinRetriever):
    print_header("STEP 3: LAZY RESOLUTION — SKIPPING CIRCLES THE QUERY SATURATES")

    print("A query arc that wraps a whole circle overlaps every document on it, so")
    print("testing that circle cannot reject anything. The plan skips it. Precision is")
    print("fixed at ingestion; what varies here is how deeply retrieval descends.")
    print()
    print("  Query period                │ traversed                 │ skipped")
    print("  ────────────────────────────┼───────────────────────────┼─────────────────")

    cases = [
        ("a single instant", TemporalInterval.point(datetime(2021, 5, 17, tzinfo=timezone.utc))),
        ("Q2 2021", TemporalInterval.of_quarter(2021, 2)),
        ("FY2021", TemporalInterval.of_year(2021)),
        ("2017-2022", TemporalInterval.spanning(2017, 2022)),
        ("2000-2100", TemporalInterval.spanning(2000, 2100)),
    ]
    for label, interval in cases:
        encoding = retriever.create_query("revenue", interval=interval).encoding
        plan = traversal_plan(encoding, retriever.hierarchy)
        traversed = ", ".join(plan.scale_names)
        skipped = ", ".join(name for name, _ in plan.skipped) or "-"
        print(f"  {label:<27} │ {traversed:<25} │ {skipped}")

    print()
    print("A full-year query skips the 1-year circle: every document in any year")
    print("saturates it. An era-wide query skips everything below the 256-year circle.")


def demo_beta_sweep(retriever: TemporalSpinRetriever):
    print_header("STEP 4: β AS A RUNTIME KNOB")

    query_text = "IBM hybrid cloud and AI strategy"
    interval = TemporalInterval.of_year(2019)

    print(f'Query  : "{query_text}"')
    print(f"Period : {interval}  (Red Hat acquisition era)")
    print()
    print("score = (1 - β) x semantic + β x temporal")
    print()

    for beta, results in retriever.search_with_beta_sweep(
        query_text, interval=interval, beta_values=[0.0, 0.25, 0.5, 0.75, 1.0], top_k=3
    ):
        print(f"{'─' * 80}")
        print(f"β = {beta:.2f}")
        for r in results:
            print(
                f"   {r.doc_id:<24} semantic {r.semantic_score:+.4f} │ "
                f"temporal {r.temporal_alignment:.4f} │ combined {r.combined_score:+.4f}"
            )

    print()
    print("Nothing was re-embedded or re-indexed between these sweeps. β moves")
    print("prioritisation at query time only.")


def demo_decomposition(retriever: TemporalSpinRetriever):
    print_header("STEP 5: NATURAL-LANGUAGE QUERY DECOMPOSITION")

    query = "What was the Q1 impact on the full year for 2021, 2022 and 2023?"
    print(f'Query: "{query}"')
    print()
    print("This is not one question. It carries three anchor years and two")
    print("granularities, so it becomes six independent retrievals run in parallel:")
    print()
    print(decompose(query).describe())
    print()

    results, decomposition = search_decomposed(retriever, query, beta=0.5, top_k_final=5)
    print(f"Merged and deduplicated: {len(results)} result(s)")
    print()
    for r in results:
        labels = ", ".join(r.metadata.get("matched_subqueries", []))
        print(f"  {r.rank}. {r.doc_id:<28} score {r.combined_score:.4f}   matched: {labels}")
    print()
    print("A document answering several sub-questions appears once, at its best")
    print("score, with every sub-query it satisfied recorded against it.")


def demo_multiple_queries(retriever: TemporalSpinRetriever):
    print_header("STEP 6: DIVERSE QUERY EXAMPLES")

    for i, (query_text, interval, description) in enumerate(generate_query_examples(), 1):
        print(f'\nQuery {i}: "{query_text}"')
        print(f"Period : {interval}")
        print(f"Context: {description}")
        results = retriever.search(query_text, interval=interval, beta=0.5, top_k_final=3)
        if not results:
            print("  (no document survived the overlap gate)")
            continue
        for r in results:
            print(f"  {r.rank}. {r.doc_id:<28} score {r.combined_score:.4f}")


# ============================================================================
# Entry point
# ============================================================================


def _parse_date(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Temporal-Phase Spin Retrieval Demo")
    parser.add_argument("--query", type=str, help="Custom query text")
    parser.add_argument("--start", type=str, help="Period start, ISO (inclusive)")
    parser.add_argument("--end", type=str, help="Period end, ISO (exclusive)")
    parser.add_argument("--beta", type=float, default=0.5, help="Temporal focus in [0, 1]")
    parser.add_argument("--beta-sweep", action="store_true", help="Show the β sweep")
    parser.add_argument("--decompose", type=str, help="Decompose and run a NL query")
    parser.add_argument("--top-k", type=int, default=5, help="Results to return")
    args = parser.parse_args()

    if not 0.0 <= args.beta <= 1.0:
        parser.error(f"--beta must be in [0, 1], got {args.beta}")

    print_header("TEMPORAL-PHASE SPIN RETRIEVAL SYSTEM")
    print("Hierarchical phase-encoded temporal vectors for semantic embeddings.")
    print()
    print("  • Time as a continuous geometric phase: φ = 2π · fmod((t - t₀)/T, 1)")
    print("  • Three concurrent circles — 1, 16 and 256 years — as [cos, sin, z]")
    print("  • Durations are arcs; instants are points (z = 0)")
    print("  • β adjusts temporal focus at query time, with no retraining")
    print()
    print("Initialising with mock embeddings (deterministic, no network)...")

    embedding_client = MockEmbeddingClient(dimension=384)
    vector_store = InMemoryVectorStore(hierarchy=DEFAULT_HIERARCHY)
    pipeline = TemporalSpinIngestionPipeline(
        embedding_client=embedding_client,
        vector_store=vector_store,
        hierarchy=DEFAULT_HIERARCHY,
    )
    retriever = TemporalSpinRetriever(
        embedding_client=embedding_client,
        vector_store=vector_store,
        hierarchy=DEFAULT_HIERARCHY,
    )
    print("✓ System initialised")

    if args.query:
        demo_ingestion(pipeline)
        print_header("CUSTOM QUERY")
        interval = None
        if args.start:
            interval = TemporalInterval(
                _parse_date(args.start), _parse_date(args.end) if args.end else None
            )
        print(f'Query : "{args.query}"')
        print(f"Period: {interval if interval else '(unconstrained)'}")
        print(f"β     : {args.beta}")
        print()
        results = retriever.search(
            args.query, interval=interval, beta=args.beta, top_k_final=args.top_k
        )
        print(format_results_table(results, max_text_length=50))

    elif args.decompose:
        demo_ingestion(pipeline)
        print_header("QUERY DECOMPOSITION")
        print(decompose(args.decompose).describe())
        print()
        results, _ = search_decomposed(
            retriever, args.decompose, beta=args.beta, top_k_final=args.top_k
        )
        print(format_results_table(results, max_text_length=50))

    elif args.beta_sweep:
        demo_ingestion(pipeline)
        demo_beta_sweep(retriever)

    else:
        demo_ingestion(pipeline)
        demo_basic_search(retriever)
        demo_lazy_traversal(retriever)
        demo_beta_sweep(retriever)
        demo_decomposition(retriever)
        demo_multiple_queries(retriever)

    print_header("SUMMARY")
    print("  ✓ Model-agnostic")
    print("    The embedding model is frozen; the temporal vector is appended post hoc.")
    print()
    print("  ✓ Hierarchical discrimination")
    print("    Concurrent circles separate 'which quarter' from 'which year' from")
    print("    'which era' — a distinction a single timestamp dimension cannot make.")
    print()
    print("  ✓ Durations, not just instants")
    print("    A period is an arc. Arc overlap is what lets a quarterly filing answer")
    print("    an annual question.")
    print()
    print("  ✓ Runtime temporal focus")
    print("    β moves between semantic and temporal priority without reindexing.")
    print()
    print("  ✓ Lazy resolution")
    print("    Circles the query saturates are skipped; rejections happen coarsest-first.")
    print()
    print("For production: LlamaStack or OpenAI embeddings, Chroma or pgvector for")
    print("storage, FastAPI for serving (see api.py).")
    print("=" * 80)


if __name__ == "__main__":
    main()
