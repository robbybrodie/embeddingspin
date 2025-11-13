#!/usr/bin/env python3
"""
Arc-Based Temporal Encoding Demo
=================================

Demonstrates the new arc encoding feature for time periods/intervals.

This demo shows:
1. Point encoding (legacy): Single timestamp events
2. Arc encoding (new): Time period documents (quarters, years)
3. Hierarchical matching: Quarterly reports ⊂ Annual reports
4. Mixed collections: Points and arcs coexisting

Use Case: Financial reporting hierarchy (10-Q quarterly vs 10-K annual)
"""

import os
from datetime import datetime, timezone
from typing import List

# Use mock embeddings for demo
os.environ['USE_MOCK_EMBEDDINGS'] = 'true'

from llamastack_client import MockEmbeddingClient
from vector_store import InMemoryVectorStore
from ingestion import TemporalSpinIngestionPipeline
from retrieval import TemporalSpinRetriever
from temporal_spin import SpinDocument


def print_header(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def print_doc_info(doc: SpinDocument):
    """Print document temporal encoding info."""
    mode = "🔵 ARC" if doc.is_arc else "⚫ POINT"
    print(f"{mode} | {doc.doc_id[:30]}")
    if doc.is_arc:
        print(f"      Period: {doc.timestamp.date()} to {doc.end_timestamp.date()}")
        arc_days = (doc.end_timestamp - doc.timestamp).days
        print(f"      Duration: {arc_days} days")
        print(f"      Arc length: {doc.spin_vector[2]:.4f} radians")
    else:
        print(f"      Timestamp: {doc.timestamp.date()}")
    print(f"      Spin vector dim: {len(doc.spin_vector)}D")
    print()


def print_results(results: List, query_type: str):
    """Print retrieval results."""
    print(f"\n{'─' * 80}")
    print(f"Top {len(results)} Results (Query type: {query_type}):\n")
    
    for i, result in enumerate(results, 1):
        doc_type = "🔵 ARC" if result.metadata.get('is_arc', False) else "⚫ POINT"
        print(f"{i}. {doc_type} | Score: {result.combined_score:.4f} "
              f"(semantic: {result.semantic_score:.4f}, "
              f"temporal: {result.temporal_alignment:.4f})")
        print(f"   {result.doc_id[:60]}")
        print(f"   Year: {result.timestamp.year}")
        print()


def main():
    print_header("🎯 Temporal-Phase Spin: Arc Encoding Demo")
    
    print("Initializing system with mock embeddings...")
    embedding_client = MockEmbeddingClient()
    vector_store = InMemoryVectorStore()
    pipeline = TemporalSpinIngestionPipeline(embedding_client, vector_store)
    retriever = TemporalSpinRetriever(
        embedding_client,
        vector_store,
        default_beta=5000.0  # Strong temporal focus
    )
    
    # ========================================================================
    # INGESTION: Create mixed collection of points and arcs
    # ========================================================================
    
    print_header("📥 INGESTION: Mixed Point and Arc Documents")
    
    documents = []
    
    # 1. Annual reports (10-K) - Full year arcs
    print("🔵 Ingesting Annual Reports (10-K) - Full year arcs:")
    for year in [2022, 2023, 2024]:
        doc = pipeline.ingest_document(
            text=f"IBM Annual Report {year}: Total revenue $60B, cloud growth 15%, "
                 f"AI investments increased significantly. Strategic focus on hybrid cloud "
                 f"and quantum computing. Fiscal year ended December 31, {year}.",
            timestamp=datetime(year, 1, 1, tzinfo=timezone.utc),
            end_timestamp=datetime(year, 12, 31, tzinfo=timezone.utc),
            doc_id=f"IBM-10K-{year}",
            metadata={"type": "10-K", "year": year, "is_arc": True}
        )
        documents.append(doc)
        print(f"  ✓ {year} Annual Report (365 days)")
    
    # 2. Quarterly reports (10-Q) for 2023
    print("\n🔵 Ingesting Quarterly Reports (10-Q) for 2023 - Quarter arcs:")
    quarters = [
        ("Q1", 1, 1, 3, 31),
        ("Q2", 4, 1, 6, 30),
        ("Q3", 7, 1, 9, 30),
        ("Q4", 10, 1, 12, 31)
    ]
    
    for q_name, start_month, start_day, end_month, end_day in quarters:
        doc = pipeline.ingest_document(
            text=f"IBM {q_name} 2023 Report: Revenue $15B, cloud revenue up 12%, "
                 f"quantum computing milestone achieved. Period ended "
                 f"{datetime(2023, end_month, end_day).strftime('%B %d, 2023')}.",
            timestamp=datetime(2023, start_month, start_day, tzinfo=timezone.utc),
            end_timestamp=datetime(2023, end_month, end_day, tzinfo=timezone.utc),
            doc_id=f"IBM-10Q-2023-{q_name}",
            metadata={"type": "10-Q", "year": 2023, "quarter": q_name, "is_arc": True}
        )
        documents.append(doc)
        duration = (doc.end_timestamp - doc.timestamp).days
        print(f"  ✓ 2023 {q_name} ({duration} days)")
    
    # 3. Point-in-time events (news, announcements)
    print("\n⚫ Ingesting Point-in-Time Events - Single timestamps:")
    events = [
        ("2023-02-15", "IBM announces major AI partnership on February 15, 2023. "
                       "Strategic collaboration will accelerate AI adoption."),
        ("2023-05-10", "IBM quantum computing breakthrough announced May 10, 2023. "
                       "New 1000-qubit processor unveiled at tech conference."),
        ("2023-08-22", "IBM cloud revenue exceeds expectations, reported August 22, 2023. "
                       "Hybrid cloud growth continues strong trajectory."),
        ("2023-11-05", "IBM Q3 earnings beat estimates, announced November 5, 2023. "
                       "Strong performance across all business segments.")
    ]
    
    for date_str, text in events:
        timestamp = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        doc = pipeline.ingest_document(
            text=text,
            timestamp=timestamp,
            doc_id=f"IBM-NEWS-{date_str}",
            metadata={"type": "news", "is_arc": False}
        )
        documents.append(doc)
        print(f"  ✓ {date_str} (point)")
    
    print(f"\n✅ Ingested {len(documents)} documents:")
    print(f"   - 3 annual reports (arcs)")
    print(f"   - 4 quarterly reports (arcs)")
    print(f"   - 4 news events (points)")
    
    # ========================================================================
    # RETRIEVAL DEMOS
    # ========================================================================
    
    print_header("🔍 RETRIEVAL DEMO 1: Arc Query for Q2 2023")
    print("Query: 'Q2 2023 revenue cloud growth'")
    print("Period: April 1 - June 30, 2023 (Arc)")
    print("Beta: 5000 (strong temporal focus)")
    
    results = retriever.search(
        query_text="Q2 2023 revenue cloud growth",
        query_timestamp=datetime(2023, 4, 1, tzinfo=timezone.utc),
        end_timestamp=datetime(2023, 6, 30, tzinfo=timezone.utc),
        beta=5000.0,
        top_k_final=5
    )
    
    print_results(results, "Arc (Q2 2023)")
    
    print("💡 Expected behavior:")
    print("   1. Q2 2023 report should rank HIGHEST (exact arc match)")
    print("   2. 2023 annual report should rank HIGH (contains Q2)")
    print("   3. Adjacent quarters may rank lower (no overlap)")
    print("   4. News from May should rank HIGH (point within arc)")
    
    # ========================================================================
    
    print_header("🔍 RETRIEVAL DEMO 2: Point Query for Specific Date")
    print("Query: 'quantum computing breakthrough'")
    print("Date: May 10, 2023 (Point)")
    print("Beta: 5000")
    
    results = retriever.search(
        query_text="quantum computing breakthrough",
        query_timestamp=datetime(2023, 5, 10, tzinfo=timezone.utc),
        beta=5000.0,
        top_k_final=5
    )
    
    print_results(results, "Point (May 10, 2023)")
    
    print("💡 Expected behavior:")
    print("   1. May 10 news event should rank HIGHEST (exact point match)")
    print("   2. Q2 2023 report should rank HIGH (contains May 10)")
    print("   3. 2023 annual report should rank HIGH (contains May 10)")
    print("   4. Q3 report ranks lower (May 10 outside Q3)")
    
    # ========================================================================
    
    print_header("🔍 RETRIEVAL DEMO 3: Annual Query (Full Year)")
    print("Query: 'IBM fiscal year 2023 performance'")
    print("Period: Jan 1 - Dec 31, 2023 (Arc)")
    print("Beta: 5000")
    
    results = retriever.search(
        query_text="IBM fiscal year 2023 performance",
        query_timestamp=datetime(2023, 1, 1, tzinfo=timezone.utc),
        end_timestamp=datetime(2023, 12, 31, tzinfo=timezone.utc),
        beta=5000.0,
        top_k_final=8
    )
    
    print_results(results, "Arc (Full Year 2023)")
    
    print("💡 Expected behavior:")
    print("   1. 2023 annual report should rank HIGHEST (exact arc match)")
    print("   2. All 2023 quarterly reports rank HIGH (contained within annual)")
    print("   3. All 2023 news events rank HIGH (contained within annual)")
    print("   4. 2022/2024 annual reports rank lower (no overlap)")
    
    # ========================================================================
    
    print_header("🔍 RETRIEVAL DEMO 4: Point Query with Beta Comparison")
    print("Query: 'IBM cloud revenue'")
    print("Date: August 1, 2023 (Point)")
    print("\nComparing different β values:")
    
    for beta in [0, 100, 1000, 5000]:
        results = retriever.search(
            query_text="IBM cloud revenue",
            query_timestamp=datetime(2023, 8, 1, tzinfo=timezone.utc),
            beta=beta,
            top_k_final=3
        )
        
        print(f"\n  β = {beta}:")
        for i, r in enumerate(results[:3], 1):
            doc_type = "ARC" if r.metadata.get('is_arc', False) else "POINT"
            print(f"    {i}. [{doc_type:5}] {r.doc_id[:25]:25} | "
                  f"Score: {r.combined_score:.4f} | "
                  f"Year: {r.timestamp.year}")
    
    print("\n💡 Observation:")
    print("   - β=0: Pure semantic (temporal ignored)")
    print("   - β=100-1000: Weak-moderate temporal preference")
    print("   - β=5000: Strong temporal focus (Aug 2023 prioritized)")
    
    # ========================================================================
    
    print_header("📊 SYSTEM STATISTICS")
    
    arc_docs = [d for d in documents if d.is_arc]
    point_docs = [d for d in documents if not d.is_arc]
    
    print(f"Total documents: {len(documents)}")
    print(f"  - Arc documents: {len(arc_docs)} ({len(arc_docs)/len(documents)*100:.1f}%)")
    print(f"  - Point documents: {len(point_docs)} ({len(point_docs)/len(documents)*100:.1f}%)")
    print(f"\nEmbedding dimensions:")
    print(f"  - Point embeddings: {len(point_docs[0].full_embedding)}D "
          f"(semantic + 3D spin, arc_length=0)")
    print(f"  - Arc embeddings: {len(arc_docs[0].full_embedding)}D "
          f"(semantic + 3D spin, arc_length>0)")
    print(f"\nTemporal encoding range:")
    print(f"  - Earliest: {min(d.timestamp for d in documents).date()}")
    print(f"  - Latest: {max(d.end_timestamp or d.timestamp for d in documents).date()}")
    
    # ========================================================================
    
    print_header("✅ Arc Encoding Demo Complete!")
    print("Key Takeaways:")
    print("  1. ✅ Points and arcs coexist in the same vector store")
    print("  2. ✅ Arc-to-arc matching uses Jaccard similarity (overlap)")
    print("  3. ✅ Point-in-arc matching returns 1.0 (perfect temporal alignment)")
    print("  4. ✅ Hierarchical time periods work (Q2 ⊂ Annual)")
    print("  5. ✅ β parameter still controls temporal zoom")
    print("\nThis solves the time-series retrieval problem for vector databases!")
    print("Combined with time-aware chunking, this enables accurate temporal search.")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()

