#!/usr/bin/env python3
"""
Temporal-Phase Spin Retrieval - Demo CLI
=========================================

Command-line demonstration of temporal spin retrieval with interactive
β (zoom) parameter adjustment.

Usage:
    python demo.py                    # Run full demo
    python demo.py --query "..."      # Custom query
    python demo.py --beta 10.0        # Set specific β
    python demo.py --beta-sweep       # Show β sweep comparison
"""

import argparse
import math
from datetime import datetime, timezone

from temporal_spin import T0_SECONDS, PERIOD_SECONDS
from llamastack_client import MockEmbeddingClient
from vector_store import InMemoryVectorStore
from ingestion import TemporalSpinIngestionPipeline
from retrieval import TemporalSpinRetriever, format_results_table
from demo_data import generate_ibm_reports, generate_query_examples


def print_header(title: str):
    """Print a formatted header."""
    print()
    print("=" * 80)
    print(title.center(80))
    print("=" * 80)
    print()


def print_section(title: str):
    """Print a section divider."""
    print()
    print("-" * 80)
    print(title)
    print("-" * 80)


def demo_ingestion(pipeline: TemporalSpinIngestionPipeline):
    """Demonstrate document ingestion with spin encoding."""
    print_header("STEP 1: DOCUMENT INGESTION WITH TEMPORAL SPIN ENCODING")
    
    print("Loading IBM financial reports (2015-2024)...")
    reports = generate_ibm_reports()
    
    texts = [text for text, _ in reports]
    timestamps = [ts for _, ts in reports]
    doc_ids = [f"ibm-report-{ts.year}" for _, ts in reports]
    
    print(f"  • Extracting timestamps from documents")
    print(f"  • Computing semantic embeddings (384-dim)")
    print(f"  • Encoding temporal phase as 2D spin vectors")
    print(f"  • Concatenating: full_embedding = [semantic + spin]")
    print()
    
    docs = pipeline.ingest_batch(
        texts=texts,
        timestamps=timestamps,
        doc_ids=doc_ids
    )
    
    print(f"✓ Ingested {len(docs)} documents")
    print()
    print("Sample Spin Encodings:")
    print()
    print("  Year  │  Timestamp         │  Phase (φ)  │  Spin Vector")
    print("  ──────┼────────────────────┼─────────────┼──────────────────────")
    
    for doc in docs[::2]:  # Show every other year
        year = doc.timestamp.year
        phi_deg = math.degrees(doc.phi)
        spin_x, spin_y = doc.spin_vector
        print(f"  {year}  │  {doc.timestamp.date()}  │  {phi_deg:6.1f}°     │  [{spin_x:+.3f}, {spin_y:+.3f}]")
    
    print()
    print("Note: Documents 10 years apart have similar phases (periodic encoding)")


def demo_basic_search(retriever: TemporalSpinRetriever):
    """Demonstrate basic temporal spin search."""
    print_header("STEP 2: BASIC TEMPORAL SPIN SEARCH")
    
    query_text = "IBM revenue and financial performance"
    query_timestamp = datetime(2016, 6, 30, tzinfo=timezone.utc)
    beta = 5.0
    
    print(f"Query: \"{query_text}\"")
    print(f"Query Timestamp: {query_timestamp.date()}")
    print(f"Temporal Zoom (β): {beta}")
    print()
    print("Executing two-pass retrieval:")
    print("  Pass 1: Coarse semantic recall (λ=0.1, broad search)")
    print("  Pass 2: Temporal zoom re-ranking (β=5.0)")
    print()
    
    results = retriever.search(
        query_text=query_text,
        query_timestamp=query_timestamp,
        beta=beta,
        top_k_final=5
    )
    
    print("Top 5 Results:")
    print()
    print(format_results_table(results, max_text_length=40))
    print()
    print("Interpretation:")
    print("  • Semantic Score: How well document matches query meaning")
    print("  • Temporal Alignment: exp(-β × (Δφ)²), measures phase alignment")
    print("  • Combined Score: Semantic × Temporal (final ranking)")
    print("  • Δφ: Angular phase difference (smaller = closer in time)")


def demo_beta_sweep(retriever: TemporalSpinRetriever):
    """Demonstrate β parameter sweep to show temporal zoom effect."""
    print_header("STEP 3: TEMPORAL ZOOM DEMONSTRATION (β Sweep)")
    
    query_text = "IBM hybrid cloud and AI strategy"
    query_timestamp = datetime(2019, 12, 31, tzinfo=timezone.utc)
    
    print(f"Query: \"{query_text}\"")
    print(f"Query Timestamp: {query_timestamp.date()} (Red Hat acquisition era)")
    print()
    print("β Parameter (Temporal Zoom Knob):")
    print("  • β = 0:  Pure semantic search (time ignored)")
    print("  • β = 1:  Slight temporal preference")
    print("  • β = 5:  Balanced semantic + temporal")
    print("  • β = 10: Strong temporal focus")
    print("  • β = 20: Very sharp temporal filter")
    print()
    
    beta_values = [0, 1, 5, 10, 20]
    sweep_results = retriever.search_with_beta_sweep(
        query_text=query_text,
        query_timestamp=query_timestamp,
        beta_values=beta_values,
        top_k=3
    )
    
    for beta, results in sweep_results:
        print(f"\n{'─' * 80}")
        print(f"β = {beta:4.1f}:")
        print(f"{'─' * 80}")
        
        for i, result in enumerate(results, 1):
            year = result.timestamp.year
            sem_score = result.semantic_score
            temp_align = result.temporal_alignment
            combined = result.combined_score
            phi_diff = math.degrees(result.phi_difference)
            
            print(f"  {i}. {year} Report")
            print(f"     Semantic: {sem_score:.4f}  |  Temporal: {temp_align:.4f}  |  Combined: {combined:.4f}  |  Δφ: {phi_diff:.1f}°")
    
    print()
    print("Observation:")
    print("  As β increases, ranking shifts toward documents closer in time to 2019.")
    print("  No model retraining needed - β is a runtime parameter!")


def demo_multiple_queries(retriever: TemporalSpinRetriever):
    """Demonstrate with multiple example queries."""
    print_header("STEP 4: DIVERSE QUERY EXAMPLES")
    
    queries = generate_query_examples()
    
    for i, (query_text, query_timestamp, description) in enumerate(queries[:3], 1):
        print(f"\nQuery {i}: \"{query_text}\"")
        print(f"Timestamp: {query_timestamp.date()}")
        print(f"Context: {description}")
        print()
        
        results = retriever.search(
            query_text=query_text,
            query_timestamp=query_timestamp,
            beta=5.0,
            top_k_final=3
        )
        
        print("Top 3 Results:")
        for j, result in enumerate(results, 1):
            year = result.timestamp.year
            score = result.combined_score
            print(f"  {j}. {year} Report (score: {score:.4f})")
        print()


def main():
    """Run the demo."""
    parser = argparse.ArgumentParser(
        description="Temporal-Phase Spin Retrieval Demo"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Custom query text"
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        help="Query timestamp (ISO format, e.g., 2016-06-30)"
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=5.0,
        help="Temporal zoom factor (default: 5.0)"
    )
    parser.add_argument(
        "--beta-sweep",
        action="store_true",
        help="Show β parameter sweep"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results (default: 5)"
    )
    args = parser.parse_args()
    
    print_header("TEMPORAL-PHASE SPIN RETRIEVAL SYSTEM")
    print("A Novel Algorithm for Time-Aware Semantic Search")
    print()
    print("Key Innovation:")
    print("  • Time encoded as angular spin on unit circle: φ = 2π × (t-t₀)/T")
    print("  • Spin vector [cos(φ), sin(φ)] concatenated with semantic embedding")
    print("  • β parameter controls temporal zoom WITHOUT model retraining")
    print("  • Multi-pass retrieval: coarse recall + temporal re-ranking")
    
    # Initialize system
    print()
    print("Initializing system...")
    
    # Use mock embeddings for demo (fast and deterministic)
    embedding_client = MockEmbeddingClient(dimension=384)
    vector_store = InMemoryVectorStore()
    
    pipeline = TemporalSpinIngestionPipeline(
        embedding_client=embedding_client,
        vector_store=vector_store
    )
    
    retriever = TemporalSpinRetriever(
        embedding_client=embedding_client,
        vector_store=vector_store
    )
    
    print("✓ System initialized")
    
    # Run demo scenarios
    if args.query:
        # Custom query mode
        print_header("CUSTOM QUERY")
        
        query_timestamp = None
        if args.timestamp:
            query_timestamp = datetime.fromisoformat(args.timestamp)
            if query_timestamp.tzinfo is None:
                query_timestamp = query_timestamp.replace(tzinfo=timezone.utc)
        
        # Need to ingest data first
        demo_ingestion(pipeline)
        
        print_section("Executing Search")
        print(f"Query: \"{args.query}\"")
        if query_timestamp:
            print(f"Timestamp: {query_timestamp.date()}")
        print(f"β: {args.beta}")
        print()
        
        results = retriever.search(
            query_text=args.query,
            query_timestamp=query_timestamp,
            beta=args.beta,
            top_k_final=args.top_k
        )
        
        print(format_results_table(results, max_text_length=50))
        
    elif args.beta_sweep:
        # Beta sweep mode
        demo_ingestion(pipeline)
        demo_beta_sweep(retriever)
        
    else:
        # Full demo
        demo_ingestion(pipeline)
        demo_basic_search(retriever)
        demo_beta_sweep(retriever)
        demo_multiple_queries(retriever)
    
    # Final summary
    print_header("SUMMARY")
    print("Temporal-Phase Spin Retrieval Advantages:")
    print()
    print("  ✓ No Model Retraining Required")
    print("    Time encoding is post-hoc, works with any embedding model")
    print()
    print("  ✓ Smooth Temporal Zoom")
    print("    β parameter provides continuous control from semantic to temporal focus")
    print()
    print("  ✓ Interpretable")
    print("    Phase angles and alignment scores have clear geometric meaning")
    print()
    print("  ✓ Efficient")
    print("    Two-pass design: fast coarse recall + targeted re-ranking")
    print()
    print("  ✓ Periodic Encoding")
    print("    Circular representation naturally handles periodic patterns")
    print()
    print("For production deployment:")
    print("  • Use LlamaStack Model Gateway for embeddings")
    print("  • Deploy with PGVector or Chroma for scalability")
    print("  • Expose via FastAPI (see api.py)")
    print()
    print("Questions? robertbrodie@example.com")
    print("=" * 80)


if __name__ == "__main__":
    main()

