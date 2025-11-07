#!/usr/bin/env python3
"""
Debug Temporal Zoom Behavior
=============================

Let's see exactly what's happening with the temporal encoding and zoom.
"""

import os
import math
from datetime import datetime, timezone

if not os.getenv('OPENAI_API_KEY'):
    print("ERROR: OPENAI_API_KEY environment variable not set")
    exit(1)

from openai_client import OpenAIEmbeddingClient
from vector_store import InMemoryVectorStore
from retrieval import TemporalSpinRetriever
from ingestion import TemporalSpinIngestionPipeline
from ingest_real_reports import load_reports_from_directory
from temporal_spin import T0_SECONDS, PERIOD_SECONDS, compute_spin_vector, angular_difference

print("="*80)
print("DEBUG: TEMPORAL ZOOM ANALYSIS")
print("="*80)
print()

# Load reports (truncated)
def truncate_text(text, max_chars=20000):
    return text[:max_chars] if len(text) > max_chars else text

print("Loading reports...")
reports = load_reports_from_directory("ibm_reports_10yr/sample10ks")
reports = [(truncate_text(text), ts, doc_id, fp) for text, ts, doc_id, fp in reports]
print(f"✓ Loaded {len(reports)} reports")
print()

# First, let's check the phase angles for key years
print("PHASE ANGLE ANALYSIS")
print("="*80)
query_year = 2007
query_time = datetime(query_year, 12, 31, tzinfo=timezone.utc)
query_seconds = query_time.timestamp()
_, query_phi = compute_spin_vector(query_seconds, T0_SECONDS, PERIOD_SECONDS)

print(f"Query Year: {query_year}")
print(f"Query Phase: {math.degrees(query_phi):.2f}°")
print()
print(f"{'Year':<6} {'Phase (°)':<12} {'Δφ (rad)':<12} {'Δφ (°)':<12}")
print("-"*80)

phase_data = []
for text, ts, doc_id, fp in reports:
    year = ts.year
    doc_seconds = ts.timestamp()
    _, doc_phi = compute_spin_vector(doc_seconds, T0_SECONDS, PERIOD_SECONDS)
    delta_phi = angular_difference(query_phi, doc_phi)
    phase_data.append((year, doc_phi, delta_phi, doc_id))
    print(f"{year:<6} {math.degrees(doc_phi):>10.2f}  {delta_phi:>10.4f}  {math.degrees(delta_phi):>10.2f}")

print()
print()

# Now let's look at what the exponential penalty does
print("TEMPORAL ALIGNMENT PENALTY ANALYSIS")
print("="*80)
print()

beta_values = [10.0, 50.0, 100.0, 200.0, 500.0]

for beta in beta_values:
    print(f"β = {beta}")
    print(f"{'Year':<6} {'Δφ (°)':<12} {'exp(-β×Δφ²)':<15} {'Impact':<20}")
    print("-"*80)
    
    penalties = []
    for year, doc_phi, delta_phi, doc_id in phase_data:
        penalty = math.exp(-beta * (delta_phi ** 2))
        impact = "PERFECT" if penalty > 0.99 else ("STRONG" if penalty > 0.9 else ("MEDIUM" if penalty > 0.5 else "WEAK"))
        penalties.append((year, delta_phi, penalty))
        
        # Only show years near 2007
        if abs(year - query_year) <= 5:
            marker = " <<<< TARGET" if year == query_year else ""
            print(f"{year:<6} {math.degrees(delta_phi):>10.2f}  {penalty:>13.6f}  {impact:<20}{marker}")
    
    print()

print()
print("NOW LET'S DO A REAL RETRIEVAL TEST")
print("="*80)
print()

# Setup
embedding_client = OpenAIEmbeddingClient(model="text-embedding-3-small")
vector_store = InMemoryVectorStore()
pipeline = TemporalSpinIngestionPipeline(
    embedding_client=embedding_client,
    vector_store=vector_store
)

# Ingest
print("Ingesting documents...")
for i, (text, ts, doc_id, fp) in enumerate(reports, 1):
    if i <= 3:
        print(f"  [{i}/{len(reports)}] {doc_id}...")
    elif i == 4:
        print(f"  ... (ingesting {len(reports)-3} more)")
    pipeline.ingest_batch([text], [ts], [doc_id], [{"filepath": fp}])

print()
print()

# Create retriever
retriever = TemporalSpinRetriever(
    embedding_client=embedding_client,
    vector_store=vector_store
)

query = "IBM 2007 total revenue net income earnings"

# Test with different beta values
for beta in [10.0, 50.0, 100.0, 500.0]:
    print(f"RETRIEVAL WITH β = {beta}")
    print("="*80)
    
    results = retriever.search(
        query_text=query,
        query_timestamp=query_time,
        beta=beta,
        top_k_final=10
    )
    
    print(f"{'Rank':<6} {'Year':<6} {'Combined':<10} {'Semantic':<10} {'Temporal':<10} {'Δφ (°)':<10}")
    print("-"*80)
    
    for i, result in enumerate(results, 1):
        year = result.timestamp.year
        is_target = year == query_year
        marker = " <<<" if is_target else ""
        
        # Calculate delta phi for this doc
        doc_seconds = result.timestamp.timestamp()
        _, doc_phi = compute_spin_vector(doc_seconds, T0_SECONDS, PERIOD_SECONDS)
        delta_phi = angular_difference(query_phi, doc_phi)
        
        print(f"{i:<6} {year:<6} {result.combined_score:<10.4f} "
              f"{result.semantic_score:<10.4f} {result.temporal_alignment:<10.6f} "
              f"{math.degrees(delta_phi):>8.2f}{marker}")
    
    print()

print()
print("DIAGNOSIS:")
print("="*80)
print("1. Check if phase angles are correct (should be ~0.36° per year)")
print("2. Check if temporal alignment penalties are being applied")
print("3. Check if semantic similarity is overwhelming temporal signal")
print("4. If 2006 has MUCH higher semantic similarity, even perfect temporal")
print("   alignment (exp(-β×0²) = 1.0) won't overcome it.")
print()

