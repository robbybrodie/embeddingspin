#!/usr/bin/env python3
"""
Test Temporal Scaling Feature
==============================

Tests the new temporal_scale parameter that amplifies the spin vector
BEFORE concatenation with semantic embeddings.

This should make temporal features more prominent in similarity calculations,
giving us stronger temporal alignment without needing extreme β values.
"""

import os
from datetime import datetime, timezone

# Check for API key
if not os.getenv('OPENAI_API_KEY'):
    print("ERROR: OPENAI_API_KEY environment variable not set")
    exit(1)

from openai_client import OpenAIEmbeddingClient
from vector_store import InMemoryVectorStore
from retrieval import TemporalSpinRetriever
from ingestion import TemporalSpinIngestionPipeline
from ingest_real_reports import load_reports_from_directory

print("="*80)
print("TESTING TEMPORAL SCALING (temporal_scale parameter)")
print("="*80)
print()

# Initialize OpenAI client
embedding_client = OpenAIEmbeddingClient(model="text-embedding-3-small")
print(f"✓ Using OpenAI text-embedding-3-small")
print()

# Load reports (truncated for speed)
def truncate_text(text, max_chars=20000):
    return text[:max_chars] if len(text) > max_chars else text

print("Loading reports...")
reports = load_reports_from_directory("ibm_reports_10yr/sample10ks")
reports = [(truncate_text(text), ts, doc_id, fp) for text, ts, doc_id, fp in reports]
print(f"✓ Loaded {len(reports)} reports")
print()

# Test multiple temporal_scale values
scale_values = [1.0, 5.0, 10.0, 20.0]

query = "IBM 2007 total revenue net income earnings"
query_time = datetime(2007, 12, 31, tzinfo=timezone.utc)

print(f"Query: '{query}'")
print(f"Target Year: 2007")
print()

for temp_scale in scale_values:
    print("="*80)
    print(f"TESTING temporal_scale = {temp_scale}")
    print("="*80)
    
    # Create fresh vector store and pipeline with this scale
    vector_store = InMemoryVectorStore()
    pipeline = TemporalSpinIngestionPipeline(
        embedding_client=embedding_client,
        vector_store=vector_store,
        temporal_scale=temp_scale
    )
    
    # Ingest with this temporal scale
    print(f"Ingesting with temporal_scale={temp_scale}...")
    for i, (text, ts, doc_id, fp) in enumerate(reports, 1):
        if i <= 3:  # Show first few
            print(f"  [{i}/{len(reports)}] {doc_id}...", end=" ", flush=True)
        elif i == 4:
            print(f"  ... (ingesting remaining {len(reports)-3} documents)", flush=True)
        
        pipeline.ingest_batch([text], [ts], [doc_id], [{"filepath": fp}])
        
        if i <= 3:
            print("✓")
    
    print()
    
    # Create retriever with matching temporal_scale
    retriever = TemporalSpinRetriever(
        embedding_client=embedding_client,
        vector_store=vector_store,
        temporal_scale=temp_scale,
        default_beta=50.0
    )
    
    # Query
    results = retriever.search(
        query_text=query,
        query_timestamp=query_time,
        beta=50.0,
        top_k_final=5
    )
    
    print(f"TOP 5 RESULTS (β=50.0):")
    print("-"*80)
    for i, result in enumerate(results, 1):
        is_target = result.timestamp.year == 2007
        marker = " <<<< TARGET" if is_target else ""
        print(f"{i}. {result.doc_id:20s} Year: {result.timestamp.year:4d} | "
              f"Score: {result.combined_score:.4f}{marker}")
    
    # Check if 2007 is first
    if results[0].timestamp.year == 2007:
        print()
        print(f"✅ SUCCESS! 2007 ranked #1 with temporal_scale={temp_scale}")
        print(f"   Temporal scaling amplified temporal features enough to win!")
        print()
        break
    else:
        rank_2007 = next((i+1 for i, r in enumerate(results) if r.timestamp.year == 2007), None)
        print()
        print(f"⚠️  2007 ranked #{rank_2007}")
        print()

print("="*80)
print("CONCLUSION")
print("="*80)
print()
print("Temporal scaling provides an alternative to extreme β values.")
print("By amplifying the spin vector BEFORE concatenation, temporal")
print("features get more weight in the similarity calculation.")
print()
print("Combined with β in re-ranking, this gives two-stage temporal control:")
print("  - Stage 1: temporal_scale affects initial cosine similarity")
print("  - Stage 2: β controls exponential temporal alignment penalty")
print()

