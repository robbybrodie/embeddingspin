"""
Temporal-Phase Spin Retrieval System
=====================================

A novel retrieval algorithm that encodes time as angular spin on the unit circle.

Main modules:
- temporal_spin: Core spin encoding and timestamp extraction
- llamastack_client: LlamaStack API integration
- vector_store: Vector database abstraction layer
- ingestion: Document ingestion pipeline
- retrieval: Two-pass temporal zoom retrieval
- demo_data: Mock dataset generator
- api: FastAPI REST API
- demo: CLI demonstration

Quick Start:
-----------
>>> from temporal_spin import compute_spin_vector
>>> from datetime import datetime, timezone
>>> 
>>> timestamp = datetime(2020, 1, 1, tzinfo=timezone.utc)
>>> spin_vector, phi = compute_spin_vector(timestamp.timestamp())
>>> print(f"Spin: {spin_vector}, Phase: {phi:.4f} rad")

For full examples, see README.md or run: python demo.py
"""

__version__ = "1.0.0"
__author__ = "Robert Brodie"

from temporal_spin import (
    compute_spin_vector,
    angular_difference,
    extract_timestamp_from_text,
    SpinDocument,
    SpinQuery,
    RetrievalResult,
)

from llamastack_client import (
    LlamaStackEmbeddingClient,
    MockEmbeddingClient,
)

from vector_store import (
    VectorStore,
    InMemoryVectorStore,
    ChromaVectorStore,
    PGVectorStore,
)

from ingestion import (
    TemporalSpinIngestionPipeline,
    create_ingestion_pipeline,
)

from retrieval import (
    TemporalSpinRetriever,
    format_results_table,
)

__all__ = [
    # Core functions
    "compute_spin_vector",
    "angular_difference",
    "extract_timestamp_from_text",
    
    # Data classes
    "SpinDocument",
    "SpinQuery",
    "RetrievalResult",
    
    # Clients
    "LlamaStackEmbeddingClient",
    "MockEmbeddingClient",
    
    # Vector stores
    "VectorStore",
    "InMemoryVectorStore",
    "ChromaVectorStore",
    "PGVectorStore",
    
    # Pipelines
    "TemporalSpinIngestionPipeline",
    "create_ingestion_pipeline",
    "TemporalSpinRetriever",
    
    # Utilities
    "format_results_table",
]

