"""
Temporal-Phase Spin Retrieval System
====================================

Time encoded as a continuous geometric phase on several concurrent circular
scales, appended to a frozen semantic embedding.

For each scale of period ``T`` years::

    φ = 2π · fmod_floored((t - t₀) / T, 1.0)

The modulo is applied to the period ratio *before* multiplication by 2π, and it
is floored, so the remainder is never negative and is exactly zero on a period
boundary. Each scale contributes ``[cos φ, sin φ, z]``, where ``z`` is the arc
length in radians: ``z = 0`` is an instant, ``z > 0`` a duration, capped at 2π
when the interval is at least as long as the period. ``t`` is the midpoint of the
interval, so ``[cos φ, sin φ]`` points at the centre of the arc and ``z`` gives
its extent either side.

The default hierarchy is three circles — 1 year, 16 years, 256 years — giving a
9-dimensional temporal block. Each period is an integer power of two and a factor
of the next, which is what lets an interval be split once at the finest scale and
stay valid on every coarser one.

Modules
-------
- ``temporal_config``     : epoch, scales, segments, the floored modulo
- ``temporal_encoding``   : intervals, encoding, splitting, traversal, scoring
- ``temporal_spin``       : documents, queries, results, text extraction
- ``llamastack_client``   : LlamaStack embeddings (plus a deterministic mock)
- ``openai_client``       : OpenAI embeddings
- ``vector_store``        : in-memory, Chroma and pgvector backends
- ``ingestion``           : document ingestion pipeline
- ``retrieval``           : two-pass retrieval with the hard arc-overlap gate
- ``query_decomposition`` : natural language to parallel temporal sub-queries
- ``demo_data``           : sample corpus
- ``api``                 : FastAPI service
- ``demo`` / ``arc_demo`` : CLI walkthroughs

Quick start
-----------
>>> from datetime import datetime, timezone
>>> from temporal_encoding import TemporalInterval, encode_single
>>> from temporal_config import DEFAULT_HIERARCHY
>>>
>>> interval = TemporalInterval.of_quarter(2026, 1)   # [1 Jan, 1 Apr), 90 days
>>> interval.midpoint.date()
datetime.date(2026, 2, 15)
>>> encoding = encode_single(interval, DEFAULT_HIERARCHY)
>>> quarter = encoding.tuple_for("quarter")
>>> round(quarter.cos, 4), round(quarter.sin, 4), round(quarter.z, 4)
(0.7147, 0.6995, 1.5493)
>>> len(encoding.to_vector())
9

Retrieval
---------
>>> from ingestion import TemporalSpinIngestionPipeline      # doctest: +SKIP
>>> from retrieval import TemporalSpinRetriever              # doctest: +SKIP
>>> from llamastack_client import MockEmbeddingClient        # doctest: +SKIP
>>> from vector_store import InMemoryVectorStore             # doctest: +SKIP
>>>
>>> client = MockEmbeddingClient(dimension=384)              # doctest: +SKIP
>>> store = InMemoryVectorStore(hierarchy=DEFAULT_HIERARCHY) # doctest: +SKIP
>>> pipeline = TemporalSpinIngestionPipeline(client, store)  # doctest: +SKIP
>>> pipeline.ingest_document(                                # doctest: +SKIP
...     "IBM Q1 2026 results", TemporalInterval.of_quarter(2026, 1)
... )
>>> retriever = TemporalSpinRetriever(client, store)         # doctest: +SKIP
>>> retriever.search(                                        # doctest: +SKIP
...     "revenue", interval=TemporalInterval.of_year(2026), beta=0.5
... )

For full examples see README.md, or run ``python demo.py``.
"""

__version__ = "2.0.0"
__author__ = "Robert Brodie"

from temporal_config import (
    SCHEMA_VERSION,
    TAU,
    EPOCH_1900,
    EPOCH_2010_LEGACY,
    YEAR_CONVENTIONS,
    DEFAULT_SCALES,
    DEFAULT_HIERARCHY,
    PATENT_EXAMPLE_HIERARCHY,
    QUARTER_SCALE,
    DECADE_SCALE,
    CENTURY_SCALE,
    MILLENNIUM_SCALE,
    ScaleSpec,
    TemporalHierarchy,
    calendar_year_position,
    default_epoch,
    default_year_convention,
    describe_epoch_migration,
    epoch_shift_is_congruent,
    floored_mod,
    years_since_epoch,
)

from temporal_encoding import (
    MAX_REPRESENTATIONS,
    ScaleMatch,
    ScaleTuple,
    TemporalEncoding,
    TemporalInterval,
    TraversalPlan,
    angular_difference,
    arc_contains_point,
    arc_overlap,
    arc_overlap3,
    boundary_moments,
    encode,
    encode_single,
    encode_tuple,
    evaluate_scale,
    full_span_interval,
    jaccard_arcs,
    neutral_tuple,
    pad_to_hierarchy,
    phase_of,
    calendar_segment,
    calendar_segments_touched,
    segment_bounds,
    segment_label,
    segment_of_phase,
    segments_intersected,
    split_at_boundaries,
    temporal_alignment,
    traversal_plan,
)

from temporal_spin import (
    RetrievalResult,
    SpinDocument,
    SpinQuery,
    cosine_similarity,
    deduplicate_by_group,
    extract_interval_from_text,
    extract_timestamp_from_text,
    normalize_vector,
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

from query_decomposition import (
    MAX_SUBQUERIES,
    Decomposition,
    SubQuery,
    decompose,
    search_decomposed,
)

__all__ = [
    # Configuration: epoch, scales, segments
    "SCHEMA_VERSION",
    "TAU",
    "EPOCH_1900",
    "EPOCH_2010_LEGACY",
    "YEAR_CONVENTIONS",
    "DEFAULT_SCALES",
    "DEFAULT_HIERARCHY",
    "PATENT_EXAMPLE_HIERARCHY",
    "QUARTER_SCALE",
    "DECADE_SCALE",
    "CENTURY_SCALE",
    "MILLENNIUM_SCALE",
    "ScaleSpec",
    "TemporalHierarchy",
    "calendar_year_position",
    "default_epoch",
    "default_year_convention",
    "describe_epoch_migration",
    "epoch_shift_is_congruent",
    "floored_mod",
    "years_since_epoch",

    # Encoding: intervals, tuples, splitting, traversal, scoring
    "MAX_REPRESENTATIONS",
    "ScaleMatch",
    "ScaleTuple",
    "TemporalEncoding",
    "TemporalInterval",
    "TraversalPlan",
    "angular_difference",
    "arc_contains_point",
    "arc_overlap",
    "arc_overlap3",
    "boundary_moments",
    "encode",
    "encode_single",
    "encode_tuple",
    "evaluate_scale",
    "full_span_interval",
    "jaccard_arcs",
    "neutral_tuple",
    "pad_to_hierarchy",
    "phase_of",
    "calendar_segment",
    "calendar_segments_touched",
    "segment_bounds",
    "segment_label",
    "segment_of_phase",
    "segments_intersected",
    "split_at_boundaries",
    "temporal_alignment",
    "traversal_plan",

    # Data classes
    "SpinDocument",
    "SpinQuery",
    "RetrievalResult",

    # Text extraction and vector helpers
    "extract_timestamp_from_text",
    "extract_interval_from_text",
    "deduplicate_by_group",
    "cosine_similarity",
    "normalize_vector",

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

    # Query decomposition
    "MAX_SUBQUERIES",
    "Decomposition",
    "SubQuery",
    "decompose",
    "search_decomposed",

    # Utilities
    "format_results_table",
]
