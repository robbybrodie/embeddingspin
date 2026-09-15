"""
FastAPI Server for Temporal-Phase Spin Retrieval
=================================================

Endpoints
---------
``POST /temporal_search``   Two-pass search with the temporal-focus parameter β.
``POST /decompose``         Preview how a natural-language query splits into sub-queries.
``POST /decomposed_search`` Decompose, search each sub-query in parallel, merge.
``POST /ingest``            Index documents with interval encoding.
``POST /beta_sweep``        One query at several β values.
``GET  /health``            Liveness.
``GET  /stats``             Corpus counts and the temporal header.
``POST /clear``             Empty the store.

β is a runtime parameter in ``[0, 1]``: 0 is pure semantic search, 1 is absolute
temporal dominance, 0.5 balances the two. It is not baked into the index, so a
caller can move it per request without re-embedding anything.

Note the distinction between *representations* and *documents* in ``/stats``: a chunk
whose interval crosses a period boundary is indexed once per component arc, so the
row count exceeds the chunk count. Search deduplicates on ``group_id`` before
returning, so callers see each chunk once.
"""

from __future__ import annotations

import math
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import Body, FastAPI, HTTPException
from pydantic import BaseModel, Field

load_dotenv()

from demo_data import generate_ibm_report_intervals
from ingestion import TemporalSpinIngestionPipeline
from llamastack_client import LlamaStackEmbeddingClient, MockEmbeddingClient
from openai_client import OpenAIEmbeddingClient
from query_decomposition import decompose, search_decomposed
from retrieval import TemporalSpinRetriever
from temporal_config import DEFAULT_HIERARCHY, TemporalHierarchy
from temporal_encoding import TemporalInterval
from vector_store import ChromaVectorStore, InMemoryVectorStore, VectorStore


# ============================================================================
# Helpers
# ============================================================================


def _parse_iso(value: Optional[str], field_name: str) -> Optional[datetime]:
    """Parse an ISO-8601 string, accepting a trailing ``Z``, and force UTC."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid ISO timestamp for {field_name}: {value!r}")
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _interval_from_fields(
    start: Optional[str], end: Optional[str], legacy_point: Optional[str] = None
) -> Optional[TemporalInterval]:
    """
    Build a half-open interval from request fields.

    ``end`` is exclusive. A start with no end is a point (an instant, ``z = 0``);
    ``legacy_point`` is the pre-interval field name and is accepted as a fallback.
    """
    start_dt = _parse_iso(start, "start") or _parse_iso(legacy_point, "query_timestamp")
    end_dt = _parse_iso(end, "end")
    if start_dt is None:
        return None
    if end_dt is not None and end_dt <= start_dt:
        raise HTTPException(
            status_code=400,
            detail="end must be strictly after start; intervals are half-open [start, end)",
        )
    return TemporalInterval(start_dt, end_dt)


def _interval_payload(interval: TemporalInterval) -> Dict[str, Any]:
    return {
        "start": interval.start.isoformat(),
        "end": interval.end.isoformat() if interval.end else None,
        "is_point": interval.is_point,
        "duration_days": round(interval.duration_days, 4) if interval.end else 0.0,
    }


# ============================================================================
# Request / response models
# ============================================================================


class Document(BaseModel):
    """A chunk to index."""

    text: str = Field(..., description="Document text content")
    start: Optional[str] = Field(
        None, description="ISO start of the period this chunk covers (inclusive)"
    )
    end: Optional[str] = Field(
        None, description="ISO end of the period (exclusive). Omit for a single instant."
    )
    timestamp: Optional[str] = Field(None, description="Deprecated alias for 'start'")
    end_timestamp: Optional[str] = Field(None, description="Deprecated alias for 'end'")
    doc_id: Optional[str] = Field(None, description="Stable id; becomes the group_id")
    metadata: Optional[dict] = Field(None, description="Additional metadata")


class TemporalSearchRequest(BaseModel):
    """Request for a two-pass temporal search."""

    query: str = Field(..., description="Search query text", json_schema_extra={"example": "IBM revenue"})
    start: Optional[str] = Field(
        None,
        description="ISO start of the period being asked about (inclusive)",
        json_schema_extra={"example": "2016-01-01T00:00:00Z"},
    )
    end: Optional[str] = Field(
        None,
        description="ISO end of the period (exclusive). Omit to query an instant.",
        json_schema_extra={"example": "2016-04-01T00:00:00Z"},
    )
    # Deprecated aliases from the point/arc era.
    query_timestamp: Optional[str] = Field(None, description="Deprecated alias for 'start'")
    query_start_timestamp: Optional[str] = Field(None, description="Deprecated alias for 'start'")
    query_end_timestamp: Optional[str] = Field(None, description="Deprecated alias for 'end'")

    beta: float = Field(
        0.5,
        description=(
            "Temporal focus. 0 = pure semantic search, 0.5 = balanced, "
            "1 = absolute temporal dominance."
        ),
        ge=0.0,
        le=1.0,
    )
    top_k: int = Field(10, description="Results to return", ge=1, le=100)
    top_k_coarse: int = Field(200, description="Candidate pool size from pass 1", ge=1, le=5000)
    deduplicate: bool = Field(
        True, description="Collapse split representations of a chunk onto one result"
    )
    concept_filter: Optional[List[str]] = Field(
        None,
        description="Optional XBRL concepts to restrict fact chunks to",
        json_schema_extra={"example": ["NetIncomeLoss"]},
    )


class ScaleMatchModel(BaseModel):
    """Per-circle detail for one result."""

    scale: str
    overlaps: bool
    jaccard: float
    delta_phi_deg: float
    shared_segments: List[int] = []


class TemporalSearchResult(BaseModel):
    """One scored hit."""

    rank: int
    doc_id: str
    group_id: str
    text: str
    interval: Dict[str, Any]
    semantic_score: float
    temporal_alignment: float
    combined_score: float
    phi_difference_deg: float
    traversed_scales: List[str] = []
    scale_matches: List[ScaleMatchModel] = []
    metadata: dict = {}


class TemporalSearchResponse(BaseModel):
    query: str
    interval: Optional[Dict[str, Any]]
    beta: float
    traversed_scales: List[str]
    skipped_scales: List[Dict[str, str]]
    results: List[TemporalSearchResult]
    execution_time_ms: float


class DecomposeRequest(BaseModel):
    query: str = Field(
        ...,
        description="Natural-language query to decompose",
        json_schema_extra={"example": "Q1 impact on the full year for 2021, 2022 and 2023"},
    )
    reference: Optional[str] = Field(
        None, description="ISO 'now' for relative expressions like 'the last three years'"
    )
    max_subqueries: int = Field(24, ge=1, le=128)


class SubQueryModel(BaseModel):
    label: str
    interval: Optional[Dict[str, Any]]


class DecomposeResponse(BaseModel):
    query: str
    subquery_count: int
    truncated: bool
    anchors: List[int]
    granularities: List[str]
    subqueries: List[SubQueryModel]


class DecomposedSearchRequest(DecomposeRequest):
    beta: float = Field(0.5, ge=0.0, le=1.0)
    top_k: int = Field(10, ge=1, le=100)


class DecomposedSearchResponse(BaseModel):
    query: str
    decomposition: DecomposeResponse
    results: List[TemporalSearchResult]
    execution_time_ms: float


class IngestRequest(BaseModel):
    documents: List[Document]


class IngestResponse(BaseModel):
    """
    ``ingested_count`` counts source chunks; ``representation_count`` counts rows
    written, which is larger when chunks were split at period boundaries.
    """

    ingested_count: int
    representation_count: int
    doc_ids: List[str]
    group_ids: List[str]


class ScaleModel(BaseModel):
    name: str
    period_years: float
    segments: int
    segment_label: str
    weight: float


class StatsResponse(BaseModel):
    total_representations: int
    total_documents: int
    embedding_model: str
    vector_store_type: str
    schema_version: int
    epoch: str
    year_convention: str
    temporal_dimensions: int
    coverage_end_year: int
    scales: List[ScaleModel]
    fingerprint: str


# ============================================================================
# Application
# ============================================================================

@asynccontextmanager
async def lifespan(_app: FastAPI):
    """
    Build the hierarchy, embedding client and store, then optionally load the demo
    corpus. Everything below the yield is teardown; the in-memory and Chroma stores
    need none, so there is nothing there.
    """
    global embedding_client, hierarchy

    hierarchy = TemporalHierarchy()
    embedding_client = _build_embedding_client()
    _wire(_build_vector_store())

    if os.getenv("LOAD_DEMO_DATA", "true").lower() == "true" and vector_store.count() == 0:
        print("Loading demo dataset (IBM reports 2015-2024)...")
        reports = generate_ibm_report_intervals()
        written = ingestion_pipeline.ingest_batch(
            texts=[text for text, _ in reports],
            intervals=[interval for _, interval in reports],
            doc_ids=[_demo_doc_id(interval) for _, interval in reports],
        )
        print(f"✓ Loaded {len(reports)} documents as {len(written)} representations")

    print(f"✓ Ready: {vector_store.count()} representations under {hierarchy.fingerprint()}")
    yield


app = FastAPI(
    title="Temporal-Phase Spin Retrieval API",
    description=(
        "Hierarchical phase-encoded temporal vectors: time as a continuous geometric "
        "phase on concurrent circular scales, concatenated onto a frozen semantic "
        "embedding. No retraining, no metadata joins."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

vector_store: Optional[VectorStore] = None
embedding_client = None
ingestion_pipeline: Optional[TemporalSpinIngestionPipeline] = None
retriever: Optional[TemporalSpinRetriever] = None
hierarchy: TemporalHierarchy = DEFAULT_HIERARCHY


def _build_embedding_client():
    use_openai = os.getenv("USE_OPENAI_EMBEDDINGS", "true").lower() == "true"
    use_mock = os.getenv("USE_MOCK_EMBEDDINGS", "false").lower() == "true"

    if use_openai:
        model_name = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
        client = OpenAIEmbeddingClient(model=model_name)
        print(f"✓ OpenAI embeddings: {model_name} ({client.dimension}-dim)")
        return client
    if use_mock:
        client = MockEmbeddingClient(dimension=384)
        print("✓ MockEmbeddingClient (set USE_OPENAI_EMBEDDINGS=true for OpenAI)")
        return client
    url = os.getenv("LLAMASTACK_URL", "http://localhost:8000")
    model_name = os.getenv("EMBEDDING_MODEL", "text-embedding-v1")
    print(f"✓ LlamaStack: {url}, model: {model_name}")
    return LlamaStackEmbeddingClient(base_url=url, model_name=model_name)


def _build_vector_store() -> VectorStore:
    store_type = os.getenv("VECTOR_STORE", "memory").lower()
    if store_type == "chroma":
        persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        print(f"✓ ChromaVectorStore: {persist_dir}")
        return ChromaVectorStore(
            collection_name="temporal_spin",
            persist_directory=persist_dir,
            hierarchy=hierarchy,
        )
    print("✓ InMemoryVectorStore")
    return InMemoryVectorStore(hierarchy=hierarchy)


def _wire(store: VectorStore) -> None:
    """Point the pipeline and retriever at a store under the active hierarchy."""
    global vector_store, ingestion_pipeline, retriever
    vector_store = store
    ingestion_pipeline = TemporalSpinIngestionPipeline(
        embedding_client=embedding_client, vector_store=store, hierarchy=hierarchy
    )
    retriever = TemporalSpinRetriever(
        embedding_client=embedding_client,
        vector_store=store,
        hierarchy=hierarchy,
        default_beta=float(os.getenv("DEFAULT_BETA", "0.5")),
    )


def _demo_doc_id(interval: TemporalInterval) -> str:
    """
    Stable id for a demo report.

    The id becomes the ``group_id``, so it must distinguish the FY2017 annual report
    from the 2017-2022 review — both start in 2017, and colliding would make
    deduplication collapse two genuinely different documents into one.
    """
    first = interval.start.year
    last = (interval.end.year - 1) if interval.end else first
    return f"ibm-report-{first}" if last <= first else f"ibm-report-{first}-{last}"


# ============================================================================
# Endpoints
# ============================================================================


@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Temporal Spin Retrieval API"}


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Corpus counts plus the temporal header the corpus was encoded under."""
    if vector_store is None:
        raise HTTPException(status_code=503, detail="System not initialized")

    model_name = getattr(
        embedding_client, "model", getattr(embedding_client, "model_name", None)
    ) or type(embedding_client).__name__

    total = vector_store.count()
    groups = vector_store.count_groups() if hasattr(vector_store, "count_groups") else total

    return StatsResponse(
        total_representations=total,
        total_documents=groups,
        embedding_model=model_name,
        vector_store_type=type(vector_store).__name__,
        schema_version=hierarchy.schema_version,
        epoch=hierarchy.epoch.isoformat(),
        year_convention=hierarchy.year_convention,
        temporal_dimensions=hierarchy.dimensions,
        coverage_end_year=hierarchy.coverage_end_year,
        scales=[
            ScaleModel(
                name=s.name,
                period_years=s.period_years,
                segments=s.segments,
                segment_label=s.segment_label,
                weight=s.weight,
            )
            for s in hierarchy.scales
        ],
        fingerprint=hierarchy.fingerprint(),
    )


@app.post("/clear")
async def clear_database():
    """Empty the store and rebuild the pipeline and retriever against it."""
    if vector_store is None or embedding_client is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    try:
        _wire(_build_vector_store())
        vector_store.clear()
        print("✓ Vector store cleared")
        return {
            "status": "success",
            "message": "Vector store cleared",
            "vector_store_type": type(vector_store).__name__,
            "hierarchy": hierarchy.fingerprint(),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


def _format_results(results) -> List[TemporalSearchResult]:
    formatted: List[TemporalSearchResult] = []
    for r in results:
        formatted.append(
            TemporalSearchResult(
                rank=r.rank,
                doc_id=r.doc_id,
                group_id=r.group_id,
                text=r.text[:500],
                interval=_interval_payload(r.interval),
                semantic_score=r.semantic_score,
                temporal_alignment=r.temporal_alignment,
                combined_score=r.combined_score,
                phi_difference_deg=math.degrees(r.phi_difference),
                traversed_scales=list(r.traversed_scales),
                scale_matches=[
                    ScaleMatchModel(
                        scale=name,
                        overlaps=m.overlaps,
                        jaccard=m.jaccard,
                        delta_phi_deg=math.degrees(m.delta_phi),
                        shared_segments=list(m.shared_segments),
                    )
                    for name, m in r.scale_matches.items()
                ],
                metadata=r.metadata or {},
            )
        )
    return formatted


@app.post("/temporal_search", response_model=TemporalSearchResponse)
async def temporal_search(request: TemporalSearchRequest):
    """
    Two-pass temporal-phase retrieval.

    Pass 1 gathers candidates on ``0.9 * semantic + 0.1 * temporal``. Pass 2 walks the
    hierarchy under a lazy traversal plan: circles the query arc saturates are
    skipped, since an overlap test against a full circle can only succeed. Documents
    failing arc overlap at any traversed circle are rejected outright; the rest are
    scored ``(1 - beta) * semantic + beta * temporal``.
    """
    if retriever is None:
        raise HTTPException(status_code=503, detail="System not initialized")

    start = time.time()
    interval = _interval_from_fields(
        request.start or request.query_start_timestamp,
        request.end or request.query_end_timestamp,
        request.query_timestamp,
    )

    try:
        query = retriever.create_query(request.query, interval=interval)
        results = retriever.search(
            query_text=request.query,
            interval=interval,
            beta=request.beta,
            top_k_coarse=request.top_k_coarse,
            top_k_final=request.top_k,
            concept_filter=request.concept_filter,
            deduplicate=request.deduplicate,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Search failed: {exc}")

    return TemporalSearchResponse(
        query=request.query,
        interval=_interval_payload(interval) if interval else None,
        beta=request.beta,
        traversed_scales=list(query.plan.scale_names),
        skipped_scales=[{"scale": name, "reason": reason} for name, reason in query.plan.skipped],
        results=_format_results(results),
        execution_time_ms=(time.time() - start) * 1000,
    )


def _decomposition_payload(decomposition) -> DecomposeResponse:
    return DecomposeResponse(
        query=decomposition.query_text,
        subquery_count=len(decomposition),
        truncated=decomposition.truncated,
        anchors=list(decomposition.anchors),
        granularities=list(decomposition.granularities),
        subqueries=[
            SubQueryModel(
                label=s.label,
                interval=_interval_payload(s.interval) if s.interval else None,
            )
            for s in decomposition.subqueries
        ],
    )


@app.post("/decompose", response_model=DecomposeResponse)
async def decompose_query(request: DecomposeRequest):
    """
    Show how a natural-language query splits into temporal sub-queries.

    "Q1 impact on the full year for 2021, 2022 and 2023" is six questions: the first
    quarter of each year, and each year entire. This endpoint runs the parse without
    executing any retrieval, so a caller can inspect the fan-out first.
    """
    reference = _parse_iso(request.reference, "reference")
    return _decomposition_payload(
        decompose(request.query, reference=reference, max_subqueries=request.max_subqueries)
    )


@app.post("/decomposed_search", response_model=DecomposedSearchResponse)
async def decomposed_search(request: DecomposedSearchRequest):
    """
    Decompose a query, run every sub-query concurrently, and merge the results.

    Results are deduplicated on ``group_id``, so a chunk satisfying several
    sub-queries appears once at its best score. Each result's metadata carries
    ``matched_subqueries``, naming which parts of the question it answered.
    """
    if retriever is None:
        raise HTTPException(status_code=503, detail="System not initialized")

    start = time.time()
    reference = _parse_iso(request.reference, "reference")
    try:
        results, decomposition = search_decomposed(
            retriever,
            request.query,
            beta=request.beta,
            top_k_final=request.top_k,
            reference=reference,
            max_subqueries=request.max_subqueries,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Decomposed search failed: {exc}")

    return DecomposedSearchResponse(
        query=request.query,
        decomposition=_decomposition_payload(decomposition),
        results=_format_results(results),
        execution_time_ms=(time.time() - start) * 1000,
    )


@app.post("/ingest", response_model=IngestResponse)
async def ingest_documents(request: IngestRequest):
    """
    Index documents with interval encoding.

    Supply ``start`` and ``end`` to encode a period; supply ``start`` alone for an
    instant; supply neither and the period is extracted from the text. A period
    crossing a boundary produces several rows sharing one ``group_id``, which is why
    ``representation_count`` can exceed ``ingested_count``.
    """
    if ingestion_pipeline is None:
        raise HTTPException(status_code=503, detail="System not initialized")

    try:
        texts: List[str] = []
        intervals: List[Optional[TemporalInterval]] = []
        doc_ids: List[Optional[str]] = []
        metadatas: List[Optional[dict]] = []

        for doc in request.documents:
            texts.append(doc.text)
            intervals.append(
                _interval_from_fields(doc.start or doc.timestamp, doc.end or doc.end_timestamp)
            )
            doc_ids.append(doc.doc_id)
            metadatas.append(doc.metadata)

        written = ingestion_pipeline.ingest_batch(
            texts=texts, intervals=intervals, doc_ids=doc_ids, metadatas=metadatas
        )

        return IngestResponse(
            ingested_count=len(request.documents),
            representation_count=len(written),
            doc_ids=[d.doc_id for d in written],
            group_ids=sorted({d.group_id for d in written}),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}")


@app.post("/beta_sweep")
async def beta_sweep(
    query: str = Body(..., embed=True),
    start: Optional[str] = Body(None, embed=True),
    end: Optional[str] = Body(None, embed=True),
    beta_values: List[float] = Body([0.0, 0.25, 0.5, 0.75, 1.0], embed=True),
    top_k: int = Body(5, embed=True),
):
    """
    Run one query at several β values.

    β shifts prioritisation between semantic and temporal evidence at query time.
    Nothing is re-embedded and nothing is re-indexed between the sweeps — the whole
    point is that temporal focus is a runtime knob.
    """
    if retriever is None:
        raise HTTPException(status_code=503, detail="System not initialized")

    for beta in beta_values:
        if not 0.0 <= beta <= 1.0:
            raise HTTPException(status_code=400, detail=f"beta must be in [0, 1], got {beta}")

    interval = _interval_from_fields(start, end)
    try:
        sweep = retriever.search_with_beta_sweep(
            query_text=query, interval=interval, beta_values=beta_values, top_k=top_k
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Beta sweep failed: {exc}")

    return {
        "query": query,
        "interval": _interval_payload(interval) if interval else None,
        "beta_values": beta_values,
        "results_by_beta": {
            f"beta_{beta}": [
                {
                    "rank": r.rank,
                    "doc_id": r.doc_id,
                    "group_id": r.group_id,
                    "interval": _interval_payload(r.interval),
                    "combined_score": r.combined_score,
                    "semantic_score": r.semantic_score,
                    "temporal_alignment": r.temporal_alignment,
                    "phi_diff_deg": math.degrees(r.phi_difference),
                }
                for r in results
            ]
            for beta, results in sweep
        },
    }


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8080"))
    host = os.getenv("HOST", "0.0.0.0")

    print("=" * 80)
    print("TEMPORAL-PHASE SPIN RETRIEVAL API")
    print("=" * 80)
    print(f"Starting server on {host}:{port}")
    print()
    print("Environment:")
    print(f"  USE_OPENAI_EMBEDDINGS: {os.getenv('USE_OPENAI_EMBEDDINGS', 'true')}")
    print(f"  USE_MOCK_EMBEDDINGS  : {os.getenv('USE_MOCK_EMBEDDINGS', 'false')}")
    print(f"  VECTOR_STORE         : {os.getenv('VECTOR_STORE', 'memory')}")
    print(f"  LOAD_DEMO_DATA       : {os.getenv('LOAD_DEMO_DATA', 'true')}")
    print(f"  DEFAULT_BETA         : {os.getenv('DEFAULT_BETA', '0.5')}")
    print()
    print(f"API docs: http://localhost:{port}/docs")
    print("=" * 80)

    uvicorn.run("api:app", host=host, port=port, reload=False)
