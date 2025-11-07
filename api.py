"""
FastAPI Server for Temporal-Phase Spin Retrieval
=================================================

RESTful API endpoints for temporal spin retrieval system.

Endpoints:
- POST /temporal_search: Execute temporal spin search with β parameter
- POST /ingest: Ingest new documents
- GET /health: Health check
- GET /stats: System statistics

This API allows interactive experimentation with the β (zoom) parameter
to demonstrate smooth temporal focus adjustment.
"""

import os
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import JSONResponse

from temporal_spin import T0_SECONDS, PERIOD_SECONDS
from llamastack_client import LlamaStackEmbeddingClient, MockEmbeddingClient
from vector_store import InMemoryVectorStore, ChromaVectorStore, VectorStore
from ingestion import TemporalSpinIngestionPipeline
from retrieval import TemporalSpinRetriever
from demo_data import generate_ibm_reports


# ============================================================================
# Pydantic Models for API
# ============================================================================

class Document(BaseModel):
    """Document for ingestion."""
    text: str = Field(..., description="Document text content")
    timestamp: Optional[str] = Field(None, description="ISO format timestamp (optional)")
    doc_id: Optional[str] = Field(None, description="Document ID (optional)")
    metadata: Optional[dict] = Field(None, description="Additional metadata (optional)")


class TemporalSearchRequest(BaseModel):
    """Request for temporal spin search."""
    query: str = Field(..., description="Search query text", example="IBM revenue 2016")
    query_timestamp: Optional[str] = Field(
        None,
        description="ISO format query timestamp (default: now)",
        example="2016-06-30T00:00:00Z"
    )
    beta: float = Field(
        5000.0,
        description="Temporal zoom factor (0=pure semantic, 100=weak, 1000=moderate, 5000=strong [default], 10000+=extreme)",
        example=5000.0,
        ge=0.0,
        le=10000.0
    )
    top_k: int = Field(
        10,
        description="Number of results to return",
        ge=1,
        le=100
    )


class TemporalSearchResult(BaseModel):
    """Single search result."""
    rank: int
    doc_id: str
    text: str
    timestamp: str
    semantic_score: float
    temporal_alignment: float
    combined_score: float
    phi_doc: float
    phi_query: float
    phi_difference_deg: float


class TemporalSearchResponse(BaseModel):
    """Response for temporal search."""
    query: str
    query_timestamp: str
    beta: float
    results: List[TemporalSearchResult]
    execution_time_ms: float


class IngestRequest(BaseModel):
    """Request to ingest documents."""
    documents: List[Document]


class IngestResponse(BaseModel):
    """Response for document ingestion."""
    ingested_count: int
    doc_ids: List[str]


class StatsResponse(BaseModel):
    """System statistics."""
    total_documents: int
    embedding_model: str
    vector_store_type: str
    t0_epoch: str
    period_years: float


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Temporal-Phase Spin Retrieval API",
    description=(
        "API for temporal-phase spin retrieval - a novel algorithm that encodes "
        "time as angular spin on the unit circle, enabling smooth temporal zoom "
        "without model retraining."
    ),
    version="1.0.0"
)


# Global state (initialized on startup)
vector_store: Optional[VectorStore] = None
embedding_client = None
ingestion_pipeline: Optional[TemporalSpinIngestionPipeline] = None
retriever: Optional[TemporalSpinRetriever] = None


@app.on_event("startup")
async def startup_event():
    """Initialize system on startup."""
    global vector_store, embedding_client, ingestion_pipeline, retriever
    
    # Determine whether to use mock embeddings
    use_mock = os.getenv("USE_MOCK_EMBEDDINGS", "true").lower() == "true"
    
    # Initialize embedding client
    if use_mock:
        embedding_client = MockEmbeddingClient(dimension=384)
        print("✓ Using MockEmbeddingClient (set USE_MOCK_EMBEDDINGS=false for LlamaStack)")
    else:
        llamastack_url = os.getenv("LLAMASTACK_URL", "http://localhost:8000")
        model_name = os.getenv("EMBEDDING_MODEL", "text-embedding-v1")
        embedding_client = LlamaStackEmbeddingClient(
            base_url=llamastack_url,
            model_name=model_name
        )
        print(f"✓ Using LlamaStack: {llamastack_url}, model: {model_name}")
    
    # Initialize vector store
    store_type = os.getenv("VECTOR_STORE", "memory").lower()
    if store_type == "memory":
        vector_store = InMemoryVectorStore()
        print("✓ Using InMemoryVectorStore")
    elif store_type == "chroma":
        persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        vector_store = ChromaVectorStore(
            collection_name="temporal_spin",
            persist_directory=persist_dir
        )
        print(f"✓ Using ChromaVectorStore: {persist_dir}")
    else:
        vector_store = InMemoryVectorStore()
        print("✓ Defaulting to InMemoryVectorStore")
    
    # Initialize ingestion pipeline
    ingestion_pipeline = TemporalSpinIngestionPipeline(
        embedding_client=embedding_client,
        vector_store=vector_store
    )
    
    # Initialize retriever
    retriever = TemporalSpinRetriever(
        embedding_client=embedding_client,
        vector_store=vector_store
    )
    
    # Load demo data if enabled
    load_demo = os.getenv("LOAD_DEMO_DATA", "true").lower() == "true"
    if load_demo and vector_store.count() == 0:
        print("Loading demo dataset (IBM reports 2015-2024)...")
        reports = generate_ibm_reports()
        texts = [text for text, _ in reports]
        timestamps = [ts for _, ts in reports]
        doc_ids = [f"ibm-report-{ts.year}" for _, ts in reports]
        
        ingestion_pipeline.ingest_batch(
            texts=texts,
            timestamps=timestamps,
            doc_ids=doc_ids
        )
        print(f"✓ Loaded {len(reports)} demo documents")
    
    print(f"✓ System ready: {vector_store.count()} documents indexed")


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "message": "Temporal Spin Retrieval API"}


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get system statistics."""
    if vector_store is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    return StatsResponse(
        total_documents=vector_store.count(),
        embedding_model=getattr(embedding_client, 'model_name', 'mock-embed'),
        vector_store_type=type(vector_store).__name__,
        t0_epoch=datetime.fromtimestamp(T0_SECONDS).isoformat(),
        period_years=PERIOD_SECONDS / (365.25 * 24 * 3600)
    )


@app.post("/temporal_search", response_model=TemporalSearchResponse)
async def temporal_search(request: TemporalSearchRequest):
    """
    Execute temporal-phase spin retrieval.
    
    This endpoint demonstrates the core innovation: adjusting β (beta) smoothly
    transitions from broad semantic search to temporally-focused retrieval.
    
    Try different β values:
    - β = 0: Pure semantic search (time ignored)
    - β = 100: Weak temporal preference
    - β = 1000: Moderate temporal focus
    - β = 5000: Strong temporal focus (exact year prioritized) [DEFAULT]
    - β = 10000: Extreme temporal filter
    
    No model retraining required - β is a runtime parameter!
    """
    if retriever is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    import time
    start_time = time.time()
    
    # Parse query timestamp
    query_timestamp = None
    if request.query_timestamp:
        try:
            query_timestamp = datetime.fromisoformat(request.query_timestamp.replace('Z', '+00:00'))
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid timestamp format")
    
    # Execute search
    try:
        results = retriever.search(
            query_text=request.query,
            query_timestamp=query_timestamp,
            beta=request.beta,
            top_k_final=request.top_k
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
    
    execution_time_ms = (time.time() - start_time) * 1000
    
    # Format results
    formatted_results = []
    for result in results:
        import math
        formatted_results.append(TemporalSearchResult(
            rank=result.rank,
            doc_id=result.doc_id,
            text=result.text[:500],  # Truncate for API response
            timestamp=result.timestamp.isoformat(),
            semantic_score=result.semantic_score,
            temporal_alignment=result.temporal_alignment,
            combined_score=result.combined_score,
            phi_doc=result.phi_doc,
            phi_query=result.phi_query,
            phi_difference_deg=math.degrees(result.phi_difference)
        ))
    
    return TemporalSearchResponse(
        query=request.query,
        query_timestamp=(query_timestamp or datetime.now()).isoformat(),
        beta=request.beta,
        results=formatted_results,
        execution_time_ms=execution_time_ms
    )


@app.post("/ingest", response_model=IngestResponse)
async def ingest_documents(request: IngestRequest):
    """
    Ingest new documents with temporal spin encoding.
    
    Documents can include explicit timestamps or they will be inferred
    from text content using regex patterns and dateutil.
    """
    if ingestion_pipeline is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        texts = []
        timestamps = []
        doc_ids = []
        metadatas = []
        
        for doc in request.documents:
            texts.append(doc.text)
            
            # Parse timestamp if provided
            if doc.timestamp:
                try:
                    ts = datetime.fromisoformat(doc.timestamp.replace('Z', '+00:00'))
                    timestamps.append(ts)
                except ValueError:
                    timestamps.append(None)
            else:
                timestamps.append(None)
            
            doc_ids.append(doc.doc_id)
            metadatas.append(doc.metadata)
        
        # Ingest batch
        ingested_docs = ingestion_pipeline.ingest_batch(
            texts=texts,
            timestamps=timestamps,
            doc_ids=doc_ids,
            metadatas=metadatas
        )
        
        return IngestResponse(
            ingested_count=len(ingested_docs),
            doc_ids=[doc.doc_id for doc in ingested_docs]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.post("/beta_sweep")
async def beta_sweep(
    query: str = Body(..., embed=True),
    query_timestamp: Optional[str] = Body(None, embed=True),
    beta_values: List[float] = Body([0, 1, 5, 10, 20], embed=True),
    top_k: int = Body(5, embed=True)
):
    """
    Execute search with multiple β values to demonstrate temporal zoom.
    
    This endpoint shows how β acts as a "temporal zoom knob" - smoothly
    adjusting from broad semantic search to sharp temporal focus.
    """
    if retriever is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    # Parse query timestamp
    query_ts = None
    if query_timestamp:
        try:
            query_ts = datetime.fromisoformat(query_timestamp.replace('Z', '+00:00'))
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid timestamp format")
    
    # Execute sweep
    try:
        sweep_results = retriever.search_with_beta_sweep(
            query_text=query,
            query_timestamp=query_ts,
            beta_values=beta_values,
            top_k=top_k
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Beta sweep failed: {str(e)}")
    
    # Format response
    import math
    formatted = {}
    for beta, results in sweep_results:
        formatted[f"beta_{beta}"] = [
            {
                "rank": r.rank,
                "doc_id": r.doc_id,
                "timestamp": r.timestamp.isoformat(),
                "combined_score": r.combined_score,
                "semantic_score": r.semantic_score,
                "temporal_alignment": r.temporal_alignment,
                "phi_diff_deg": math.degrees(r.phi_difference)
            }
            for r in results
        ]
    
    return {
        "query": query,
        "query_timestamp": (query_ts or datetime.now()).isoformat(),
        "beta_values": beta_values,
        "results_by_beta": formatted
    }


# ============================================================================
# Main Entry Point
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
    print("Environment Variables:")
    print(f"  USE_MOCK_EMBEDDINGS: {os.getenv('USE_MOCK_EMBEDDINGS', 'true')}")
    print(f"  VECTOR_STORE: {os.getenv('VECTOR_STORE', 'memory')}")
    print(f"  LOAD_DEMO_DATA: {os.getenv('LOAD_DEMO_DATA', 'true')}")
    print()
    print("API Documentation: http://localhost:8080/docs")
    print("=" * 80)
    
    uvicorn.run(
        "api:app",
        host=host,
        port=port,
        reload=False
    )

