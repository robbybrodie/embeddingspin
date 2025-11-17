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
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from temporal_spin import (
    T0_SECONDS,
    QUARTER_SCALE_YEARS,
    DECADE_SCALE_YEARS,
    CENTURY_SCALE_YEARS,
    QUARTER_WEIGHT,
    DECADE_WEIGHT,
    CENTURY_WEIGHT
)
from llamastack_client import LlamaStackEmbeddingClient, MockEmbeddingClient
from openai_client import OpenAIEmbeddingClient
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
    timestamp: Optional[str] = Field(None, description="ISO format timestamp (optional, start for arcs)")
    end_timestamp: Optional[str] = Field(None, description="ISO format end timestamp for arc encoding (optional)")
    doc_id: Optional[str] = Field(None, description="Document ID (optional)")
    metadata: Optional[dict] = Field(None, description="Additional metadata (optional)")


class TemporalSearchRequest(BaseModel):
    """Request for temporal spin search."""
    query: str = Field(..., description="Search query text", example="IBM revenue 2016")
    
    # Arc-based query (preferred for period matching)
    query_start_timestamp: Optional[str] = Field(
        None,
        description="ISO format arc start timestamp for period queries",
        example="2016-01-01T00:00:00Z"
    )
    query_end_timestamp: Optional[str] = Field(
        None,
        description="ISO format arc end timestamp for period queries",
        example="2016-03-31T23:59:59Z"
    )
    
    # Point-based query (fallback)
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
    
    # NEW: Concept filtering for precision retrieval
    concept_filter: Optional[List[str]] = Field(
        None,
        description="Optional list of XBRL concepts to filter retrieval (e.g., ['NetIncomeLoss', 'Revenues'])",
        example=["NetIncomeLoss", "IncomeLossFromContinuingOperations"]
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
    metadata: dict = {}  # Include metadata for agent orchestration


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
    temporal_encoding: str = "multi-scale"
    period_years: List[float]  # [quarter, decade, century]
    period_weights: List[float]  # [quarter_weight, decade_weight, century_weight]


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
    
    # Determine which embedding client to use
    use_openai = os.getenv("USE_OPENAI_EMBEDDINGS", "true").lower() == "true"  # DEFAULT TO OPENAI
    use_mock = os.getenv("USE_MOCK_EMBEDDINGS", "false").lower() == "true"  # NO MOCK BY DEFAULT
    
    # Initialize embedding client
    if use_openai:
        # Use OpenAI's best embedding model
        model_name = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
        embedding_client = OpenAIEmbeddingClient(model=model_name)
        print(f"✓ Using OpenAI Embeddings: {model_name} ({embedding_client.dimension}-dim)")
    elif use_mock:
        embedding_client = MockEmbeddingClient(dimension=384)
        print("✓ Using MockEmbeddingClient (set USE_OPENAI_EMBEDDINGS=true for OpenAI)")
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
    
    # Determine correct embedding model name
    if hasattr(embedding_client, 'model'):
        # OpenAI client
        model_name = embedding_client.model
    elif hasattr(embedding_client, 'model_name'):
        # LlamaStack client
        model_name = embedding_client.model_name
    else:
        # MockEmbeddingClient or unknown
        model_name = type(embedding_client).__name__
    
    return StatsResponse(
        total_documents=vector_store.count(),
        embedding_model=model_name,
        vector_store_type=type(vector_store).__name__,
        t0_epoch=datetime.fromtimestamp(T0_SECONDS).isoformat(),
        temporal_encoding="multi-scale (1/16/256 years)",
        period_years=[QUARTER_SCALE_YEARS, DECADE_SCALE_YEARS, CENTURY_SCALE_YEARS],
        period_weights=[QUARTER_WEIGHT, DECADE_WEIGHT, CENTURY_WEIGHT]
    )


@app.post("/clear")
async def clear_database():
    """
    Clear all documents from the vector store.
    
    This endpoint recreates the vector store, effectively removing all documents.
    Useful before starting a fresh ingestion.
    """
    global vector_store, ingestion_pipeline, retriever
    
    if vector_store is None or embedding_client is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        # Get current config
        persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        dimension = embedding_client.dimension
        use_chroma = os.getenv("USE_CHROMA", "true").lower() == "true"
        
        # Recreate vector store (this clears ChromaDB)
        if use_chroma:
            vector_store = ChromaVectorStore(
                collection_name="temporal_spin",
                persist_directory=persist_dir
            )
        else:
            vector_store = InMemoryVectorStore(dimension=dimension)
        
        # Reinitialize ingestion pipeline
        ingestion_pipeline = TemporalSpinIngestionPipeline(
            embedding_client=embedding_client,
            vector_store=vector_store,
            t0_seconds=T0_SECONDS
            # period_seconds deprecated - multi-scale encoding
        )
        
        # Reinitialize retriever
        retriever = TemporalSpinRetriever(
            embedding_client=embedding_client,
            vector_store=vector_store,
            t0_seconds=T0_SECONDS
            # period_seconds deprecated - multi-scale encoding
        )
        
        print(f"✓ Vector database cleared successfully")
        
        return {
            "status": "success",
            "message": "Vector database cleared",
            "dimension": dimension,
            "vector_store_type": type(vector_store).__name__,
            "persist_directory": persist_dir if use_chroma else None
        }
    except Exception as e:
        print(f"✗ Error clearing database: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
    
    # Parse arc or point query timestamps
    query_timestamp = None
    query_start_timestamp = None
    query_end_timestamp = None
    
    # Prefer arc queries over point queries
    if request.query_start_timestamp and request.query_end_timestamp:
        # Arc query
        try:
            query_start_timestamp = datetime.fromisoformat(request.query_start_timestamp.replace('Z', '+00:00'))
            query_end_timestamp = datetime.fromisoformat(request.query_end_timestamp.replace('Z', '+00:00'))
            print(f"🔍 API: Arc query parsed: {query_start_timestamp} to {query_end_timestamp}")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid arc timestamp format")
    elif request.query_timestamp:
        # Point query (fallback)
        try:
            query_timestamp = datetime.fromisoformat(request.query_timestamp.replace('Z', '+00:00'))
            print(f"🔍 API: Point query parsed: {query_timestamp}")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid timestamp format")
    else:
        print(f"🔍 API: No timestamp provided, will use NOW")
    
    # Execute search with arc or point query
    try:
        print(f"🔍 API: Calling retriever.search() with:")
        print(f"   query_timestamp={query_timestamp}")
        print(f"   query_start_timestamp={query_start_timestamp}")
        print(f"   query_end_timestamp={query_end_timestamp}")
        if request.concept_filter:
            print(f"   concept_filter={request.concept_filter[:5]}{'...' if len(request.concept_filter) > 5 else ''}")
        
        results = retriever.search(
            query_text=request.query,
            query_timestamp=query_timestamp,
            query_start_timestamp=query_start_timestamp,
            query_end_timestamp=query_end_timestamp,
            beta=request.beta,
            top_k_final=request.top_k,
            concept_filter=request.concept_filter  # NEW: Pass concept filter
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
    
    execution_time_ms = (time.time() - start_time) * 1000
    
    # Format results
    formatted_results = []
    for result in results:
        import math
        # Extract metadata from the SpinDocument
        metadata = result.metadata if hasattr(result, 'metadata') and result.metadata else {}
        
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
            phi_difference_deg=math.degrees(result.phi_difference),
            metadata=metadata  # Include metadata for agent orchestration
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
        end_timestamps = []
        doc_ids = []
        metadatas = []
        
        for doc in request.documents:
            texts.append(doc.text)
            
            # Parse timestamp if provided (start timestamp for arcs)
            if doc.timestamp:
                try:
                    ts = datetime.fromisoformat(doc.timestamp.replace('Z', '+00:00'))
                    timestamps.append(ts)
                except ValueError:
                    timestamps.append(None)
            else:
                timestamps.append(None)
            
            # Parse end_timestamp if provided (for arc encoding)
            if doc.end_timestamp:
                try:
                    end_ts = datetime.fromisoformat(doc.end_timestamp.replace('Z', '+00:00'))
                    end_timestamps.append(end_ts)
                except ValueError:
                    end_timestamps.append(None)
            else:
                end_timestamps.append(None)
            
            doc_ids.append(doc.doc_id)
            metadatas.append(doc.metadata)
        
        # Ingest batch with arc encoding support
        ingested_docs = ingestion_pipeline.ingest_batch(
            texts=texts,
            timestamps=timestamps,
            end_timestamps=end_timestamps,
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

