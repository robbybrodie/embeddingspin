"""
Temporal-Phase Spin Ingestion Pipeline
=======================================

Handles document ingestion with timestamp extraction and spin encoding.

Pipeline:
1. Extract or infer timestamp from document
2. Obtain semantic embedding from LlamaStack
3. Compute temporal spin vector
4. Concatenate embeddings
5. Store in vector database
"""

import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
import os

from temporal_spin import (
    SpinDocument,
    compute_spin_vector,
    extract_timestamp_from_text,
    T0_SECONDS,
    PERIOD_SECONDS
)
from llamastack_client import LlamaStackEmbeddingClient, MockEmbeddingClient
from vector_store import VectorStore


class TemporalSpinIngestionPipeline:
    """
    Pipeline for ingesting documents with temporal-phase spin encoding.
    
    This pipeline:
    1. Accepts raw text documents with optional timestamps
    2. Extracts or infers timestamps from content or metadata
    3. Obtains semantic embeddings from LlamaStack Model Gateway
    4. Computes 2D spin vectors from timestamps
    5. Concatenates semantic + spin into full embeddings
    6. Stores in vector database
    
    No model retraining required - spin encoding is applied post-hoc.
    """
    
    def __init__(
        self,
        embedding_client: LlamaStackEmbeddingClient,
        vector_store: VectorStore,
        t0_seconds: float = T0_SECONDS,
        period_seconds: float = PERIOD_SECONDS,
        spin_weight: float = 1.0
    ):
        """
        Initialize ingestion pipeline.
        
        Args:
            embedding_client: Client for obtaining semantic embeddings
            vector_store: Vector database for storage
            t0_seconds: Base epoch for timestamp normalization
            period_seconds: Period for spin encoding (default: 10 years)
            spin_weight: Scaling factor for spin vector (default: 1.0)
        """
        self.embedding_client = embedding_client
        self.vector_store = vector_store
        self.t0_seconds = t0_seconds
        self.period_seconds = period_seconds
        self.spin_weight = spin_weight
    
    def ingest_document(
        self,
        text: str,
        timestamp: Optional[datetime] = None,
        doc_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> SpinDocument:
        """
        Ingest a single document.
        
        Args:
            text: Document text
            timestamp: Optional explicit timestamp (if None, will be extracted)
            doc_id: Optional document ID (if None, will be generated)
            metadata: Optional metadata dictionary
        
        Returns:
            SpinDocument with embeddings and spin encoding
        """
        # Generate ID if not provided
        if doc_id is None:
            doc_id = str(uuid.uuid4())
        
        # Extract or infer timestamp
        if timestamp is None:
            timestamp = extract_timestamp_from_text(
                text,
                fallback=datetime.now(timezone.utc)
            )
        
        # Ensure timezone-aware
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        
        # Get semantic embedding from LlamaStack
        semantic_embedding = self.embedding_client.embed_single(text)
        
        # Compute temporal spin vector
        timestamp_seconds = timestamp.timestamp()
        spin_vector, phi = compute_spin_vector(
            timestamp_seconds,
            self.t0_seconds,
            self.period_seconds
        )
        
        # Scale spin vector by weight
        weighted_spin = [self.spin_weight * x for x in spin_vector]
        
        # Concatenate: full_embedding = [semantic_embedding, weighted_spin]
        full_embedding = semantic_embedding + weighted_spin
        
        # Create SpinDocument
        doc = SpinDocument(
            doc_id=doc_id,
            text=text,
            timestamp=timestamp,
            semantic_embedding=semantic_embedding,
            spin_vector=spin_vector,
            phi=phi,
            full_embedding=full_embedding,
            metadata=metadata or {}
        )
        
        # Store in vector database
        self.vector_store.add_documents([doc])
        
        return doc
    
    def ingest_batch(
        self,
        texts: List[str],
        timestamps: Optional[List[datetime]] = None,
        doc_ids: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> List[SpinDocument]:
        """
        Ingest multiple documents in a batch (more efficient).
        
        Args:
            texts: List of document texts
            timestamps: Optional list of timestamps (None = auto-extract)
            doc_ids: Optional list of document IDs
            metadatas: Optional list of metadata dicts
        
        Returns:
            List of SpinDocument objects
        """
        n = len(texts)
        
        # Handle optional arguments
        if timestamps is None:
            timestamps = [None] * n
        if doc_ids is None:
            doc_ids = [str(uuid.uuid4()) for _ in range(n)]
        if metadatas is None:
            metadatas = [{}] * n
        
        # Extract timestamps where needed
        resolved_timestamps = []
        for i, (text, ts) in enumerate(zip(texts, timestamps)):
            if ts is None:
                ts = extract_timestamp_from_text(
                    text,
                    fallback=datetime.now(timezone.utc)
                )
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            resolved_timestamps.append(ts)
        
        # Batch embedding request (efficient!)
        semantic_embeddings = self.embedding_client.embed(texts)
        
        # Create SpinDocument objects
        documents = []
        for i in range(n):
            # Compute spin vector
            timestamp_seconds = resolved_timestamps[i].timestamp()
            spin_vector, phi = compute_spin_vector(
                timestamp_seconds,
                self.t0_seconds,
                self.period_seconds
            )
            
            # Scale and concatenate
            weighted_spin = [self.spin_weight * x for x in spin_vector]
            full_embedding = semantic_embeddings[i] + weighted_spin
            
            # Create document
            doc = SpinDocument(
                doc_id=doc_ids[i],
                text=texts[i],
                timestamp=resolved_timestamps[i],
                semantic_embedding=semantic_embeddings[i],
                spin_vector=spin_vector,
                phi=phi,
                full_embedding=full_embedding,
                metadata=metadatas[i]
            )
            documents.append(doc)
        
        # Batch insert into vector store
        self.vector_store.add_documents(documents)
        
        return documents
    
    def ingest_from_files(
        self,
        file_paths: List[str],
        extract_timestamp_from_filename: bool = True
    ) -> List[SpinDocument]:
        """
        Ingest documents from files.
        
        Args:
            file_paths: List of file paths to ingest
            extract_timestamp_from_filename: Try to parse timestamp from filename
        
        Returns:
            List of ingested SpinDocument objects
        """
        texts = []
        timestamps = []
        doc_ids = []
        metadatas = []
        
        for file_path in file_paths:
            # Read file
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Try to extract timestamp
            timestamp = None
            if extract_timestamp_from_filename:
                filename = os.path.basename(file_path)
                try:
                    timestamp = extract_timestamp_from_text(filename)
                except Exception:
                    pass
            
            # Fallback to file modification time
            if timestamp is None:
                mtime = os.path.getmtime(file_path)
                timestamp = datetime.fromtimestamp(mtime, tz=timezone.utc)
            
            texts.append(text)
            timestamps.append(timestamp)
            doc_ids.append(file_path)  # Use file path as ID
            metadatas.append({"file_path": file_path})
        
        return self.ingest_batch(texts, timestamps, doc_ids, metadatas)


def create_ingestion_pipeline(
    vector_store: VectorStore,
    llamastack_url: Optional[str] = None,
    model_name: str = "text-embedding-v1",
    use_mock_embeddings: bool = False,
    embedding_dim: int = 384
) -> TemporalSpinIngestionPipeline:
    """
    Convenience factory to create an ingestion pipeline.
    
    Args:
        vector_store: Vector database instance
        llamastack_url: LlamaStack API URL (or use LLAMASTACK_URL env var)
        model_name: Embedding model name
        use_mock_embeddings: Use mock embeddings for testing
        embedding_dim: Embedding dimension (for mock client)
    
    Returns:
        Configured TemporalSpinIngestionPipeline
    """
    if use_mock_embeddings:
        embedding_client = MockEmbeddingClient(
            model_name="mock-embed",
            dimension=embedding_dim
        )
    else:
        embedding_client = LlamaStackEmbeddingClient(
            base_url=llamastack_url,
            model_name=model_name
        )
    
    return TemporalSpinIngestionPipeline(
        embedding_client=embedding_client,
        vector_store=vector_store
    )

