"""
Vector Database Abstraction Layer
==================================

Provides a unified interface to different vector databases (PGVector, Chroma)
for storing and retrieving temporal-phase spin embeddings.

This abstraction allows the spin retrieval system to work with any backend
without changing the core algorithm.
"""

import json
import uuid
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import asdict

from temporal_spin import SpinDocument, cosine_similarity


class VectorStore(ABC):
    """
    Abstract base class for vector database backends.
    
    Implementations must support:
    - Storing documents with high-dimensional embeddings
    - Retrieving top-k similar documents by cosine similarity
    - Storing and retrieving metadata
    """
    
    @abstractmethod
    def add_documents(self, documents: List[SpinDocument]) -> None:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of SpinDocument objects with embeddings
        """
        pass
    
    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[SpinDocument, float]]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            filter_dict: Optional metadata filters
        
        Returns:
            List of (document, similarity_score) tuples
        """
        pass
    
    @abstractmethod
    def get_document(self, doc_id: str) -> Optional[SpinDocument]:
        """Retrieve a document by ID."""
        pass
    
    @abstractmethod
    def count(self) -> int:
        """Return total number of documents."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Remove all documents."""
        pass


class InMemoryVectorStore(VectorStore):
    """
    Simple in-memory vector store for prototyping and testing.
    
    Stores all documents in RAM and performs brute-force cosine similarity
    search. Suitable for datasets < 10k documents.
    
    For production with larger datasets, use ChromaVectorStore or
    PGVectorStore instead.
    """
    
    def __init__(self):
        """Initialize empty in-memory store."""
        self.documents: Dict[str, SpinDocument] = {}
        self.embeddings: Dict[str, List[float]] = {}
    
    def add_documents(self, documents: List[SpinDocument]) -> None:
        """Add documents to in-memory store."""
        for doc in documents:
            self.documents[doc.doc_id] = doc
            self.embeddings[doc.doc_id] = doc.full_embedding
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[SpinDocument, float]]:
        """
        Brute-force cosine similarity search.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            filter_dict: Optional metadata filters (not implemented)
        
        Returns:
            List of (document, similarity_score) tuples, sorted by score
        """
        if not self.documents:
            return []
        
        # Compute similarities for all documents
        scores = []
        for doc_id, doc in self.documents.items():
            embedding = self.embeddings[doc_id]
            score = cosine_similarity(query_embedding, embedding)
            scores.append((doc, score))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return scores[:top_k]
    
    def get_document(self, doc_id: str) -> Optional[SpinDocument]:
        """Retrieve document by ID."""
        return self.documents.get(doc_id)
    
    def count(self) -> int:
        """Return number of stored documents."""
        return len(self.documents)
    
    def clear(self) -> None:
        """Clear all documents."""
        self.documents.clear()
        self.embeddings.clear()


class ChromaVectorStore(VectorStore):
    """
    Chroma DB vector store implementation.
    
    Chroma is a lightweight vector database that's easy to set up and
    works well for prototypes and medium-sized datasets (< 1M docs).
    
    Installation:
        pip install chromadb
    
    Usage:
        store = ChromaVectorStore(
            collection_name="temporal_spin_docs",
            persist_directory="./chroma_db"
        )
    """
    
    def __init__(
        self,
        collection_name: str = "temporal_spin_collection",
        persist_directory: Optional[str] = None,
        embedding_function=None
    ):
        """
        Initialize Chroma vector store.
        
        Args:
            collection_name: Name of the Chroma collection
            persist_directory: Directory to persist data (None = in-memory)
            embedding_function: Optional Chroma embedding function
                               (we manage embeddings ourselves, so pass None)
        """
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ImportError(
                "chromadb not installed. Install with: pip install chromadb"
            )
        
        # Create Chroma client
        if persist_directory:
            self.client = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=persist_directory
            ))
        else:
            self.client = chromadb.Client()
        
        # Get or create collection
        # embedding_function=None means we provide embeddings ourselves
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Temporal-phase spin embeddings"}
        )
    
    def add_documents(self, documents: List[SpinDocument]) -> None:
        """Add documents to Chroma collection."""
        if not documents:
            return
        
        ids = [doc.doc_id for doc in documents]
        embeddings = [doc.full_embedding for doc in documents]
        
        # Prepare metadata (Chroma requires JSON-serializable values)
        metadatas = []
        documents_text = []
        for doc in documents:
            metadata = {
                "timestamp": doc.timestamp.isoformat(),
                "phi": doc.phi,
                "spin_vector": json.dumps(doc.spin_vector),
            }
            if doc.metadata:
                # Add custom metadata (ensure JSON-serializable)
                for k, v in doc.metadata.items():
                    if isinstance(v, (str, int, float, bool)):
                        metadata[k] = v
            
            metadatas.append(metadata)
            documents_text.append(doc.text)
        
        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents_text
        )
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[SpinDocument, float]]:
        """
        Search Chroma collection.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results
            filter_dict: Optional metadata filters (Chroma where clause)
        
        Returns:
            List of (SpinDocument, similarity_score) tuples
        """
        # Query Chroma (uses cosine similarity by default)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_dict
        )
        
        # Parse results
        documents = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                text = results["documents"][0][i]
                metadata = results["metadatas"][0][i]
                distance = results["distances"][0][i] if "distances" in results else 0
                
                # Reconstruct SpinDocument
                timestamp = datetime.fromisoformat(metadata["timestamp"])
                phi = metadata["phi"]
                spin_vector = json.loads(metadata["spin_vector"])
                
                # Note: We don't have the original semantic embedding here,
                # but for retrieval we only need these fields
                doc = SpinDocument(
                    doc_id=doc_id,
                    text=text,
                    timestamp=timestamp,
                    semantic_embedding=[],  # Not stored separately
                    spin_vector=spin_vector,
                    phi=phi,
                    full_embedding=[],  # Can retrieve if needed
                    metadata={k: v for k, v in metadata.items() 
                             if k not in ["timestamp", "phi", "spin_vector"]}
                )
                
                # Convert distance to similarity (Chroma returns L2 distance)
                # For normalized vectors: similarity ≈ 1 - (distance²/2)
                similarity = 1.0 - (distance ** 2) / 2.0
                
                documents.append((doc, similarity))
        
        return documents
    
    def get_document(self, doc_id: str) -> Optional[SpinDocument]:
        """Retrieve document by ID."""
        results = self.collection.get(ids=[doc_id])
        if not results["ids"]:
            return None
        
        text = results["documents"][0]
        metadata = results["metadatas"][0]
        
        timestamp = datetime.fromisoformat(metadata["timestamp"])
        phi = metadata["phi"]
        spin_vector = json.loads(metadata["spin_vector"])
        
        return SpinDocument(
            doc_id=doc_id,
            text=text,
            timestamp=timestamp,
            semantic_embedding=[],
            spin_vector=spin_vector,
            phi=phi,
            full_embedding=[],
            metadata={k: v for k, v in metadata.items() 
                     if k not in ["timestamp", "phi", "spin_vector"]}
        )
    
    def count(self) -> int:
        """Return number of documents in collection."""
        return self.collection.count()
    
    def clear(self) -> None:
        """Clear all documents from collection."""
        # Chroma doesn't have a clear method, so delete and recreate
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.create_collection(
            name=self.collection.name,
            metadata={"description": "Temporal-phase spin embeddings"}
        )


class PGVectorStore(VectorStore):
    """
    PostgreSQL with pgvector extension.
    
    pgvector adds vector similarity search to PostgreSQL, making it suitable
    for production deployments with large datasets.
    
    Setup:
        1. Install PostgreSQL with pgvector extension
        2. CREATE EXTENSION vector;
        3. pip install psycopg2-binary
    
    Usage:
        store = PGVectorStore(
            connection_string="postgresql://user:pass@localhost:5432/vectordb",
            table_name="spin_documents"
        )
    """
    
    def __init__(
        self,
        connection_string: str,
        table_name: str = "spin_documents",
        embedding_dim: int = 386  # 384 semantic + 2 spin
    ):
        """
        Initialize PGVector store.
        
        Args:
            connection_string: PostgreSQL connection string
            table_name: Table name for documents
            embedding_dim: Dimension of full embeddings
        """
        try:
            import psycopg2
            from psycopg2.extras import Json
        except ImportError:
            raise ImportError(
                "psycopg2 not installed. Install with: pip install psycopg2-binary"
            )
        
        self.connection_string = connection_string
        self.table_name = table_name
        self.embedding_dim = embedding_dim
        
        # Create table if it doesn't exist
        self._init_table()
    
    def _get_connection(self):
        """Create a new database connection."""
        import psycopg2
        return psycopg2.connect(self.connection_string)
    
    def _init_table(self):
        """Create table with pgvector extension."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                # Enable pgvector extension
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                
                # Create table
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.table_name} (
                        doc_id TEXT PRIMARY KEY,
                        text TEXT NOT NULL,
                        timestamp TIMESTAMPTZ NOT NULL,
                        phi DOUBLE PRECISION NOT NULL,
                        spin_vector JSONB NOT NULL,
                        embedding vector({self.embedding_dim}) NOT NULL,
                        metadata JSONB
                    );
                """)
                
                # Create index for vector similarity search
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS {self.table_name}_embedding_idx
                    ON {self.table_name}
                    USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100);
                """)
                
                conn.commit()
    
    def add_documents(self, documents: List[SpinDocument]) -> None:
        """Insert documents into PostgreSQL."""
        import psycopg2
        from psycopg2.extras import Json
        
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                for doc in documents:
                    # Convert embedding to string format for pgvector
                    embedding_str = "[" + ",".join(map(str, doc.full_embedding)) + "]"
                    
                    cur.execute(f"""
                        INSERT INTO {self.table_name}
                        (doc_id, text, timestamp, phi, spin_vector, embedding, metadata)
                        VALUES (%s, %s, %s, %s, %s, %s::vector, %s)
                        ON CONFLICT (doc_id) DO UPDATE SET
                            text = EXCLUDED.text,
                            timestamp = EXCLUDED.timestamp,
                            phi = EXCLUDED.phi,
                            spin_vector = EXCLUDED.spin_vector,
                            embedding = EXCLUDED.embedding,
                            metadata = EXCLUDED.metadata;
                    """, (
                        doc.doc_id,
                        doc.text,
                        doc.timestamp,
                        doc.phi,
                        Json(doc.spin_vector),
                        embedding_str,
                        Json(doc.metadata or {})
                    ))
                
                conn.commit()
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[SpinDocument, float]]:
        """
        Vector similarity search using pgvector.
        
        Uses cosine similarity (1 - cosine_distance) for ranking.
        """
        embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"
        
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                # Cosine similarity: 1 - (embedding <=> query)
                query = f"""
                    SELECT doc_id, text, timestamp, phi, spin_vector, metadata,
                           1 - (embedding <=> %s::vector) AS similarity
                    FROM {self.table_name}
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s;
                """
                
                cur.execute(query, (embedding_str, embedding_str, top_k))
                rows = cur.fetchall()
                
                documents = []
                for row in rows:
                    doc_id, text, timestamp, phi, spin_vector, metadata, similarity = row
                    
                    doc = SpinDocument(
                        doc_id=doc_id,
                        text=text,
                        timestamp=timestamp,
                        semantic_embedding=[],  # Not stored separately
                        spin_vector=spin_vector,
                        phi=phi,
                        full_embedding=[],
                        metadata=metadata or {}
                    )
                    
                    documents.append((doc, similarity))
                
                return documents
    
    def get_document(self, doc_id: str) -> Optional[SpinDocument]:
        """Retrieve document by ID."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"""
                    SELECT doc_id, text, timestamp, phi, spin_vector, metadata
                    FROM {self.table_name}
                    WHERE doc_id = %s;
                """, (doc_id,))
                
                row = cur.fetchone()
                if not row:
                    return None
                
                doc_id, text, timestamp, phi, spin_vector, metadata = row
                
                return SpinDocument(
                    doc_id=doc_id,
                    text=text,
                    timestamp=timestamp,
                    semantic_embedding=[],
                    spin_vector=spin_vector,
                    phi=phi,
                    full_embedding=[],
                    metadata=metadata or {}
                )
    
    def count(self) -> int:
        """Count total documents."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {self.table_name};")
                return cur.fetchone()[0]
    
    def clear(self) -> None:
        """Delete all documents."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"DELETE FROM {self.table_name};")
                conn.commit()

