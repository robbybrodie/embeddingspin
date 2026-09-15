"""
Vector Store Abstraction
========================

Backends for storing modified semantic vectors — ``[semantic | temporal]`` — and
retrieving them by similarity.

Every stored row carries its temporal **header** alongside the vector. The header
declares the schema version, the epoch, the year convention, and the period of each
tuple present. A reader therefore never has to assume the temporal block is nine
dimensions: it reads the tuple count from the header and strips exactly that many
triples off the tail of the embedding. That is what makes the hierarchy extensible
without a migration — a corpus written with three circles and one written with four
can sit in the same collection and still be parsed unambiguously.

What a store does *not* do is deduplicate. A chunk split across period boundaries is
several rows here on purpose, so that geometric overlap works at every boundary;
collapsing them back to one result happens at the application layer, on ``group_id``.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from temporal_config import DEFAULT_HIERARCHY, TemporalHierarchy
from temporal_encoding import TemporalEncoding, TemporalInterval
from temporal_spin import SpinDocument, cosine_similarity

logger = logging.getLogger(__name__)

# Metadata keys the store owns. Everything else in a document's metadata is treated
# as caller-supplied and round-trips untouched.
_RESERVED_KEYS = frozenset(
    {"temporal_encoding", "temporal_header", "group_id", "interval_start", "interval_end"}
)


class VectorStore(ABC):
    """
    Base class for backends.

    Subclasses must preserve the temporal encoding losslessly: a document written and
    read back must produce identical tuples, header and ``group_id``.
    """

    #: Hierarchy the stored vectors were encoded under. Set on first write.
    hierarchy: Optional[TemporalHierarchy] = None

    def set_hierarchy(self, hierarchy: TemporalHierarchy) -> None:
        """Record the hierarchy this store's contents are encoded under."""
        if self.hierarchy is not None and not self.hierarchy.is_compatible_with(hierarchy):
            raise ValueError(
                "cannot mix incompatible hierarchies in one store:\n"
                f"  existing: {self.hierarchy.fingerprint()}\n"
                f"  incoming: {hierarchy.fingerprint()}"
            )
        self.hierarchy = hierarchy

    @abstractmethod
    def add_documents(self, documents: List[SpinDocument]) -> None:
        """Write documents. Each representation of a split chunk is its own row."""

    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[SpinDocument, float]]:
        """Return up to ``top_k`` ``(document, similarity)`` pairs, best first."""

    @abstractmethod
    def get_document(self, doc_id: str) -> Optional[SpinDocument]:
        """Fetch one representation by its unique id."""

    @abstractmethod
    def count(self) -> int:
        """Number of stored representations (not distinct chunks)."""

    @abstractmethod
    def clear(self) -> None:
        """Remove everything."""

    # -- shared helpers ---------------------------------------------------

    def _encode_metadata(self, doc: SpinDocument) -> Dict[str, Any]:
        """Flatten a document's temporal state into scalar-valued metadata."""
        encoding = doc.encoding.to_dict()
        metadata: Dict[str, Any] = {
            "temporal_encoding": json.dumps(encoding),
            "group_id": doc.group_id,
            "interval_start": doc.interval.start.isoformat(),
            "interval_end": doc.interval.end.isoformat() if doc.interval.end else "",
        }
        for key, value in (doc.metadata or {}).items():
            if key in _RESERVED_KEYS:
                continue
            if isinstance(value, (str, int, float, bool)):
                metadata[key] = value
        return metadata

    def _decode_document(
        self,
        doc_id: str,
        text: str,
        metadata: Dict[str, Any],
        full_embedding: Optional[List[float]],
    ) -> SpinDocument:
        """Rebuild a ``SpinDocument`` from a stored row."""
        encoding = TemporalEncoding.from_dict(json.loads(metadata["temporal_encoding"]))

        # The header declares how many tuples were written, so the split point
        # between the semantic and temporal blocks is known exactly rather than
        # assumed. This is what lets three-circle and four-circle vectors coexist.
        semantic_embedding: List[float] = []
        if full_embedding:
            temporal_dims = 3 * len(encoding.tuples)
            if len(full_embedding) > temporal_dims:
                semantic_embedding = list(full_embedding[:-temporal_dims])

        user_metadata = {
            k: v for k, v in metadata.items() if k not in _RESERVED_KEYS
        }
        user_metadata["group_id"] = encoding.group_id

        return SpinDocument(
            doc_id=doc_id,
            text=text,
            semantic_embedding=semantic_embedding,
            encoding=encoding,
            full_embedding=list(full_embedding) if full_embedding else [],
            group_id=encoding.group_id,
            metadata=user_metadata,
        )


# ============================================================================
# In-memory
# ============================================================================


class InMemoryVectorStore(VectorStore):
    """
    Brute-force store for prototyping and tests. Suitable below ~10k rows.

    Supports the same metadata filter grammar as Chroma (``$and``, ``$or``, ``$in``,
    ``$ne``, and bare equality) so that switching backends does not change query
    behaviour.
    """

    def __init__(self, hierarchy: Optional[TemporalHierarchy] = None) -> None:
        self.documents: Dict[str, SpinDocument] = {}
        self.hierarchy = hierarchy

    def add_documents(self, documents: List[SpinDocument]) -> None:
        for doc in documents:
            if self.hierarchy is None:
                self.hierarchy = doc.encoding.hierarchy
            self.documents[doc.doc_id] = doc

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[SpinDocument, float]]:
        scored: List[Tuple[SpinDocument, float]] = []
        for doc in self.documents.values():
            if filter_dict and not _matches_filter(doc.metadata, filter_dict):
                continue
            scored.append((doc, cosine_similarity(query_embedding, doc.full_embedding)))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]

    def get_document(self, doc_id: str) -> Optional[SpinDocument]:
        return self.documents.get(doc_id)

    def get_group(self, group_id: str) -> List[SpinDocument]:
        """Every representation sharing a ``group_id``, in representation order."""
        found = [d for d in self.documents.values() if d.group_id == group_id]
        return sorted(found, key=lambda d: d.encoding.representation_index)

    def count(self) -> int:
        return len(self.documents)

    def count_groups(self) -> int:
        """Number of distinct chunks, collapsing split representations."""
        return len({d.group_id for d in self.documents.values()})

    def clear(self) -> None:
        self.documents.clear()


def _matches_filter(metadata: Dict[str, Any], clause: Dict[str, Any]) -> bool:
    """Evaluate a Chroma-style metadata filter against one document's metadata."""
    for key, condition in clause.items():
        if key == "$and":
            if not all(_matches_filter(metadata, sub) for sub in condition):
                return False
        elif key == "$or":
            if not any(_matches_filter(metadata, sub) for sub in condition):
                return False
        elif isinstance(condition, dict):
            value = metadata.get(key)
            for operator, operand in condition.items():
                if operator == "$in" and value not in operand:
                    return False
                if operator == "$nin" and value in operand:
                    return False
                if operator == "$ne" and value == operand:
                    return False
                if operator == "$eq" and value != operand:
                    return False
        elif metadata.get(key) != condition:
            return False
    return True


# ============================================================================
# Chroma
# ============================================================================


class ChromaVectorStore(VectorStore):
    """
    Chroma backend. Install with ``pip install chromadb``.

    Chroma requires a fixed dimensionality per collection. Extending the hierarchy
    therefore changes the vector width and needs a new collection — which is exactly
    why the header records the tuple count, so the two collections remain
    individually interpretable and a reader can tell them apart.
    """

    def __init__(
        self,
        collection_name: str = "temporal_spin_collection",
        persist_directory: Optional[str] = None,
        hierarchy: Optional[TemporalHierarchy] = None,
    ) -> None:
        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError("chromadb not installed. Install with: pip install chromadb") from exc

        self.client = (
            chromadb.PersistentClient(path=persist_directory)
            if persist_directory
            else chromadb.Client()
        )
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Hierarchical phase-encoded temporal vectors"},
        )
        self.hierarchy = hierarchy

    def add_documents(self, documents: List[SpinDocument]) -> None:
        if not documents:
            return
        for doc in documents:
            if self.hierarchy is None:
                self.hierarchy = doc.encoding.hierarchy
        self.collection.add(
            ids=[d.doc_id for d in documents],
            embeddings=[d.full_embedding for d in documents],
            metadatas=[self._encode_metadata(d) for d in documents],
            documents=[d.text for d in documents],
        )

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[SpinDocument, float]]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_dict,
            include=["documents", "metadatas", "distances", "embeddings"],
        )
        if not results.get("ids") or not results["ids"][0]:
            return []

        out: List[Tuple[SpinDocument, float]] = []
        for i, doc_id in enumerate(results["ids"][0]):
            embedding = _as_list(_index(results, "embeddings", i))
            metadata = results["metadatas"][0][i]
            distance = _index(results, "distances", i) or 0.0
            try:
                doc = self._decode_document(
                    doc_id, results["documents"][0][i], metadata, embedding
                )
            except (KeyError, ValueError, json.JSONDecodeError):
                logger.warning("skipping row %s: unreadable temporal encoding", doc_id)
                continue
            # Chroma returns squared L2 distance over normalised vectors.
            out.append((doc, 1.0 - (distance ** 2) / 2.0))
        return out

    def get_document(self, doc_id: str) -> Optional[SpinDocument]:
        results = self.collection.get(ids=[doc_id], include=["documents", "metadatas", "embeddings"])
        if not results["ids"]:
            return None
        embedding = _as_list(results["embeddings"][0]) if results.get("embeddings") else None
        return self._decode_document(
            doc_id, results["documents"][0], results["metadatas"][0], embedding
        )

    def count(self) -> int:
        return self.collection.count()

    def clear(self) -> None:
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"description": "Hierarchical phase-encoded temporal vectors"},
        )


def _index(results: Dict[str, Any], key: str, i: int) -> Any:
    block = results.get(key)
    if block is None or len(block) == 0 or block[0] is None or len(block[0]) <= i:
        return None
    return block[0][i]


def _as_list(value: Any) -> Optional[List[float]]:
    if value is None:
        return None
    return value.tolist() if hasattr(value, "tolist") else list(value)


# ============================================================================
# PostgreSQL + pgvector
# ============================================================================


class PGVectorStore(VectorStore):
    """
    PostgreSQL backend using the pgvector extension.

    Setup::

        CREATE EXTENSION vector;
        pip install psycopg2-binary

    The temporal header and tuples are stored in a JSONB column rather than being
    spread across typed columns. That keeps the schema stable when the hierarchy is
    extended: adding a fourth circle changes the vector width and the JSON contents,
    but not the table definition.

    ``group_id`` is indexed because deduplication reads it on every query.
    """

    def __init__(
        self,
        connection_string: str,
        table_name: str = "spin_documents",
        embedding_dim: int = 1545,  # 1536 semantic + 9 temporal
        hierarchy: Optional[TemporalHierarchy] = None,
    ) -> None:
        try:
            import psycopg2  # noqa: F401
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "psycopg2 not installed. Install with: pip install psycopg2-binary"
            ) from exc

        self.connection_string = connection_string
        self.table_name = table_name
        self.embedding_dim = embedding_dim
        self.hierarchy = hierarchy
        self._init_table()

    def _connect(self):
        import psycopg2

        return psycopg2.connect(self.connection_string)

    def _init_table(self) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    doc_id            TEXT PRIMARY KEY,
                    group_id          TEXT NOT NULL,
                    text              TEXT NOT NULL,
                    interval_start    TIMESTAMPTZ NOT NULL,
                    interval_end      TIMESTAMPTZ,
                    temporal_encoding JSONB NOT NULL,
                    embedding         vector({self.embedding_dim}) NOT NULL,
                    metadata          JSONB
                );
                """
            )
            cur.execute(
                f"""CREATE INDEX IF NOT EXISTS {self.table_name}_embedding_idx
                    ON {self.table_name} USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100);"""
            )
            cur.execute(
                f"""CREATE INDEX IF NOT EXISTS {self.table_name}_group_idx
                    ON {self.table_name} (group_id);"""
            )
            conn.commit()

    def add_documents(self, documents: List[SpinDocument]) -> None:
        from psycopg2.extras import Json

        with self._connect() as conn, conn.cursor() as cur:
            for doc in documents:
                if self.hierarchy is None:
                    self.hierarchy = doc.encoding.hierarchy
                cur.execute(
                    f"""
                    INSERT INTO {self.table_name}
                        (doc_id, group_id, text, interval_start, interval_end,
                         temporal_encoding, embedding, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s::vector, %s)
                    ON CONFLICT (doc_id) DO UPDATE SET
                        group_id          = EXCLUDED.group_id,
                        text              = EXCLUDED.text,
                        interval_start    = EXCLUDED.interval_start,
                        interval_end      = EXCLUDED.interval_end,
                        temporal_encoding = EXCLUDED.temporal_encoding,
                        embedding         = EXCLUDED.embedding,
                        metadata          = EXCLUDED.metadata;
                    """,
                    (
                        doc.doc_id,
                        doc.group_id,
                        doc.text,
                        doc.interval.start,
                        doc.interval.end,
                        Json(doc.encoding.to_dict()),
                        "[" + ",".join(map(str, doc.full_embedding)) + "]",
                        Json(
                            {
                                k: v
                                for k, v in (doc.metadata or {}).items()
                                if k not in _RESERVED_KEYS
                            }
                        ),
                    ),
                )
            conn.commit()

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[SpinDocument, float]]:
        vector = "[" + ",".join(map(str, query_embedding)) + "]"
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT doc_id, text, temporal_encoding, metadata, embedding,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM {self.table_name}
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
                """,
                (vector, vector, top_k),
            )
            out: List[Tuple[SpinDocument, float]] = []
            for doc_id, text, encoding_json, metadata, embedding, similarity in cur.fetchall():
                merged = dict(metadata or {})
                merged["temporal_encoding"] = json.dumps(encoding_json)
                out.append(
                    (
                        self._decode_document(doc_id, text, merged, _parse_pg_vector(embedding)),
                        float(similarity),
                    )
                )
            return out

    def get_document(self, doc_id: str) -> Optional[SpinDocument]:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"""SELECT doc_id, text, temporal_encoding, metadata, embedding
                    FROM {self.table_name} WHERE doc_id = %s;""",
                (doc_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            doc_id, text, encoding_json, metadata, embedding = row
            merged = dict(metadata or {})
            merged["temporal_encoding"] = json.dumps(encoding_json)
            return self._decode_document(doc_id, text, merged, _parse_pg_vector(embedding))

    def count(self) -> int:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self.table_name};")
            return cur.fetchone()[0]

    def count_groups(self) -> int:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(DISTINCT group_id) FROM {self.table_name};")
            return cur.fetchone()[0]

    def clear(self) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(f"DELETE FROM {self.table_name};")
            conn.commit()


def _parse_pg_vector(value: Any) -> Optional[List[float]]:
    """pgvector returns its vector type as a bracketed string."""
    if value is None:
        return None
    if isinstance(value, str):
        return [float(x) for x in value.strip("[]").split(",") if x]
    return list(value)
