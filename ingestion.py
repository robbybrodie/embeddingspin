"""
Ingestion Pipeline
==================

Turns raw chunks into indexed, temporally-encoded vectors:

1. Resolve the chunk's temporal interval — supplied explicitly, or extracted from
   the text.
2. Embed the text with the frozen semantic model (one batched call per batch).
3. Encode the interval across every circle in the hierarchy, splitting at period
   boundaries into one or more representations.
4. Concatenate each temporal vector onto the semantic vector.
5. Write every representation to the store under a shared ``group_id``.

Step 3 is where a chunk can become several rows. A 10-K covering 2017 through 2022
crosses five 1-year boundaries and is indexed six times: same text, same semantic
embedding, same ``group_id``, six different 1-year arcs. Its 16-year and 256-year
tuples are identical across all six, because the span sits inside one block at those
scales. Retrieval deduplicates on ``group_id``.

Precision is fixed here, at ingestion, by how many circles the hierarchy declares —
nothing is discarded later. What retrieval varies is how deeply it *traverses* that
encoding, not how much of it exists.
"""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from temporal_config import DEFAULT_HIERARCHY, TemporalHierarchy
from temporal_encoding import MAX_REPRESENTATIONS, TemporalInterval, encode
from temporal_spin import SpinDocument, extract_interval_from_text, extract_timestamp_from_text
from vector_store import VectorStore

logger = logging.getLogger(__name__)


class TemporalSpinIngestionPipeline:
    """
    Post-hoc temporal augmentation of a frozen embedding model.

    The embedding model is never retrained or fine-tuned; the temporal vector is
    computed independently and appended. That is what keeps the approach
    model-agnostic and lets the semantic model be swapped without touching the
    temporal machinery.

    Args:
        embedding_client: Frozen semantic embedding model.
        vector_store: Destination backend.
        hierarchy: Scales and epoch. Must match the retriever's.
        split_at_boundaries: Emit one representation per component arc when an
            interval crosses a period boundary. Disabling this keeps one row per
            chunk but lets long arcs saturate, which blurs boundary-crossing
            documents.
        max_representations: Ceiling on rows emitted for a single chunk.
    """

    def __init__(
        self,
        embedding_client,
        vector_store: VectorStore,
        hierarchy: TemporalHierarchy = DEFAULT_HIERARCHY,
        split_at_boundaries: bool = True,
        max_representations: int = MAX_REPRESENTATIONS,
    ) -> None:
        self.embedding_client = embedding_client
        self.vector_store = vector_store
        self.hierarchy = hierarchy
        self.split_at_boundaries = split_at_boundaries
        self.max_representations = max_representations

        store_hierarchy = getattr(vector_store, "hierarchy", None)
        if store_hierarchy is None and hasattr(vector_store, "set_hierarchy"):
            vector_store.set_hierarchy(hierarchy)
        elif store_hierarchy is not None and not hierarchy.is_compatible_with(store_hierarchy):
            raise ValueError(
                "ingestion hierarchy is incompatible with the vector store's:\n"
                f"  pipeline: {hierarchy.fingerprint()}\n"
                f"  store   : {store_hierarchy.fingerprint()}"
            )

    # ------------------------------------------------------------------
    # Interval resolution
    # ------------------------------------------------------------------

    def resolve_interval(
        self,
        text: str,
        interval: Optional[TemporalInterval] = None,
        timestamp: Optional[datetime] = None,
        end_timestamp: Optional[datetime] = None,
    ) -> TemporalInterval:
        """
        Determine the half-open interval for a chunk.

        Precedence: an explicit ``interval``, then an explicit ``timestamp``
        (optionally with ``end_timestamp``), then extraction from the text. Text
        extraction prefers a period — "Q3 2023" becomes the full quarter, not a
        single instant — because a duration is what enables hierarchical matching
        between a quarterly filing and an annual query.
        """
        if interval is not None:
            return interval
        if timestamp is not None:
            return TemporalInterval(timestamp, end_timestamp)

        extracted = extract_interval_from_text(text)
        if extracted is not None:
            return extracted
        return TemporalInterval.point(
            extract_timestamp_from_text(text, fallback=datetime.now(timezone.utc))
        )

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_document(
        self,
        text: str,
        interval: Optional[TemporalInterval] = None,
        timestamp: Optional[datetime] = None,
        end_timestamp: Optional[datetime] = None,
        doc_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[SpinDocument]:
        """
        Ingest one chunk, returning every representation written.

        The returned list has one entry per component arc; for an interval that
        crosses no boundary it has exactly one.
        """
        return self.ingest_batch(
            texts=[text],
            intervals=[interval],
            timestamps=[timestamp],
            end_timestamps=[end_timestamp],
            doc_ids=[doc_id],
            metadatas=[metadata],
        )

    def ingest_batch(
        self,
        texts: Sequence[str],
        intervals: Optional[Sequence[Optional[TemporalInterval]]] = None,
        timestamps: Optional[Sequence[Optional[datetime]]] = None,
        end_timestamps: Optional[Sequence[Optional[datetime]]] = None,
        doc_ids: Optional[Sequence[Optional[str]]] = None,
        metadatas: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
    ) -> List[SpinDocument]:
        """
        Ingest many chunks with a single batched embedding call.

        Returns every representation written, flattened. The count can exceed
        ``len(texts)`` when chunks are split at boundaries — check ``group_id`` to
        recover the original grouping.
        """
        n = len(texts)
        intervals = list(intervals) if intervals else [None] * n
        timestamps = list(timestamps) if timestamps else [None] * n
        end_timestamps = list(end_timestamps) if end_timestamps else [None] * n
        doc_ids = list(doc_ids) if doc_ids else [None] * n
        metadatas = list(metadatas) if metadatas else [None] * n

        resolved = [
            self.resolve_interval(texts[i], intervals[i], timestamps[i], end_timestamps[i])
            for i in range(n)
        ]

        # One batched call to the frozen embedding model.
        embeddings = self.embedding_client.embed(list(texts))

        documents: List[SpinDocument] = []
        for i in range(n):
            group_id = doc_ids[i] or str(uuid.uuid4())
            encodings = encode(
                resolved[i],
                self.hierarchy,
                group_id=group_id,
                split=self.split_at_boundaries,
                max_representations=self.max_representations,
            )

            base_metadata = dict(metadatas[i] or {})
            for encoding in encodings:
                # A split chunk needs distinct primary keys but one shared identity.
                doc_id = (
                    group_id
                    if encoding.representation_count == 1
                    else f"{group_id}#{encoding.representation_index}"
                )
                metadata = dict(base_metadata)
                metadata.update(
                    {
                        "group_id": group_id,
                        "representation_index": encoding.representation_index,
                        "representation_count": encoding.representation_count,
                    }
                )
                documents.append(
                    SpinDocument(
                        doc_id=doc_id,
                        text=texts[i],
                        semantic_embedding=embeddings[i],
                        encoding=encoding,
                        group_id=group_id,
                        metadata=metadata,
                    )
                )

            if len(encodings) > 1:
                logger.debug(
                    "chunk %s spans %s and crosses %d boundary/ies -> %d representations",
                    group_id,
                    resolved[i],
                    len(encodings) - 1,
                    len(encodings),
                )

        self.vector_store.add_documents(documents)
        logger.info(
            "ingested %d chunk(s) as %d representation(s) under %s",
            n,
            len(documents),
            self.hierarchy.fingerprint(),
        )
        return documents

    def ingest_from_files(
        self,
        file_paths: Sequence[str],
        extract_interval_from_filename: bool = True,
    ) -> List[SpinDocument]:
        """
        Ingest whole files, deriving the interval from the filename, then the
        content, then the file's modification time.
        """
        texts: List[str] = []
        intervals: List[Optional[TemporalInterval]] = []
        doc_ids: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for path in file_paths:
            with open(path, "r", encoding="utf-8") as handle:
                text = handle.read()

            interval: Optional[TemporalInterval] = None
            if extract_interval_from_filename:
                interval = extract_interval_from_text(os.path.basename(path))
            if interval is None:
                interval = extract_interval_from_text(text)
            if interval is None:
                mtime = datetime.fromtimestamp(os.path.getmtime(path), tz=timezone.utc)
                interval = TemporalInterval.point(mtime)

            texts.append(text)
            intervals.append(interval)
            doc_ids.append(path)
            metadatas.append({"file_path": path})

        return self.ingest_batch(
            texts=texts, intervals=intervals, doc_ids=doc_ids, metadatas=metadatas
        )


def create_ingestion_pipeline(
    vector_store: VectorStore,
    llamastack_url: Optional[str] = None,
    model_name: str = "text-embedding-v1",
    use_mock_embeddings: bool = False,
    embedding_dim: int = 384,
    hierarchy: TemporalHierarchy = DEFAULT_HIERARCHY,
) -> TemporalSpinIngestionPipeline:
    """Convenience factory wiring an embedding client to a pipeline."""
    from llamastack_client import LlamaStackEmbeddingClient, MockEmbeddingClient

    if use_mock_embeddings:
        client = MockEmbeddingClient(model_name="mock-embed", dimension=embedding_dim)
    else:
        client = LlamaStackEmbeddingClient(base_url=llamastack_url, model_name=model_name)

    return TemporalSpinIngestionPipeline(
        embedding_client=client, vector_store=vector_store, hierarchy=hierarchy
    )
