"""
Multi-Pass Temporal-Phase Spin Retrieval
=========================================

Implements the two-pass retrieval algorithm:

Pass 1 - Coarse Recall:
    Use small λ for broad semantic search across all time periods.
    Retrieves top-K candidates using cosine similarity on full embeddings.

Pass 2 - Temporal Zoom Re-ranking:
    Recompute scores for top-K results using temporal alignment:
        score = semantic_similarity × exp(-β × (Δφ)²)

    where:
    - Δφ = smallest angular difference between query and document phases
    - β = zoom factor (0 = no temporal focus, 10+ = sharp temporal focus)

The β parameter acts as a "temporal zoom knob" - adjusting it lets you
smoothly transition from broad semantic search to temporally-focused retrieval.
"""

import logging
import math
from datetime import datetime, timezone
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

from llamastack_client import LlamaStackEmbeddingClient
from temporal_spin import (
    CENTURY_PERIOD_SECONDS,
    CENTURY_WEIGHT,
    DECADE_PERIOD_SECONDS,
    DECADE_WEIGHT,
    QUARTER_PERIOD_SECONDS,
    QUARTER_WEIGHT,
    T0_SECONDS,
    RetrievalResult,
    SpinDocument,
    SpinQuery,
    angular_difference,
    arc_overlap,
    compute_spin_vector,
    cosine_similarity,
    jaccard_similarity_arcs,
)
from vector_store import VectorStore


class TemporalSpinRetriever:
    """
    Multi-pass temporal-phase spin retrieval system.

    This retriever implements a two-stage approach:

    Stage 1: Coarse semantic recall using small λ to get candidates
    Stage 2: Temporal zoom re-ranking using β to focus on query time

    Key Innovation:
    ---------------
    No model retraining required. Time is encoded as a continuous angular
    coordinate on the unit circle, allowing smooth interpolation and
    explicit control over temporal alignment via the β parameter.

    β Parameter (Temporal Zoom Knob):
    ---------------------------------
    With 100-year period, phase spacing is ~3.6° per year:
    - β = 0.0: Pure semantic search (time ignored)
    - β = 0.3: Light temporal preference
    - β = 0.5: Balanced temporal-semantic weighting [DEFAULT]
    - β = 0.7: Strong temporal focus
    - β = 1.0: Temporal alignment dominates

    Combined score = (1-β) × semantic_similarity + β × temporal_alignment

    With β = 0.5 (50/50 balance):
    - Same quarter/year gets full temporal boost
    - Adjacent quarters still score well
    - Distant years (10+ years apart) are distinguished
    - Semantic similarity remains important

    Note: Lower β values (0.3-0.7) work well with 100-year period because
    tighter phase spacing provides stronger temporal signal.
    """

    def __init__(
        self,
        embedding_client: LlamaStackEmbeddingClient,
        vector_store: VectorStore,
        t0_seconds: float = T0_SECONDS,
        period_seconds: float = None,  # Deprecated - multi-scale now
        default_lambda: float = 1.0,
        default_beta: float = 0.5,
        temporal_scale: float = 1.0,
    ):
        """
        Initialize retriever.

        Args:
            embedding_client: Client for query embeddings
            vector_store: Vector database with documents
            t0_seconds: Base epoch for spin encoding
            period_seconds: Deprecated (kept for backward compatibility, ignored)
            default_lambda: Default weight for spin in Pass 1 (coarse recall)
            default_beta: Default zoom factor for Pass 2 (re-ranking)
            temporal_scale: Scaling factor for spin vectors (default: 1.0)
                           Note: Has no effect on cosine similarity (scale-invariant).
                           Must match the temporal_scale used during ingestion
        """
        self.embedding_client = embedding_client
        self.vector_store = vector_store
        self.t0_seconds = t0_seconds
        # period_seconds is deprecated - multi-scale encoding now used
        self.default_lambda = default_lambda
        self.default_beta = default_beta
        self.temporal_scale = temporal_scale

    def create_query(
        self,
        query_text: str,
        query_timestamp: Optional[datetime] = None,
        lambda_factor: Optional[float] = None,
        end_timestamp: Optional[datetime] = None,
    ) -> SpinQuery:
        """
        Create a SpinQuery with temporal encoding (point or arc mode).

        Args:
            query_text: Query string
            query_timestamp: Target timestamp (default: now, start for arcs)
            lambda_factor: Weight for spin component (default: self.default_lambda)
            end_timestamp: Optional end timestamp for arc queries

        Returns:
            SpinQuery with embeddings and spin encoding (point or arc)
        """
        print(
            f"🔍 CREATE_QUERY: query_ts={query_timestamp}, end_ts={end_timestamp}"
        )

        if query_timestamp is None:
            query_timestamp = datetime.now(timezone.utc)
            print(f"🔍 CREATE_QUERY: Fell back to NOW: {query_timestamp}")

        if query_timestamp.tzinfo is None:
            query_timestamp = query_timestamp.replace(tzinfo=timezone.utc)

        if lambda_factor is None:
            lambda_factor = self.default_lambda

        # Get semantic embedding
        semantic_embedding = self.embedding_client.embed_single(query_text)

        # Compute spin vector (point or arc mode, must match ingestion)
        query_seconds = query_timestamp.timestamp()
        end_seconds = end_timestamp.timestamp() if end_timestamp else None

        spin_vector, phi_centers, phi_starts, phi_ends = compute_spin_vector(
            query_seconds,
            self.t0_seconds,
            period_seconds=None,  # Deprecated - multi-scale encoding used
            temporal_scale=self.temporal_scale,
            end_timestamp_seconds=end_seconds,
        )

        # Create query object (handles concatenation internally)
        query = SpinQuery(
            query_text=query_text,
            query_timestamp=query_timestamp,
            semantic_embedding=semantic_embedding,
            spin_vector=spin_vector,
            phi=phi_centers,  # Now a dict with keys 'quarter', 'decade', 'century'
            lambda_factor=lambda_factor,
            end_timestamp=end_timestamp,
            phi_start=phi_starts,  # Now a dict
            phi_end=phi_ends,  # Now a dict
            is_arc=(end_timestamp is not None),
        )

        return query

    def search(
        self,
        query_text: str,
        query_timestamp: Optional[datetime] = None,
        query_start_timestamp: Optional[datetime] = None,
        query_end_timestamp: Optional[datetime] = None,
        beta: Optional[float] = None,
        lambda_coarse: float = 0.1,
        top_k_coarse: int = 50,
        top_k_final: int = 10,
        end_timestamp: Optional[
            datetime
        ] = None,  # Deprecated - use query_end_timestamp
        concept_filter: Optional[
            List[str]
        ] = None,  # NEW: Filter by XBRL concepts
    ) -> List[RetrievalResult]:
        """
        Execute two-pass temporal-phase spin retrieval.

        Pass 1 (Coarse Recall):
        -----------------------
        Use small λ (e.g., 0.1) to perform broad semantic search.
        This retrieves top_k_coarse candidates that are semantically relevant,
        with only minor temporal weighting.

        NEW: If concept_filter is provided, constrains retrieval to only documents
        matching those XBRL concepts (e.g., ['NetIncomeLoss', 'Revenues']).

        Pass 2 (Temporal Zoom Re-ranking):
        ----------------------------------
        Recompute scores for Pass 1 results using:
            score = semantic_sim × exp(-β × (Δφ)²)

        This applies temporal alignment based on phase difference,
        controlled by β (zoom factor).

        Args:
            query_text: Query string
            query_timestamp: Target timestamp for retrieval (start for arcs)
            query_start_timestamp: Arc start timestamp (preferred over query_timestamp)
            query_end_timestamp: Arc end timestamp
            beta: Zoom factor for temporal focus (default: self.default_beta)
            lambda_coarse: Spin weight for coarse recall (default: 0.1)
            top_k_coarse: Number of candidates from Pass 1 (default: 50)
            top_k_final: Number of final results to return (default: 10)
            end_timestamp: Optional end timestamp for arc queries (deprecated)
            concept_filter: Optional list of XBRL concepts to filter by

        Returns:
            List of RetrievalResult objects, sorted by combined score
        """
        print(
            f"🔍 SEARCH CALLED: query_start={query_start_timestamp}, query_end={query_end_timestamp}, query_ts={query_timestamp}"
        )

        if beta is None:
            beta = self.default_beta

        # Determine if this is an arc or point query
        # Prefer explicit arc parameters over legacy end_timestamp
        if query_start_timestamp and query_end_timestamp:
            # Arc query (new style)
            query_ts = query_start_timestamp
            end_ts = query_end_timestamp
            print(f"🔍 RETRIEVER: ARC QUERY (new): {query_ts} to {end_ts}")
            logger.info(f"🔍 ARC QUERY: {query_ts} to {end_ts}")
        elif query_timestamp and end_timestamp:
            # Arc query (legacy style)
            query_ts = query_timestamp
            end_ts = end_timestamp
            print(f"🔍 RETRIEVER: ARC QUERY (legacy): {query_ts} to {end_ts}")
            logger.info(f"🔍 ARC QUERY (legacy): {query_ts} to {end_ts}")
        else:
            # Point query
            query_ts = query_timestamp
            end_ts = None
            print(f"🔍 RETRIEVER: POINT QUERY: {query_ts}")
            logger.info(f"🔍 POINT QUERY: {query_ts}")

        # ====================================================================
        # PASS 1: COARSE RECALL (broad semantic search)
        # ====================================================================

        # Create query with small λ for broad search
        query = self.create_query(
            query_text=query_text,
            query_timestamp=query_ts,
            lambda_factor=lambda_coarse,
            end_timestamp=end_ts,
        )

        # Build metadata filter if concept_filter is provided
        filter_dict = None
        if concept_filter:
            # Filter for facts matching any of the target concepts
            # Check both 'concept' (local name) and 'concept_full' (namespace-prefixed)
            filter_dict = {
                "$and": [
                    {"chunk_type": "fact"},  # Only filter facts
                    {
                        "$or": [
                            {"concept": {"$in": concept_filter}},
                            {"concept_full": {"$in": concept_filter}},
                        ]
                    },
                ]
            }
            logger.info(
                f"🔍 CONCEPT FILTER: Constraining retrieval to {len(concept_filter)} concepts"
            )
            print(
                f"🔍 CONCEPT FILTER: {concept_filter[:5]}{'...' if len(concept_filter) > 5 else ''}"
            )

        # Retrieve top-K candidates from vector store
        candidates = self.vector_store.search(
            query_embedding=query.full_embedding,
            top_k=top_k_coarse,
            filter_dict=filter_dict,  # NEW: Pass metadata filter
        )
        logger.debug(
            f"🔍 PASS 1: Retrieved {len(candidates)} candidates from coarse search"
        )
        if not candidates:
            return []

        # ====================================================================
        # PASS 2: TEMPORAL ZOOM RE-RANKING (Arc-aware)
        # ====================================================================

        results = []
        for doc, coarse_score in candidates:
            # Compute semantic similarity (without spin component)
            semantic_score = cosine_similarity(
                query.semantic_embedding, doc.semantic_embedding
            )

            # ========================================================
            # MULTI-SCALE TEMPORAL ALIGNMENT
            # ========================================================
            # Compute alignment at each scale (quarter, decade, century)
            # and combine with weights
            scale_alignments = []
            scale_weights = [QUARTER_WEIGHT, DECADE_WEIGHT, CENTURY_WEIGHT]
            delta_phi_avg = 0.0

            # NEW: Hard boundary check for arc-to-arc queries
            # If arcs don't overlap at ANY scale, reject the document entirely
            # This ensures proper year-to-year separation (decade scale) AND
            # within-year position matching (quarter scale)
            logger.debug(f"doc.doc_id: {doc.doc_id}")
            if query.is_arc and doc.is_arc:
                logger.debug(
                    f"Decade: {query.phi_start['decade']} to {query.phi_end['decade']} vs {doc.phi_start['decade']} to {doc.phi_end['decade']}"
                )
                logger.debug(
                    f"Quarter: {query.phi_start['quarter']} to {query.phi_end['quarter']} vs {doc.phi_start['quarter']} to {doc.phi_end['quarter']}"
                )
                logger.debug(
                    f"Century: {query.phi_start['century']} to {query.phi_end['century']} vs {doc.phi_start['century']} to {doc.phi_end['century']}"
                )
            if query.is_arc and doc.is_arc:
                # Check overlap at ALL scales (quarter, decade, century)
                # If ANY scale has zero overlap, reject the document
                reject_doc = False

                # Check scales in order of hierarchy: century → decade → quarter
                # This ensures broad temporal boundaries are validated before finer-grained scales
                for scale_name in ["century", "decade", "quarter"]:
                    scale_overlap = arc_overlap(
                        query.phi_start[scale_name],
                        query.phi_end[scale_name],
                        doc.phi_start[scale_name],
                        doc.phi_end[scale_name],
                    )
                    logger.debug(
                        f"Scale: {scale_name}, Overlap: {scale_overlap}"
                    )
                    if scale_overlap == 0.0:
                        # HARD REJECT: Document arc doesn't intersect query arc at this scale
                        # This prevents cross-year bleeding (decade) and ensures temporal accuracy
                        logger.debug(
                            f"Rejecting doc (arc boundary at {scale_name} scale): "
                            f"query [{query.phi_start[scale_name]:.3f}, {query.phi_end[scale_name]:.3f}] "
                            f"vs doc [{doc.phi_start[scale_name]:.3f}, {doc.phi_end[scale_name]:.3f}] "
                            f"have zero overlap"
                        )
                        reject_doc = True
                        break  # No need to check other scales

                if reject_doc:
                    continue  # Skip to next candidate

            for scale_name in ["quarter", "decade", "century"]:
                query_phi = query.phi[scale_name]
                doc_phi = doc.phi[scale_name]

                # Compute alignment at this scale based on point/arc types
                if query.is_arc and doc.is_arc:
                    # Arc-to-arc: Use Jaccard similarity
                    scale_alignment = jaccard_similarity_arcs(
                        query.phi_start[scale_name],
                        query.phi_end[scale_name],
                        doc.phi_start[scale_name],
                        doc.phi_end[scale_name],
                    )
                    logger.debug(
                        f"🔍 Jaccard ({scale_name}): {scale_alignment:.3f}"
                    )
                elif query.is_arc and not doc.is_arc:
                    # Arc-to-point: Check if point falls within query arc
                    overlap = arc_overlap(
                        query.phi_start[scale_name],
                        query.phi_end[scale_name],
                        doc_phi,
                        doc_phi,
                    )
                    if overlap > 0:
                        scale_alignment = 1.0  # Point is within arc
                    else:
                        # Point outside arc: use distance to arc center
                        delta = angular_difference(query_phi, doc_phi)
                        scale_alignment = math.exp(-beta * (delta**2))
                elif not query.is_arc and doc.is_arc:
                    # Point-to-arc: Check if query point falls within doc arc
                    overlap = arc_overlap(
                        doc.phi_start[scale_name],
                        doc.phi_end[scale_name],
                        query_phi,
                        query_phi,
                    )
                    if overlap > 0:
                        scale_alignment = 1.0  # Query point is within arc
                    else:
                        # Query point outside arc: use distance to arc center
                        delta = angular_difference(query_phi, doc_phi)
                        scale_alignment = math.exp(-beta * (delta**2))
                else:
                    # Point-to-point: Use standard angular difference
                    delta = angular_difference(query_phi, doc_phi)
                    scale_alignment = math.exp(-beta * (delta**2))
                    delta_phi_avg += delta

                scale_alignments.append(scale_alignment)

            # Weighted combination of scale alignments
            # Decade scale gets highest weight (0.5) for year-to-year discrimination
            temporal_alignment = sum(
                w * a for w, a in zip(scale_weights, scale_alignments)
            ) / sum(scale_weights)

            # Average delta_phi for reporting (using decade scale as primary)
            delta_phi = angular_difference(
                query.phi["decade"], doc.phi["decade"]
            )

            # Combined score: semantic similarity weighted by temporal alignment
            combined_score = semantic_score * temporal_alignment

            # Create result object
            # For reporting, use decade-scale phi (highest weight for year discrimination)
            result = RetrievalResult(
                doc_id=doc.doc_id,
                text=doc.text,
                timestamp=doc.timestamp,
                semantic_score=semantic_score,
                phi_doc=doc.phi["decade"],  # Use decade scale for reporting
                phi_query=query.phi[
                    "decade"
                ],  # Use decade scale for reporting
                phi_difference=delta_phi,  # Already using decade scale
                temporal_alignment=temporal_alignment,  # Multi-scale weighted
                combined_score=combined_score,
                metadata=doc.metadata,
            )
            results.append(result)

        # ====================================================================
        # PASS 3: PRIORITY BOOST (based on chunk_type metadata)
        # ====================================================================
        # Apply multipliers to prioritize structured data over narrative
        PRIORITY_MULTIPLIERS = {
            "fact": 3.0,  # 3.0x - Structured XBRL facts (highest priority)
            "footnote": 2.0,  # 2.0x - Footnote disclosures
            "financial_table": 1.5,  # 1.5x - Financial statement tables
            "narrative": 1.0,  # 1.0x - General narrative (baseline)
            "legacy": 1.0,  # 1.0x - Documents without chunk_type (baseline)
        }

        for result in results:
            # Get chunk_type from document metadata
            chunk_type = "legacy"  # default
            doc_id_to_find = (
                result.doc_id if hasattr(result, "doc_id") else None
            )

            if doc_id_to_find:
                # Find the original document to get metadata (only works with InMemoryVectorStore)
                if hasattr(self.vector_store, "documents"):
                    for doc in self.vector_store.documents.values():
                        try:
                            if (
                                hasattr(doc, "doc_id")
                                and doc.doc_id == doc_id_to_find
                            ):
                                if (
                                    hasattr(doc, "metadata")
                                    and doc.metadata
                                    and isinstance(doc.metadata, dict)
                                ):
                                    chunk_type = doc.metadata.get(
                                        "chunk_type", "legacy"
                                    )
                                break
                        except (AttributeError, TypeError):
                            continue

            # Apply priority multiplier
            multiplier = PRIORITY_MULTIPLIERS.get(chunk_type, 1.0)
            result.combined_score *= multiplier

        # Sort by boosted combined score (descending)
        results.sort(key=lambda r: r.combined_score, reverse=True)

        # Assign ranks
        for i, result in enumerate(results[:top_k_final]):
            result.rank = i + 1

        return results[:top_k_final]

    def search_with_beta_sweep(
        self,
        query_text: str,
        query_timestamp: Optional[datetime] = None,
        query_start_timestamp: Optional[datetime] = None,
        query_end_timestamp: Optional[datetime] = None,
        beta_values: Optional[List[float]] = None,
        top_k: int = 10,
    ) -> List[Tuple[float, List[RetrievalResult]]]:
        """
        Perform retrieval with multiple β values to demonstrate temporal zoom.

        This method shows how adjusting β smoothly transitions from broad
        semantic search to temporally-focused retrieval.

        Args:
            query_text: Query string
            query_timestamp: Target timestamp
            beta_values: List of β values to try (default: [0, 1, 5, 10, 20])
            top_k: Number of results per β

        Returns:
            List of (beta, results) tuples
        """
        if beta_values is None:
            beta_values = [0, 1, 5, 10, 20]

        sweep_results = []
        for beta in beta_values:
            results = self.search(
                query_text=query_text,
                query_timestamp=query_timestamp,
                query_start_timestamp=query_start_timestamp,
                query_end_timestamp=query_end_timestamp,
                beta=beta,
                top_k_final=top_k,
            )
            sweep_results.append((beta, results))

        return sweep_results

    def explain_result(self, result: RetrievalResult) -> str:
        """
        Generate human-readable explanation of a retrieval result.

        Args:
            result: RetrievalResult to explain

        Returns:
            Formatted explanation string
        """
        lines = [
            f"Rank #{result.rank}",
            f"Document ID: {result.doc_id}",
            f"Timestamp: {result.timestamp.isoformat()}",
            f"",
            f"Scores:",
            f"  Semantic Similarity: {result.semantic_score:.4f}",
            f"  Temporal Alignment:  {result.temporal_alignment:.4f}",
            f"  Combined Score:      {result.combined_score:.4f}",
            f"",
            f"Phase Information:",
            f"  Document Phase (φ_doc):   {result.phi_doc:.4f} rad ({math.degrees(result.phi_doc):.1f}°)",
            f"  Query Phase (φ_query):    {result.phi_query:.4f} rad ({math.degrees(result.phi_query):.1f}°)",
            f"  Phase Difference (Δφ):    {result.phi_difference:.4f} rad ({math.degrees(result.phi_difference):.1f}°)",
            f"",
            f"Text Preview:",
            (
                f"  {result.text[:200]}..."
                if len(result.text) > 200
                else f"  {result.text}"
            ),
        ]
        return "\n".join(lines)


def format_results_table(
    results: List[RetrievalResult], max_text_length: int = 100
) -> str:
    """
    Format retrieval results as a table.

    Args:
        results: List of RetrievalResult objects
        max_text_length: Maximum text preview length

    Returns:
        Formatted table string
    """
    if not results:
        return "No results."

    # Header
    lines = [
        "┌────┬──────────┬─────────────┬───────────┬──────────┬"
        + "─" * (max_text_length + 2)
        + "┐",
        f"│ #  │ Semantic │ Temporal    │ Combined  │ Δφ (deg) │ {'Text Preview'.ljust(max_text_length)} │",
        "├────┼──────────┼─────────────┼───────────┼──────────┼"
        + "─" * (max_text_length + 2)
        + "┤",
    ]

    # Rows
    for result in results:
        rank = str(result.rank).rjust(2)
        semantic = f"{result.semantic_score:.4f}"
        temporal = f"{result.temporal_alignment:.4f}"
        combined = f"{result.combined_score:.4f}"
        delta_deg = f"{math.degrees(result.phi_difference):.1f}"

        # Truncate text
        text = result.text.replace("\n", " ")[:max_text_length]
        text = text.ljust(max_text_length)

        lines.append(
            f"│ {rank} │ {semantic} │ {temporal}   │ {combined} │ {delta_deg.rjust(8)} │ {text} │"
        )

    # Footer
    lines.append(
        "└────┴──────────┴─────────────┴───────────┴──────────┴"
        + "─" * (max_text_length + 2)
        + "┘"
    )

    return "\n".join(lines)
