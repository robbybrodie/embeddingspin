"""
Two-Pass Temporal-Phase Spin Retrieval
======================================

Pass 1 — coarse semantic recall
    Score every stored vector as ``0.9 * semantic + 0.1 * temporal`` and keep the
    top ``top_k_coarse`` (200 by default). The small temporal weight keeps semantic
    meaning as the primary criterion and stops chronological constraints from
    excluding relevant documents before they have been considered on merit.

Pass 2 — temporal re-ranking
    For each candidate, descend the hierarchy under a :class:`TraversalPlan`, which
    visits only the circles the query actually needs. At each visited circle:

    - **Hard gate.** No arc overlap at any traversed scale rejects the document
      outright. This is what stops a Q2 2024 report from answering a Q2 2023
      question: the two coincide exactly on the 1-year circle and separate only on
      the 16-year circle.
    - **Soft score.** A Jaccard overlap coefficient per scale, combined using the
      hierarchy's weights.

    The final score blends the two axes under the temporal-focus parameter β::

        score = (1 - beta) * semantic_similarity + beta * temporal_alignment

    β = 0 is pure semantic search; β = 1 is absolute temporal dominance; the default
    0.5 balances them.

Pass 3 — priority and deduplication
    Optional metadata-driven priority multipliers, then deduplication on
    ``group_id`` so a chunk split across period boundaries is returned once.
"""

from __future__ import annotations

import logging
import math
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from temporal_config import DEFAULT_HIERARCHY, TemporalHierarchy
from temporal_encoding import (
    ScaleMatch,
    TemporalInterval,
    TraversalPlan,
    encode_single,
    evaluate_scale,
    full_span_interval,
    traversal_plan,
)
from temporal_spin import (
    RetrievalResult,
    SpinDocument,
    SpinQuery,
    cosine_similarity,
    deduplicate_by_group,
)
from vector_store import VectorStore

logger = logging.getLogger(__name__)


# Pass-1 blend. Semantic meaning dominates so that the candidate pool is broad.
COARSE_SEMANTIC_WEIGHT = 0.9
COARSE_TEMPORAL_WEIGHT = 0.1

# Multipliers applied after scoring, keyed on the ``chunk_type`` metadata field.
# Structured facts outrank narrative prose when both are temporally valid.
DEFAULT_PRIORITY_MULTIPLIERS: Dict[str, float] = {
    "fact": 3.0,
    "footnote": 2.0,
    "financial_table": 1.5,
    "narrative": 1.0,
    "legacy": 1.0,
}


class TemporalSpinRetriever:
    """
    Two-pass retriever over a corpus of temporally-encoded semantic vectors.

    The retriever's hierarchy must match the one used at ingestion. Mismatched
    epochs or periods produce phases that are not comparable, so the constructor
    refuses hierarchies that are not compatible extensions of one another when the
    store reports its own.

    Args:
        embedding_client: Provides query embeddings. Must be the same model used at
            ingestion.
        vector_store: Backend holding the indexed documents.
        hierarchy: Scales and epoch. Defaults to 1/16/256 from 1900-01-01.
        default_beta: Temporal-focus parameter in ``[0, 1]``.
        lambda_coarse: Weight applied to the temporal block during pass 1.
        priority_multipliers: Optional overrides for the ``chunk_type`` boosts.
    """

    def __init__(
        self,
        embedding_client,
        vector_store: VectorStore,
        hierarchy: TemporalHierarchy = DEFAULT_HIERARCHY,
        default_beta: float = 0.5,
        lambda_coarse: float = 0.1,
        priority_multipliers: Optional[Dict[str, float]] = None,
    ) -> None:
        if not 0.0 <= default_beta <= 1.0:
            raise ValueError(f"default_beta must be in [0, 1], got {default_beta}")
        self.embedding_client = embedding_client
        self.vector_store = vector_store
        self.hierarchy = hierarchy
        self.default_beta = default_beta
        self.lambda_coarse = lambda_coarse
        self.priority_multipliers = (
            DEFAULT_PRIORITY_MULTIPLIERS
            if priority_multipliers is None
            else priority_multipliers
        )

        store_hierarchy = getattr(vector_store, "hierarchy", None)
        if store_hierarchy is not None and not hierarchy.is_compatible_with(store_hierarchy):
            raise ValueError(
                "retriever hierarchy is incompatible with the vector store's:\n"
                f"  retriever: {hierarchy.fingerprint()}\n"
                f"  store    : {store_hierarchy.fingerprint()}\n"
                "Re-index the corpus or construct the retriever with the store's hierarchy."
            )

    # ------------------------------------------------------------------
    # Query construction
    # ------------------------------------------------------------------

    def create_query(
        self,
        query_text: str,
        interval: Optional[TemporalInterval] = None,
        lambda_factor: Optional[float] = None,
    ) -> SpinQuery:
        """
        Embed a query and encode its temporal constraint.

        A query with no temporal constraint is encoded as a full-span arc, which
        saturates every circle and therefore reduces to pure semantic search — the
        lazy traversal plan will find nothing worth checking and fall back to the
        outermost scale alone.

        Note what it is *not* encoded as: an instant at "now". That would be a point
        query against the present moment, which rejects the entire corpus. "I did not
        say when" and "I mean right now" are different questions.
        """
        if interval is None:
            interval = full_span_interval(self.hierarchy)

        semantic_embedding = self.embedding_client.embed_single(query_text)
        encoding = encode_single(interval, self.hierarchy)
        plan = traversal_plan(encoding, self.hierarchy)

        logger.debug(
            "query %r interval=%s traverse=%s skipped=%s",
            query_text[:60],
            interval,
            plan.scale_names,
            [s[0] for s in plan.skipped],
        )

        return SpinQuery(
            query_text=query_text,
            semantic_embedding=semantic_embedding,
            encoding=encoding,
            lambda_factor=self.lambda_coarse if lambda_factor is None else lambda_factor,
            plan=plan,
        )

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score_candidate(
        self,
        query: SpinQuery,
        doc: SpinDocument,
        beta: float,
        plan: TraversalPlan,
    ) -> Optional[RetrievalResult]:
        """
        Apply the hard gate and compute the blended score for one candidate.

        Returns ``None`` when the document fails the overlap gate at any traversed
        scale. Scales are visited coarsest-first so the widest, cheapest rejection
        happens before any fine-grained work.
        """
        semantic_score = cosine_similarity(
            query.semantic_embedding, doc.semantic_embedding
        )

        matches: Dict[str, ScaleMatch] = {}
        for scale_name in plan:
            scale = self.hierarchy.scale(scale_name)
            try:
                q_tuple = query.encoding.tuple_for(scale_name)
                d_tuple = doc.encoding.tuple_for(scale_name)
            except KeyError:
                # The document predates this scale and was padded, or the query
                # never encoded it. Either way there is nothing to check here.
                continue

            match = evaluate_scale(q_tuple, d_tuple, scale)
            matches[scale_name] = match
            if not match.overlaps:
                logger.debug(
                    "reject %s at %s: query arc [%.4f,+%.4f] vs doc [%.4f,+%.4f]",
                    doc.doc_id,
                    scale_name,
                    q_tuple.phi_start,
                    q_tuple.z,
                    d_tuple.phi_start,
                    d_tuple.z,
                )
                return None

        if not matches:
            # Nothing was traversable: fall back to pure semantic scoring.
            return RetrievalResult(
                doc_id=doc.doc_id,
                group_id=doc.group_id,
                text=doc.text,
                interval=doc.interval,
                semantic_score=semantic_score,
                temporal_alignment=1.0,
                combined_score=semantic_score,
                traversed_scales=tuple(plan.scale_names),
                metadata=doc.metadata,
            )

        weights = self.hierarchy.normalized_weights(list(matches))
        temporal = sum(weights[name] * m.jaccard for name, m in matches.items())

        # Blend the two axes. beta shifts prioritisation from pure semantic search
        # at 0 to absolute temporal dominance at 1.
        combined = (1.0 - beta) * semantic_score + beta * temporal

        return RetrievalResult(
            doc_id=doc.doc_id,
            group_id=doc.group_id,
            text=doc.text,
            interval=doc.interval,
            semantic_score=semantic_score,
            temporal_alignment=temporal,
            combined_score=combined,
            scale_matches=matches,
            traversed_scales=tuple(plan.scale_names),
            metadata=doc.metadata,
        )

    def _apply_priority(self, result: RetrievalResult) -> None:
        chunk_type = (result.metadata or {}).get("chunk_type", "legacy")
        result.combined_score *= self.priority_multipliers.get(chunk_type, 1.0)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_text: str,
        interval: Optional[TemporalInterval] = None,
        beta: Optional[float] = None,
        top_k_coarse: int = 200,
        top_k_final: int = 10,
        concept_filter: Optional[List[str]] = None,
        deduplicate: bool = True,
    ) -> List[RetrievalResult]:
        """
        Execute the two-pass search.

        Args:
            query_text: Natural-language query.
            interval: Temporal constraint. ``None`` means no constraint.
            beta: Temporal focus in ``[0, 1]``. Defaults to the retriever's.
            top_k_coarse: Candidate pool size from pass 1.
            top_k_final: Results returned after re-ranking.
            concept_filter: Optional XBRL concept names to restrict fact chunks to.
            deduplicate: Collapse split representations on ``group_id``.

        Returns:
            Results sorted by descending combined score, ranked from 1.
        """
        beta = self.default_beta if beta is None else beta
        if not 0.0 <= beta <= 1.0:
            raise ValueError(f"beta must be in [0, 1], got {beta}")

        query = self.create_query(query_text, interval)
        plan = query.plan or traversal_plan(query.encoding, self.hierarchy)

        # -- Pass 1: coarse semantic recall -----------------------------
        filter_dict = _concept_filter_clause(concept_filter)
        candidates = self.vector_store.search(
            query_embedding=query.full_embedding,
            top_k=top_k_coarse,
            filter_dict=filter_dict,
        )
        if not candidates:
            return []

        # -- Pass 2: temporal re-ranking --------------------------------
        results: List[RetrievalResult] = []
        for doc, _coarse_score in candidates:
            result = self.score_candidate(query, doc, beta, plan)
            if result is not None:
                results.append(result)

        # -- Pass 3: priority, dedup, rank ------------------------------
        for result in results:
            self._apply_priority(result)

        results.sort(key=lambda r: r.combined_score, reverse=True)
        if deduplicate:
            results = deduplicate_by_group(results)

        top = results[:top_k_final]
        for index, result in enumerate(top):
            result.rank = index + 1

        logger.info(
            "query=%r beta=%.2f traversed=%s candidates=%d survived=%d returned=%d",
            query_text[:60],
            beta,
            plan.scale_names,
            len(candidates),
            len(results),
            len(top),
        )
        return top

    def search_many(
        self,
        subqueries: Sequence[Tuple[str, Optional[TemporalInterval]]],
        beta: Optional[float] = None,
        top_k_final: int = 10,
        max_workers: int = 8,
        **kwargs: Any,
    ) -> List[List[RetrievalResult]]:
        """
        Run several sub-queries concurrently, preserving input order.

        A decomposed natural-language query — "Q1 impact on full year for 2021, 2022,
        2023" becomes six sub-queries — issues its retrieval calls in parallel rather
        than serially, since each is independent.
        """
        def run(item: Tuple[str, Optional[TemporalInterval]]) -> List[RetrievalResult]:
            text, interval = item
            return self.search(text, interval=interval, beta=beta, top_k_final=top_k_final, **kwargs)

        if len(subqueries) == 1:
            return [run(subqueries[0])]

        with ThreadPoolExecutor(max_workers=min(max_workers, len(subqueries))) as pool:
            return list(pool.map(run, subqueries))

    def search_with_beta_sweep(
        self,
        query_text: str,
        interval: Optional[TemporalInterval] = None,
        beta_values: Optional[List[float]] = None,
        top_k: int = 10,
    ) -> List[Tuple[float, List[RetrievalResult]]]:
        """
        Repeat a search across several β values to show the temporal-focus sweep.

        Useful for demonstrating that β is a runtime knob, not a property baked into
        the index.
        """
        if beta_values is None:
            beta_values = [0.0, 0.25, 0.5, 0.75, 1.0]
        return [
            (beta, self.search(query_text, interval=interval, beta=beta, top_k_final=top_k))
            for beta in beta_values
        ]

    def explain_result(self, result: RetrievalResult) -> str:
        """Human-readable breakdown of why a result scored as it did."""
        return result.explain()


def _concept_filter_clause(
    concepts: Optional[List[str]],
) -> Optional[Dict[str, Any]]:
    """Build a metadata filter restricting fact chunks to the given XBRL concepts."""
    if not concepts:
        return None
    return {
        "$and": [
            {"chunk_type": "fact"},
            {
                "$or": [
                    {"concept": {"$in": concepts}},
                    {"concept_full": {"$in": concepts}},
                ]
            },
        ]
    }


# ============================================================================
# Formatting
# ============================================================================


def format_results_table(
    results: List[RetrievalResult], max_text_length: int = 44
) -> str:
    """Render results as a fixed-width table."""
    if not results:
        return "No results."

    header = (
        f"│ {'#':>2} │ {'Period':<21} │ {'Semantic':>8} │ {'Temporal':>8} │ "
        f"{'Combined':>8} │ {'Text':<{max_text_length}} │"
    )
    rule = (
        "─" * 4 + "┼" + "─" * 23 + "┼" + "─" * 10 + "┼" + "─" * 10 + "┼"
        + "─" * 10 + "┼" + "─" * (max_text_length + 2)
    )
    lines = ["┌" + rule.replace("┼", "┬") + "┐", header, "├" + rule + "┤"]

    for r in results:
        if r.interval.end:
            period = f"{r.interval.start.date()}→{r.interval.end.date()}"
        else:
            period = f"{r.interval.start.date()} (point)"
        text = r.text.replace("\n", " ")[:max_text_length].ljust(max_text_length)
        lines.append(
            f"│ {r.rank:>2} │ {period:<21} │ {r.semantic_score:>8.4f} │ "
            f"{r.temporal_alignment:>8.4f} │ {r.combined_score:>8.4f} │ {text} │"
        )

    lines.append("└" + rule.replace("┼", "┴") + "┘")
    return "\n".join(lines)
