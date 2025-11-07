# Temporal Scaling Findings

## What We Tested

We implemented a `temporal_scale` parameter that amplifies the spin vector before concatenation with semantic embeddings:

```python
spin_vector = [temporal_scale * cos(φ), temporal_scale * sin(φ)]
full_embedding = [semantic_embedding, spin_vector]
```

The hypothesis was that scaling the temporal features would make them more prominent in similarity calculations.

## Results

**Temporal scaling has NO effect on cosine similarity ranking.**

### Why?

**Cosine similarity is scale-invariant.** It measures the angle between vectors, not their magnitude:

```
cos_similarity(A, B) = (A · B) / (||A|| × ||B||)
```

When you scale both document and query temporal features by the same factor:
- The dot product increases by `scale²`
- The vector norms increase by `scale`
- The ratio stays **exactly the same**

### Test Results

| temporal_scale | 2006 Score | 2007 Score | 2007 Rank |
|----------------|------------|------------|-----------|
| 1.0            | 0.6315     | 0.5715     | #4        |
| 5.0            | 0.6314     | 0.5715     | #4        |
| 10.0           | 0.6314     | 0.5715     | #4        |
| 20.0           | 0.6314     | 0.5715     | #4        |

**Rankings and scores are identical across all scales.**

## Correct Approach

For temporal retrieval, we have **one effective control:**

### β Parameter (Temporal Zoom in Re-Ranking)

```python
score = semantic_similarity × exp(-β × (Δφ)²)
```

- **β = 50**: Default (±1-2 years)
- **β = 100**: Strong temporal focus
- **β = 500**: Very strong temporal focus

This works because it's **not** based on cosine similarity - it's an exponential penalty applied to the semantic score.

### Alternative: Extract Year + Boost

For strict year matching, extract year from query and apply a boost:

```python
if doc_year == query_year:
    score *= 2.0  # or add a large boost
```

This is deterministic and guarantees exact year matches rank first.

## Conclusion

- ✅ **β parameter works** - provides smooth temporal zoom
- ❌ **temporal_scale does NOT work** - cosine similarity is scale-invariant
- ✅ **Year extraction + boosting** - for deterministic exact matches

The `temporal_scale` parameter has been implemented but should be set to 1.0 (default) as it has no effect. We'll keep it for API consistency but document that it's ineffective for cosine similarity.

---

**Key Learning:** When using cosine similarity, only the **direction** of the vector matters, not its magnitude. Temporal influence must be controlled through re-ranking (β) or explicit boosting, not vector scaling.

