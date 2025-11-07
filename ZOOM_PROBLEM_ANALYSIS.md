# Temporal Zoom Problem Analysis

## The Issue

You're right - the "zoom" isn't working well enough. Here's why:

### Current Scoring Formula

```python
combined_score = semantic_score × exp(-β × (Δφ)²)
```

### Example: 2007 Query

| Year | Semantic | Δφ (°) | β=100 Penalty | Combined | Rank |
|------|----------|--------|---------------|----------|------|
| 2006 | 0.6327   | 0.36   | 0.9961        | 0.6302   | #1   |
| 2007 | 0.5715   | 0.00   | 1.0000        | 0.5715   | #3   |

**Problem:** 2006 has 10% higher semantic score. Even with perfect temporal alignment (1.0), 2007 can't win because:

```
0.5715 × 1.0 = 0.5715 < 0.6327 × 0.996 = 0.6302
```

The temporal penalty on 2006 is only 0.4% (from 0.9961), which isn't enough to overcome the 10% semantic advantage.

## Why This Happens

### Multiplicative Penalty is Too Weak

The exponential `exp(-β × (Δφ)²)` decays slowly for small angles:

| Δφ (°) | Δφ (rad) | β=100 Penalty | Penalty Loss |
|--------|----------|---------------|--------------|
| 0.36   | 0.0063   | 0.9961        | -0.39%       |
| 0.72   | 0.0126   | 0.9843        | -1.57%       |
| 1.44   | 0.0251   | 0.9388        | -6.12%       |

Even at β=100, being 1 year off (0.36°) only loses 0.39% of your score!

### The 2006 Document is Semantically Strong

The 2006 report probably discusses 2007 projections, guidance, or has similar financial language, making it genuinely more relevant to the query "IBM 2007 revenue."

## Solutions

### Option 1: Additive Temporal Boost (Recommended)

Instead of multiplying by a penalty, ADD a temporal bonus:

```python
combined_score = semantic_score + α × exp(-β × (Δφ)²)
```

Where `α` is a tunable boost weight (e.g., 0.5).

**Example:**
- 2006: 0.6327 + 0.5 × 0.9961 = **1.1308**
- 2007: 0.5715 + 0.5 × 1.0000 = **1.0715**

Still not enough! Need α = 2.0:
- 2006: 0.6327 + 2.0 × 0.9961 = **2.6249**
- 2007: 0.5715 + 2.0 × 1.0000 = **2.5715**

Still loses! The semantic gap is huge.

### Option 2: Exponential Temporal Boost

Give perfect matches an exponential advantage:

```python
temporal_boost = exp(γ × exp(-β × (Δφ)²))
combined_score = semantic_score × temporal_boost
```

### Option 3: Year Extraction + Hard Filter

Most practical for production:

1. Extract year from query: "2007"
2. Filter or boost exact year matches:

```python
if doc_year == query_year:
    combined_score *= 10.0  # or filter to only show exact year
```

### Option 4: Stronger Exponential (Higher β Range)

Current β=100 isn't strong enough. Try β=1000-10000:

| β    | Penalty for 0.36° | Penalty for 0.72° |
|------|-------------------|-------------------|
| 100  | 0.9961            | 0.9843            |
| 1000 | 0.9610            | 0.8465            |
| 5000 | 0.8121            | 0.4089            |

At β=5000:
- 2006: 0.6327 × 0.8121 = 0.5138
- 2007: 0.5715 × 1.0000 = 0.5715 ✅ WINS!

## Recommendation

**Use Option 3 (Year Extraction)** for production - it's deterministic and users expect exact year matches.

**Use Option 4 (Very High β)** to test if pure temporal zoom can work - try β=5000 or β=10000.

The current formula CAN work, but needs MUCH higher β values than we thought because:
1. Semantic differences between adjacent years are larger than expected
2. The exponential penalty is gentler than intuitive for small angles
3. We need ~10x penalty difference to overcome ~10% semantic gaps

**The zoom knob works, but the scale is wrong!** β=100 feels like it should be "strong," but it's actually quite gentle. β=1000-10000 is where real temporal filtering happens.

