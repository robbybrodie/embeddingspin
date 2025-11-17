# Temporal-Phase Spin Retrieval System (Three Circles Method)

**A novel retrieval algorithm that encodes time using three concurrent angular circles (multi-scale hierarchical encoding), enabling precise temporal matching without model retraining.**

---

## 📌 What Problem Does This Solve?

Traditional vector databases fail at **time-series retrieval** for semantically similar documents from different time periods:

- **The Problem**: "IBM Q2 2023 revenue" and "IBM Q2 2024 revenue" have nearly identical semantic embeddings, making it hard to retrieve the correct temporal document.
- **Traditional Solutions**: Metadata filters (brittle), timestamp features (discrete), or fine-tuning (expensive).
- **Our Solution**: Encode time as a **continuous geometric phase** on the unit circle, enabling:
  - 🎯 **Smooth temporal zoom** (from broad to exact via β parameter)
  - 🔄 **Hierarchical time matching** (points within arcs, arc-to-arc overlap)
  - 🚫 **No model retraining** (post-hoc geometric augmentation)
  - ⚡ **Efficient retrieval** (two-pass search with arc-aware scoring)

**Combined with time-aware chunking strategies, this provides a simple and elegant fix for temporal retrieval in vector databases.**

## 🎯 Core Concept: Three Circles Method

Traditional retrieval systems treat time as a scalar feature or discrete bucket. This system represents time using **three concurrent circles** (multi-scale temporal encoding):

```
Three hierarchical scales (powers of 2):
  - Quarter scale:  1 year period   → φ_q (quarterly precision)
  - Decade scale:   16 year period  → φ_d (year-to-year discrimination)
  - Century scale:  256 year period → φ_c (historical context)

For each scale:
  φ = 2π × ((t - t₀) / period) mod 1.0

Multi-scale spin vector (9D):
  spin = [cos(φ_q), sin(φ_q), z_q,    # Quarter scale
          cos(φ_d), sin(φ_d), z_d,    # Decade scale
          cos(φ_c), sin(φ_c), z_c]    # Century scale

  where z = 0 for points, z = arc_length for time periods

Full embedding:
  full_embedding = [semantic_embedding, spin_vector]
  dimension = semantic_dim + 9
```

**Why three circles?**
- **Quarter scale**: Distinguishes Q1 from Q2 within the same year
- **Decade scale**: Primary discriminator for year-to-year separation (highest weight: 0.5)
- **Century scale**: Provides long-term historical context

This creates a **hierarchical temporal fingerprint** where documents are encoded at multiple resolutions simultaneously.

### Key Innovation: No Model Retraining Required

The semantic embedding model is **frozen**. Time encoding happens post-hoc in the vector space via geometric augmentation, making this approach:

- ✅ Model-agnostic (works with any embedding model)
- ✅ Efficient (no retraining overhead)
- ✅ Interpretable (phase angles have clear geometric meaning)
- ✅ Controllable (β parameter adjusts temporal focus at runtime)

### 🆕 Arc-Based Temporal Encoding (Time Periods)

**NEW**: The system now supports **both point and arc encoding** at each of the three scales:

#### Point Mode
- Single timestamp → 9D spin vector with z=0 at all scales
- `[cos(φ_q), sin(φ_q), 0, cos(φ_d), sin(φ_d), 0, cos(φ_c), sin(φ_c), 0]`
- For point-in-time events (news articles, tweets, instant messages)
- Each scale encodes the point's position on its respective circle

#### Arc Mode
- Time interval `[t_start, t_end]` → 9D spin vector with z=arc_length at each scale
- `[cos(φ_q_center), sin(φ_q_center), arc_q, cos(φ_d_center), sin(φ_d_center), arc_d, cos(φ_c_center), sin(φ_c_center), arc_c]`
- For time periods (quarterly reports, annual reviews, multi-day events)
- Each scale's arc_length encodes how much of that circle the period spans

**Visual Example (Three Concurrent Circles):**

```
QUARTER SCALE (1 year):        DECADE SCALE (16 years):      CENTURY SCALE (256 years):
Q2 2023 report:                2023 annual report:           2010-2025 historical period:
      ───                            •                              •
   (90° arc)                    (≈22.5° point)                  (≈5° point)

Multi-scale overlap detection:
- Q2 report in annual: Quarter=100%, Decade=100%, Century=100% → Strong match
- Q2 2023 vs Q2 2024: Quarter=100%, Decade=0%, Century=0% → Rejected (no decade overlap)
- 2023 vs 2024 docs: Quarter=varies, Decade=0%, Century=0% → Rejected (hard boundary)
```

**Multi-Scale Arc-Aware Retrieval:**

| Query Type | Document Type | Matching Logic |
|------------|---------------|----------------|
| Point → Point | Point → Point | Multi-scale angular distance (weighted combination) |
| Point → Arc | Query falls within doc period? | 1.0 if inside at all scales, else distance to center |
| Arc → Point | Doc falls within query period? | 1.0 if inside at all scales, else distance to center |
| Arc → Arc | Temporal overlap | **Hard boundary check**: Reject if zero overlap at ANY scale, else weighted Jaccard |

**Key Innovation - Hard Boundary Enforcement:**
- Arc-to-arc queries check overlap at **all three scales**
- If ANY scale shows zero overlap → document is **rejected** (not just down-weighted)
- This prevents temporal bleeding (e.g., 2023 docs contaminating 2024 queries)
- Decade scale enforces year-to-year separation
- Quarter scale enforces within-year position matching
- Century scale provides sanity check for historical boundaries

**Use Cases:**
- **Financial reporting hierarchy**: 10-Q (quarterly) ⊂ 10-K (annual) detected by quarter-scale arc containment
- **Event periods**: "Q2 2023 performance" retrieves docs from Apr-Jun 2023, hard-rejects 2024 docs
- **Time-series chunking**: Each chunk knows its temporal extent at multiple resolutions
- **Periodic data**: Automatically handles wrapping (e.g., fiscal years) at appropriate scale

## 🔬 How It Works

### Ingestion Pipeline

1. **Timestamp Extraction**: Parse timestamps from document text using regex patterns and dateutil
   - Recognizes formats like "for the period ended 31 December 2019"
   - Falls back to file metadata or ingestion time

2. **Semantic Embedding**: Obtain text embedding from LlamaStack Model Gateway
   - Uses registered embedding models (e.g., `text-embedding-v1`)
   - No special temporal training needed

3. **Multi-Scale Spin Encoding**: Convert timestamp(s) to 9D spin vector (3 scales × 3D each)
   
   **Point mode (single timestamp):**
   ```python
   # Encode at each of the three scales
   for scale in [quarter_period, decade_period, century_period]:
       fraction = ((timestamp - t₀) / scale) % 1.0
       φ = 2π × fraction
       spin.extend([cos(φ), sin(φ), 0.0])  # z=0 for points
   
   # Result: 9D vector
   spin = [cos(φ_q), sin(φ_q), 0,    # Quarter scale
           cos(φ_d), sin(φ_d), 0,    # Decade scale
           cos(φ_c), sin(φ_c), 0]    # Century scale
   ```
   
   **Arc mode (time interval):**
   ```python
   # Encode at each of the three scales
   for scale in [quarter_period, decade_period, century_period]:
       φ_start = 2π × ((t_start - t₀) / scale) % 1.0
       φ_end = 2π × ((t_end - t₀) / scale) % 1.0
       φ_center = (φ_start + φ_end) / 2
       arc_length = φ_end - φ_start
       spin.extend([cos(φ_center), sin(φ_center), arc_length])
   
   # Result: 9D vector with arc lengths
   spin = [cos(φ_q_c), sin(φ_q_c), arc_q,    # Quarter scale arc
           cos(φ_d_c), sin(φ_d_c), arc_d,    # Decade scale arc
           cos(φ_c_c), sin(φ_c_c), arc_c]    # Century scale arc
   ```

4. **Concatenation**: Combine semantic + multi-scale spin into full embedding
   ```python
   # Both points and arcs use 9D spin vectors for consistent dimensionality
   full_embedding = [semantic_embedding..., spin_vector...]
   # semantic_dim + 9 total dimensions
   # Points have all z=0, arcs have z=arc_length at each scale
   ```

5. **Storage**: Index in vector database (PGVector, Chroma, or in-memory)

### Retrieval Algorithm: Two-Pass Temporal Zoom

#### Pass 1: Coarse Recall (Broad Semantic Search)

```python
query_full = [query_semantic, λ × query_spin]  # Small λ ≈ 0.1
candidates = vector_db.search(query_full, top_k=50)
```

Uses small λ to perform broad semantic search with minor temporal weighting.

#### Pass 2: Multi-Scale Temporal Zoom Re-ranking (Arc-Aware)

```python
for doc in candidates:
    # HARD BOUNDARY CHECK for arc-to-arc queries
    if query.is_arc and doc.is_arc:
        # Check overlap at ALL three scales
        for scale in ['quarter', 'decade', 'century']:
            if arc_overlap(query.arc[scale], doc.arc[scale]) == 0:
                reject_document()  # Zero overlap at any scale = hard reject
                break
    
    # Compute alignment at each scale
    scale_alignments = []
    for scale in ['quarter', 'decade', 'century']:
        if query.is_arc and doc.is_arc:
            # Arc-to-arc: Jaccard similarity
            alignment = jaccard_similarity(query.arc[scale], doc.arc[scale])
        elif point-in-arc or arc-contains-point:
            alignment = 1.0 if overlap > 0 else exp(-β × (Δφ_center)²)
        else:
            # Point-to-point: Angular distance
            Δφ = angular_difference(φ_query[scale], φ_doc[scale])
            alignment = exp(-β × (Δφ)²)
        scale_alignments.append(alignment)
    
    # Weighted combination (decade scale has highest weight: 0.5)
    temporal_alignment = (0.4 × align_q + 0.5 × align_d + 0.1 × align_c)
    score = semantic_similarity × temporal_alignment
```

**Multi-Scale β (zoom factor)** controls temporal focus at each scale:

- **β = 0**: Pure semantic search (time ignored at all scales)
- **β = 0.3**: Light temporal preference (good for 100-year periods)
- **β = 0.5**: Balanced temporal-semantic weighting **[DEFAULT for multi-scale]**
- **β = 0.7**: Strong temporal focus
- **β = 1.0**: Temporal alignment dominates

**Key differences from single-scale:**
- **Lower β values** (0.3-0.7) work well due to three concurrent signals
- **Decade scale** (weight=0.5) provides primary year-to-year discrimination
- **Quarter scale** (weight=0.4) handles within-year positioning
- **Century scale** (weight=0.1) prevents century-crossing errors
- **Hard boundaries** at arc-to-arc queries prevent temporal bleeding

The temporal alignment factor at each scale:
- Equals 1.0 when phases align perfectly (Δφ = 0)
- Decays smoothly as phases diverge
- Combined across scales with weights [0.4, 0.5, 0.1]

## 🚀 Quick Start

### Installation

```bash
# Clone or create project directory
cd embeddingspin

# Install dependencies
pip install -r requirements.txt
```

### Run Demo (Mock Embeddings)

The demo uses mock embeddings for fast, standalone testing:

```bash
# Full interactive demo
python demo.py

# Custom query with specific β
python demo.py --query "IBM cloud strategy" --timestamp 2019-06-30 --beta 10.0

# Show β parameter sweep
python demo.py --beta-sweep
```

### Run API Server

```bash
# Start FastAPI server with mock embeddings
python api.py

# Visit interactive docs
open http://localhost:8080/docs
```

**Example API Request:**

```bash
curl -X POST "http://localhost:8080/temporal_search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "IBM revenue 2016",
    "query_timestamp": "2016-06-30T00:00:00Z",
    "beta": 0.5,
    "top_k": 10
  }'
```

**Note:** Multi-scale encoding uses lower β values (0.3-0.7 typical) compared to single-scale (5000+).

### Arc Encoding Usage Example

**Ingesting time periods (quarterly/annual reports):**

```python
from datetime import datetime
from ingestion import TemporalSpinIngestionPipeline
from openai_client import OpenAIEmbeddingClient
from vector_store import InMemoryVectorStore

# Initialize
client = OpenAIEmbeddingClient(api_key="your-key")
store = InMemoryVectorStore()
pipeline = TemporalSpinIngestionPipeline(client, store)

# Ingest annual report (arc mode)
pipeline.ingest_document(
    text="IBM's fiscal year 2023 saw record AI growth...",
    timestamp=datetime(2023, 1, 1),
    end_timestamp=datetime(2023, 12, 31),  # Full year arc
    metadata={"type": "10-K", "year": 2023}
)

# Ingest quarterly report (arc mode)
pipeline.ingest_document(
    text="Q2 2023 performance exceeded expectations...",
    timestamp=datetime(2023, 4, 1),
    end_timestamp=datetime(2023, 6, 30),  # Q2 arc
    metadata={"type": "10-Q", "quarter": "Q2", "year": 2023}
)

# Ingest point-in-time event (point mode)
pipeline.ingest_document(
    text="IBM announced major acquisition on March 15, 2023...",
    timestamp=datetime(2023, 3, 15),
    # No end_timestamp = point mode
    metadata={"type": "news"}
)
```

**Querying with multi-scale arc matching:**

```python
from retrieval import TemporalSpinRetriever

retriever = TemporalSpinRetriever(client, store, default_beta=0.5)

# Query for a specific quarter (arc query)
results = retriever.search(
    query_text="Q2 2023 revenue growth",
    query_timestamp=datetime(2023, 4, 1),
    end_timestamp=datetime(2023, 6, 30),  # Arc query
    beta=0.5  # Multi-scale: lower β works well
)
# Returns: Both Q2 report (exact match) and annual report (contains Q2)
# Hard-rejects: Q2 2024 (zero overlap at decade scale)

# Query for a point in time
results = retriever.search(
    query_text="March 2023 acquisition",
    query_timestamp=datetime(2023, 3, 15),  # Point query
    beta=0.7  # Stronger temporal focus
)
# Returns: News article (exact), Q1 report (contains March), annual (contains March)
# Decade scale ensures only 2023 documents match
```

### Production Setup (LlamaStack + PGVector)

```bash
# Set environment variables
export USE_MOCK_EMBEDDINGS=false
export LLAMASTACK_URL=http://localhost:8000
export EMBEDDING_MODEL=text-embedding-v1
export VECTOR_STORE=pgvector
export DATABASE_URL=postgresql://user:pass@localhost:5432/vectordb

# Run API server
python api.py
```

## 📊 Demo Dataset

Includes 10 mock IBM financial reports (2015-2024) with:
- Realistic revenue and profit figures
- Strategic initiatives per year (Watson AI, Red Hat, hybrid cloud, quantum)
- Natural language suitable for semantic search
- Explicit date markers for timestamp extraction

**Example Query Demonstrations (Multi-Scale):**

| Query | Timestamp | β | Expected Behavior |
|-------|-----------|---|-------------------|
| "IBM revenue" | 2016-06-30 | 0.5 | Prioritizes 2016 report (balanced) |
| "IBM cloud strategy" | 2019-12-31 | 0.7 | Strong focus on 2019-2020 era |
| "IBM quantum computing" | 2024-06-30 | 0.3 | Light temporal preference, broad search |

**Note:** Lower β values (0.3-0.7) work well with multi-scale encoding due to hierarchical signal amplification.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     User Query + Timestamp                       │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│              LlamaStack Embedding Client                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Model Gateway API → text-embedding-v1                   │   │
│  │  Returns: semantic_embedding (e.g., 384-dim)            │   │
│  └─────────────────────────────────────────────────────────┘   │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│              Multi-Scale Temporal Spin Encoder                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  For each scale s ∈ {quarter, decade, century}:        │   │
│  │    φ_s = 2π × ((timestamp - t₀) / period_s) mod 1.0    │   │
│  │    spin_s = [cos(φ_s), sin(φ_s), z_s]                  │   │
│  │  spin_vector = concat(spin_q, spin_d, spin_c)  # 9D    │   │
│  │  query_full = [semantic, λ × spin_vector]              │   │
│  └─────────────────────────────────────────────────────────┘   │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│              PASS 1: Coarse Recall (λ = 0.1)                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Vector DB Search (cosine similarity)                    │   │
│  │  Retrieve top-K candidates (broad semantic search)       │   │
│  └─────────────────────────────────────────────────────────┘   │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│         PASS 2: Multi-Scale Temporal Zoom Re-ranking             │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  For each candidate:                                     │   │
│  │    // Hard boundary check for arc-to-arc                │   │
│  │    if query.is_arc and doc.is_arc:                      │   │
│  │      reject if zero overlap at any scale                │   │
│  │    // Compute alignment at each scale                    │   │
│  │    for scale in [quarter, decade, century]:             │   │
│  │      Δφ_s = angular_difference(φ_query[s], φ_doc[s])    │   │
│  │      align_s = exp(-β × (Δφ_s)²) or jaccard(arcs)       │   │
│  │    // Weighted combination (decade=0.5 highest)          │   │
│  │    alignment = Σ(w_s × align_s)                         │   │
│  │    score = semantic_sim × alignment                      │   │
│  │  Sort by score, return top-k                             │   │
│  └─────────────────────────────────────────────────────────┘   │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
                          Ranked Results
```

## 📁 Project Structure

```
embeddingspin/
├── temporal_spin.py        # Core: spin encoding, timestamp extraction
├── llamastack_client.py    # LlamaStack API wrapper + mock client
├── vector_store.py         # Vector DB abstraction (Memory/Chroma/PGVector)
├── ingestion.py            # Document ingestion pipeline
├── retrieval.py            # Two-pass retrieval algorithm
├── demo_data.py            # Mock IBM reports generator
├── demo.py                 # CLI demo script
├── api.py                  # FastAPI REST API
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_MOCK_EMBEDDINGS` | `true` | Use mock embeddings for testing |
| `LLAMASTACK_URL` | `http://localhost:8000` | LlamaStack API base URL |
| `LLAMASTACK_API_KEY` | - | Optional API key |
| `EMBEDDING_MODEL` | `text-embedding-v1` | Embedding model name |
| `VECTOR_STORE` | `memory` | Vector store type: `memory`, `chroma`, `pgvector` |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | Chroma persistence directory |
| `DATABASE_URL` | - | PostgreSQL connection string (for pgvector) |
| `LOAD_DEMO_DATA` | `true` | Auto-load IBM demo reports on startup |
| `PORT` | `8080` | API server port |
| `HOST` | `0.0.0.0` | API server host |

### Multi-Scale Temporal Encoding Parameters

```python
T0_EPOCH = datetime(2010, 1, 1)      # Base epoch

# Three hierarchical periods (powers of 2)
QUARTER_SCALE_YEARS = 1              # 1 year period (quarterly precision)
DECADE_SCALE_YEARS = 16              # 16 year period (year-to-year discrimination)
CENTURY_SCALE_YEARS = 256            # 256 year period (historical context)

# Scale weights for temporal alignment
QUARTER_WEIGHT = 0.4                 # Within-year precision
DECADE_WEIGHT = 0.5                  # Year discrimination (highest)
CENTURY_WEIGHT = 0.1                 # Historical context

# Default β for multi-scale encoding
DEFAULT_BETA = 0.5                   # Balanced temporal-semantic weighting
```

**Design rationale:**
- Powers of 2 maintain mathematical consistency across scales
- Decade scale gets highest weight (0.5) for primary year-to-year discrimination
- Quarter scale provides granular within-year positioning
- Century scale prevents long-term temporal errors

## 🎓 Use Cases

### 1. Financial Report Search
Query: "Q4 earnings 2019"  
→ Retrieves reports from Q4 2019, with β controlling temporal window

### 2. Legal Document Retrieval
Query: "GDPR compliance for the period ended 2020"  
→ Finds documents from 2020 compliance period

### 3. News Archive Search
Query: "COVID-19 vaccine development December 2020"  
→ Focuses on December 2020 news articles

### 4. Medical Records
Query: "patient symptoms January 2023"  
→ Retrieves records from January 2023 visit

### 5. Code Repository Search
Query: "authentication bug fix"  
Timestamp: Last month  
→ Prioritizes recent commits

## 🔬 Advanced Features

### Beta Sweep API (Multi-Scale)

Compare results across multiple β values:

```python
POST /beta_sweep
{
  "query": "IBM AI strategy",
  "query_timestamp": "2019-06-30T00:00:00Z",
  "beta_values": [0, 0.3, 0.5, 0.7, 1.0],
  "top_k": 5
}
```

Returns results for each β, showing smooth transition from semantic to temporal focus.

**Multi-scale β interpretation:**
- **0**: Pure semantic (all scales ignored)
- **0.3**: Light temporal preference (year matters, quarters less)
- **0.5**: Balanced (default, good year discrimination)
- **0.7**: Strong temporal focus (tight year matching)
- **1.0**: Temporal dominates (very precise)

### Custom Timestamp Extraction

Add custom regex patterns for domain-specific date formats:

```python
from temporal_spin import DATE_PATTERNS

# Add custom pattern
DATE_PATTERNS.append(r'report\s+date:\s+(\d{4}-\d{2}-\d{2})')
```

### Multiple Embedding Models

Switch models without changing spin encoding:

```python
# Use different model
client = LlamaStackEmbeddingClient(
    model_name="nomic-embed-text-v1.5"
)
```

Spin encoding works with any embedding model!

## 📈 Performance

### Ingestion
- **Single document**: ~50-100ms (embedding + spin encoding + DB insert)
- **Batch (100 docs)**: ~2-5s (batched embeddings amortize overhead)

### Retrieval
- **Pass 1 (coarse recall)**: ~10-50ms (vector DB search)
- **Pass 2 (re-ranking)**: ~1-5ms (in-memory computation)
- **Total**: ~15-55ms for typical queries

### Scalability
- **In-Memory**: < 10k documents
- **Chroma**: < 1M documents
- **PGVector**: 10M+ documents (with proper indexing)

## 🧪 Testing

```bash
# Run demo with mock data (uses multi-scale encoding)
python demo.py

# Test specific query (note: lower β for multi-scale)
python demo.py --query "test query" --timestamp 2020-01-01 --beta 0.5

# Show β sweep (demonstrates multi-scale β range)
python demo.py --beta-sweep

# Test arc encoding demo
python arc_demo.py

# Test API endpoints
pytest tests/  # (if you add tests/)
```

## 🤝 Integration with Red Hat AI 3 (LlamaStack)

This system is designed for Red Hat AI 3 environments:

1. **Model Gateway**: Automatically discovers registered embedding models
2. **Vector Store**: Works with PGVector (often bundled with LlamaStack)
3. **API**: FastAPI server integrates with existing services
4. **Scalability**: Horizontal scaling via stateless API design

**Deployment:**

```bash
# In your LlamaStack environment
pip install -r requirements.txt

# Configure
export USE_MOCK_EMBEDDINGS=false
export LLAMASTACK_URL=$MODEL_GATEWAY_URL
export VECTOR_STORE=pgvector
export DATABASE_URL=$POSTGRES_CONNECTION_STRING

# Run
python api.py
```

## 📚 References & Theory

### Why Multi-Scale Spin Encoding (Three Circles)?

**Three concurrent circular representations** of time provide hierarchical advantages:

1. **Periodicity at multiple scales**: Natural for hierarchical patterns (quarters → years → decades)
2. **Continuity**: Smooth interpolation between timestamps at each scale
3. **Bounded**: Always 9D (3 scales × 3D), regardless of time range
4. **Interpretable**: Angular differences have clear geometric meaning at each scale
5. **Hierarchical matching**: Documents can match at fine (quarter) or coarse (decade) granularity
6. **Hard boundaries**: Arc-to-arc queries can enforce strict temporal separation

### Mathematical Foundation

The **multi-scale temporal alignment** combines three Gaussian-like kernels:

```
At each scale s ∈ {quarter, decade, century}:
  alignment_s(Δφ_s; β) = exp(-β × (Δφ_s)²)

Combined alignment:
  alignment = w_q × align_quarter + w_d × align_decade + w_c × align_century
  where w_q=0.4, w_d=0.5, w_c=0.1 (sum to 1.0)
```

**Properties:**
- Maximum = 1 when all scales align perfectly (Δφ = 0 at all scales)
- Decays at different rates per scale (faster for shorter periods)
- **Quarter scale** (1 year): Δφ changes by 2π per year → rapid decay for cross-year queries
- **Decade scale** (16 years): Δφ changes by 2π/16 ≈ 0.39 rad/year → primary year discrimination
- **Century scale** (256 years): Δφ changes by 2π/256 ≈ 0.025 rad/year → historical context

**With β = 0.5 (default):**
- Quarter scale: exp(-0.5 × (2π)²) ≈ 1.4e-9 for 1 year apart (strong rejection)
- Decade scale: exp(-0.5 × (0.39)²) ≈ 0.93 for 1 year apart (gentle penalty)
- Century scale: exp(-0.5 × (0.025)²) ≈ 0.9997 for 1 year apart (negligible)

This **hierarchical decay** enables precise year matching (decade scale) while tolerating
within-year variations (quarter scale) and maintaining historical sanity checks (century scale).

### Comparison to Alternatives

| Approach | Pros | Cons |
|----------|------|------|
| **Scalar timestamp** | Simple | Doesn't capture periodicity |
| **Discrete buckets** | Interpretable | Hard boundaries, no interpolation |
| **Learned temporal embeddings** | Flexible | Requires retraining, less interpretable |
| **Single-scale spin encoding** | No retraining, interpretable, periodic | Temporal collisions for large ranges |
| **Multi-scale spin (ours)** | Hierarchical resolution, no collisions, hard boundaries | More complex (9D vs 2D) |

### Evolution: Single-Scale → Multi-Scale

**Original approach (deprecated):**
- Single circle with configurable period (e.g., 1000 years)
- 2D spin vector: `[cos(φ), sin(φ)]`
- Problem: β values needed to be very high (5000+) to overcome semantic similarity

**Current approach (three circles):**
- Three concurrent circles with periods: 1 year, 16 years, 256 years
- 9D spin vector: 3 scales × 3D each
- Advantage: Lower β values (0.3-0.7) work well due to multi-scale signal
- Each scale operates at its natural frequency for its purpose

## 🐛 Troubleshooting

### "Failed to get embeddings from LlamaStack"

- Check `LLAMASTACK_URL` is correct
- Verify embedding model is registered: `curl $LLAMASTACK_URL/v1/models`
- Try with `USE_MOCK_EMBEDDINGS=true` to isolate issue

### "ImportError: No module named 'chromadb'"

```bash
pip install chromadb
```

### "No documents found"

- Ensure demo data is loaded: `LOAD_DEMO_DATA=true`
- Or manually ingest: `POST /ingest`

### Results don't vary with β

- Check timestamps are properly parsed (not all defaulting to same time)
- Verify β is being passed correctly in API request
- Try larger β values (10-20) for sharper focus

## 📄 License

MIT License - See LICENSE file

## 👤 Author

Robby Brodie  
For questions or collaboration: robbytherobot@redhat.com

## 🙏 Acknowledgments

- Red Hat AI 3 (LlamaStack) team for Model Gateway API
- PGVector and Chroma DB for vector search capabilities
- Community contributors to dateutil, FastAPI, and related libraries
- **Bryon Baker** (brbaker@redhat.com) for collaborative development of the arc-based temporal encoding extension, solving hierarchical time-period matching for financial reporting (10-Q/10-K) and time-series chunking strategies

---

**Ready to revolutionize temporal retrieval?** 🚀

Start with: `python demo.py`

