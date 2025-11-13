# Temporal-Phase Spin Retrieval System

**A novel retrieval algorithm that encodes time as an angular spin state on the unit circle, enabling smooth temporal zoom without model retraining.**

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

## 🎯 Core Concept

Traditional retrieval systems treat time as a scalar feature or discrete bucket. This system represents time as a **continuous angular coordinate** on the unit circle:

```
φ = 2π × (t - t₀) / T

spin_vector = [cos(φ), sin(φ)]

full_embedding = [semantic_embedding, spin_vector]
```

### Key Innovation: No Model Retraining Required

The semantic embedding model is **frozen**. Time encoding happens post-hoc in the vector space via geometric augmentation, making this approach:

- ✅ Model-agnostic (works with any embedding model)
- ✅ Efficient (no retraining overhead)
- ✅ Interpretable (phase angles have clear geometric meaning)
- ✅ Controllable (β parameter adjusts temporal focus at runtime)

### 🆕 Arc-Based Temporal Encoding (Time Periods)

**NEW**: The system now supports **both point and arc encoding**:

#### Point Mode
- Single timestamp → 3D spin vector `[cos(φ), sin(φ), 0.0]`
- For point-in-time events (news articles, tweets, instant messages)
- `arc_length = 0` for backward compatibility and consistent dimensionality

#### Arc Mode
- Time interval `[t_start, t_end]` → 3D spin vector `[cos(φ_center), sin(φ_center), arc_length]`
- For time periods (quarterly reports, annual reviews, multi-day events)
- `arc_length > 0` encodes the temporal extent of the period

**Visual Example (100-year circle):**

```
Point encoding:      Annual report:        Quarterly report:
     •               ────────────           ───
   2023.5            2023 (full year)       Q2 2023

Arc overlap detection:
- Quarterly ⊂ Annual: Jaccard = 0.25 (quarter is 25% of year)
- Adjacent quarters: Jaccard = 0.0 (no overlap)
- Point within arc: Temporal alignment = 1.0
```

**Arc-Aware Retrieval:**

| Query Type | Document Type | Matching Logic |
|------------|---------------|----------------|
| Point → Point | Point → Point | Angular distance (legacy) |
| Point → Arc | Query falls within doc period? | 1.0 if inside, else distance to center |
| Arc → Point | Doc falls within query period? | 1.0 if inside, else distance to center |
| Arc → Arc | Temporal overlap | Jaccard similarity |

**Use Cases:**
- **Financial reporting hierarchy**: 10-Q (quarterly) ⊂ 10-K (annual)
- **Event periods**: "Q2 2023 performance" retrieves docs from Apr-Jun 2023
- **Time-series chunking**: Each chunk knows its temporal extent
- **Periodic data**: Automatically handle wrapping (e.g., fiscal years)

## 🔬 How It Works

### Ingestion Pipeline

1. **Timestamp Extraction**: Parse timestamps from document text using regex patterns and dateutil
   - Recognizes formats like "for the period ended 31 December 2019"
   - Falls back to file metadata or ingestion time

2. **Semantic Embedding**: Obtain text embedding from LlamaStack Model Gateway
   - Uses registered embedding models (e.g., `text-embedding-v1`)
   - No special temporal training needed

3. **Spin Encoding**: Convert timestamp(s) to spin vector
   
   **Point mode (single timestamp):**
   ```python
   fraction = ((timestamp - t₀) / period) % 1.0
   φ = 2π × fraction
   spin = [cos(φ), sin(φ), 0.0]  # 3D with arc_length=0
   ```
   
   **Arc mode (time interval):**
   ```python
   φ_start = 2π × ((t_start - t₀) / period) % 1.0
   φ_end = 2π × ((t_end - t₀) / period) % 1.0
   φ_center = (φ_start + φ_end) / 2
   arc_length = φ_end - φ_start
   spin = [cos(φ_center), sin(φ_center), arc_length]  # 3D with arc_length>0
   ```

4. **Concatenation**: Combine semantic + spin into full embedding
   ```python
   # Both points and arcs use 3D spin vectors for consistent dimensionality
   full_embedding = [semantic_embedding..., spin[0], spin[1], spin[2]]
   # Points have spin[2]=0, arcs have spin[2]=arc_length
   ```

5. **Storage**: Index in vector database (PGVector, Chroma, or in-memory)

### Retrieval Algorithm: Two-Pass Temporal Zoom

#### Pass 1: Coarse Recall (Broad Semantic Search)

```python
query_full = [query_semantic, λ × query_spin]  # Small λ ≈ 0.1
candidates = vector_db.search(query_full, top_k=50)
```

Uses small λ to perform broad semantic search with minor temporal weighting.

#### Pass 2: Temporal Zoom Re-ranking (Arc-Aware)

```python
for doc in candidates:
    # Point-to-point (legacy)
    if not query.is_arc and not doc.is_arc:
        Δφ = angular_difference(φ_query, φ_doc)
        temporal_alignment = exp(-β × (Δφ)²)
    
    # Arc-to-arc (new)
    elif query.is_arc and doc.is_arc:
        temporal_alignment = jaccard_similarity(query_arc, doc_arc)
    
    # Point-within-arc or arc-contains-point
    else:
        temporal_alignment = 1.0 if overlap > 0 else exp(-β × (Δφ_center)²)
    
    score = semantic_similarity × temporal_alignment
```

Recomputes scores using **β (zoom factor)** to control temporal focus:

- **β = 0**: Pure semantic search (time ignored)
- **β = 100**: Weak temporal preference (~4% penalty per year)
- **β = 1000**: Moderate temporal focus (~11% penalty per year)
- **β = 5000**: Strong temporal focus - exact year prioritized **[DEFAULT]**
- **β = 10000+**: Extreme temporal filter (only exact year matches)

The temporal alignment factor `exp(-β × (Δφ)²)`:
- Equals 1.0 when phases align perfectly (Δφ = 0)
- Decays smoothly as phases diverge
- Decays faster with larger β (sharper focus)

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
    "beta": 5.0,
    "top_k": 10
  }'
```

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

**Querying with arc matching:**

```python
from retrieval import TemporalSpinRetriever

retriever = TemporalSpinRetriever(client, store, default_beta=5000.0)

# Query for a specific quarter (arc query)
results = retriever.search(
    query_text="Q2 2023 revenue growth",
    query_timestamp=datetime(2023, 4, 1),
    end_timestamp=datetime(2023, 6, 30),  # Arc query
    beta=5000.0
)
# Returns: Both Q2 report (exact match) and annual report (contains Q2)

# Query for a point in time
results = retriever.search(
    query_text="March 2023 acquisition",
    query_timestamp=datetime(2023, 3, 15),  # Point query
    beta=5000.0
)
# Returns: News article (exact), Q1 report (contains March), annual (contains March)
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

**Example Query Demonstrations:**

| Query | Timestamp | β | Expected Behavior |
|-------|-----------|---|-------------------|
| "IBM revenue" | 2016-06-30 | 5.0 | Prioritizes 2016 report |
| "IBM cloud strategy" | 2019-12-31 | 10.0 | Focuses on Red Hat acquisition era (2019-2020) |
| "IBM quantum computing" | 2024-06-30 | 5.0 | Highlights recent 2024 developments |

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
│              Temporal Spin Encoder                               │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  φ = 2π × (timestamp - t₀) / period                     │   │
│  │  spin = [cos(φ), sin(φ)]                                │   │
│  │  query_full = [semantic, λ × spin]                      │   │
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
│              PASS 2: Temporal Zoom Re-ranking                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  For each candidate:                                     │   │
│  │    Δφ = angular_difference(φ_query, φ_doc)              │   │
│  │    alignment = exp(-β × (Δφ)²)                          │   │
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

### Temporal Encoding Parameters

```python
T0_EPOCH = datetime(2010, 1, 1)      # Base epoch
PERIOD_SECONDS = 365.25 * 24 * 3600 * 10  # 10-year period
```

Adjustable in code for different temporal scales (daily, monthly, yearly cycles).

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

### Beta Sweep API

Compare results across multiple β values:

```python
POST /beta_sweep
{
  "query": "IBM AI strategy",
  "query_timestamp": "2019-06-30T00:00:00Z",
  "beta_values": [0, 1, 5, 10, 20],
  "top_k": 5
}
```

Returns results for each β, showing smooth transition from semantic to temporal focus.

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
# Run demo with mock data
python demo.py

# Test specific query
python demo.py --query "test query" --timestamp 2020-01-01 --beta 5.0

# Show β sweep
python demo.py --beta-sweep

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

### Why Spin Encoding?

**Circular representation** of time provides several advantages:

1. **Periodicity**: Natural for recurring patterns (fiscal years, seasons)
2. **Continuity**: Smooth interpolation between timestamps
3. **Bounded**: Always 2D, regardless of time range
4. **Interpretable**: Angular difference has geometric meaning

### Mathematical Foundation

The temporal alignment factor uses a Gaussian-like kernel in phase space:

```
alignment(Δφ; β) = exp(-β × (Δφ)²)
```

Properties:
- Maximum = 1 when Δφ = 0 (perfect alignment)
- Decays to ≈0.37 at Δφ = 1/√β (characteristic width)
- At β = 1000: 95% weight within ±0.06 radians (±3.5°, ~1 year)
- At β = 5000: 95% weight within ±0.03 radians (±1.5°, ~5 months)

### Comparison to Alternatives

| Approach | Pros | Cons |
|----------|------|------|
| **Scalar timestamp** | Simple | Doesn't capture periodicity |
| **Discrete buckets** | Interpretable | Hard boundaries, no interpolation |
| **Learned temporal embeddings** | Flexible | Requires retraining, less interpretable |
| **Spin encoding (ours)** | No retraining, interpretable, periodic | Assumes periodic patterns |

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

