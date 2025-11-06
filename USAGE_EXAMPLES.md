# Usage Examples

Complete examples demonstrating the Temporal-Phase Spin Retrieval System.

## Example 1: Basic Python Usage

```python
from datetime import datetime, timezone
from temporal_spin import compute_spin_vector
from llamastack_client import MockEmbeddingClient
from vector_store import InMemoryVectorStore
from ingestion import TemporalSpinIngestionPipeline
from retrieval import TemporalSpinRetriever

# Initialize components
embedding_client = MockEmbeddingClient(dimension=384)
vector_store = InMemoryVectorStore()

# Create ingestion pipeline
pipeline = TemporalSpinIngestionPipeline(
    embedding_client=embedding_client,
    vector_store=vector_store
)

# Ingest documents
documents = [
    ("IBM reported strong Q4 2019 earnings with cloud growth.", 
     datetime(2019, 12, 31, tzinfo=timezone.utc)),
    ("IBM announces Red Hat acquisition completion in July 2019.", 
     datetime(2019, 7, 9, tzinfo=timezone.utc)),
    ("IBM Q1 2020 results show continued cloud momentum.", 
     datetime(2020, 3, 31, tzinfo=timezone.utc)),
]

for text, timestamp in documents:
    pipeline.ingest_document(text=text, timestamp=timestamp)

# Create retriever
retriever = TemporalSpinRetriever(
    embedding_client=embedding_client,
    vector_store=vector_store
)

# Search with temporal focus
results = retriever.search(
    query_text="IBM cloud performance",
    query_timestamp=datetime(2019, 12, 1, tzinfo=timezone.utc),
    beta=5.0,  # Temporal zoom factor
    top_k_final=3
)

# Display results
for result in results:
    print(f"Rank {result.rank}: {result.timestamp.date()}")
    print(f"  Combined Score: {result.combined_score:.4f}")
    print(f"  Text: {result.text[:100]}...")
    print()
```

## Example 2: Beta Parameter Sweep

Demonstrate how β controls temporal focus:

```python
from datetime import datetime, timezone
from retrieval import TemporalSpinRetriever

# Assuming retriever is already initialized with documents

query_text = "IBM revenue"
query_timestamp = datetime(2019, 6, 30, tzinfo=timezone.utc)

# Try different β values
for beta in [0, 1, 5, 10, 20]:
    print(f"\n{'='*60}")
    print(f"β = {beta}")
    print('='*60)
    
    results = retriever.search(
        query_text=query_text,
        query_timestamp=query_timestamp,
        beta=beta,
        top_k_final=3
    )
    
    for result in results:
        year = result.timestamp.year
        score = result.combined_score
        temporal = result.temporal_alignment
        print(f"  {year}: score={score:.4f}, temporal_align={temporal:.4f}")
```

**Expected Output Pattern:**

```
β = 0 (pure semantic, time ignored)
  2018: score=0.8234, temporal_align=1.0000
  2019: score=0.8156, temporal_align=1.0000
  2020: score=0.8089, temporal_align=1.0000

β = 5 (balanced)
  2019: score=0.8156, temporal_align=1.0000
  2020: score=0.7234, temporal_align=0.8945
  2018: score=0.7123, temporal_align=0.8654

β = 20 (sharp temporal focus)
  2019: score=0.8156, temporal_align=1.0000
  2020: score=0.3421, temporal_align=0.4201
  2018: score=0.2987, temporal_align=0.3634
```

As β increases, the ranking strongly favors documents closer in time to mid-2019.

## Example 3: FastAPI Client

Use the REST API from Python:

```python
import requests
import json

API_URL = "http://localhost:8080"

# Search request
response = requests.post(
    f"{API_URL}/temporal_search",
    json={
        "query": "IBM cloud strategy",
        "query_timestamp": "2019-06-30T00:00:00Z",
        "beta": 5.0,
        "top_k": 5
    }
)

results = response.json()

print(f"Query: {results['query']}")
print(f"Beta: {results['beta']}")
print(f"Execution time: {results['execution_time_ms']:.2f}ms")
print("\nResults:")

for result in results['results']:
    print(f"  {result['rank']}. {result['timestamp'][:10]}")
    print(f"     Score: {result['combined_score']:.4f}")
    print(f"     Phase diff: {result['phi_difference_deg']:.1f}°")
```

## Example 4: Custom Timestamp Extraction

Add custom date patterns for your domain:

```python
from temporal_spin import DATE_PATTERNS, extract_timestamp_from_text

# Add pattern for "Report Date: YYYY-MM-DD"
DATE_PATTERNS.insert(0, r'Report\s+Date:\s+(\d{4}-\d{2}-\d{2})')

# Now extraction works
text = """
Medical Record
Report Date: 2023-05-15
Patient presented with symptoms...
"""

timestamp = extract_timestamp_from_text(text)
print(f"Extracted: {timestamp.date()}")  # 2023-05-15
```

## Example 5: Using Real LlamaStack Embeddings

Connect to LlamaStack Model Gateway:

```python
import os
from llamastack_client import LlamaStackEmbeddingClient
from vector_store import ChromaVectorStore
from ingestion import TemporalSpinIngestionPipeline

# Configure LlamaStack connection
os.environ['USE_MOCK_EMBEDDINGS'] = 'false'
os.environ['LLAMASTACK_URL'] = 'http://llamastack.example.com:8000'
os.environ['EMBEDDING_MODEL'] = 'text-embedding-v1'

# Initialize client
embedding_client = LlamaStackEmbeddingClient(
    base_url=os.environ['LLAMASTACK_URL'],
    model_name=os.environ['EMBEDDING_MODEL'],
    api_key=os.environ.get('LLAMASTACK_API_KEY')
)

# Use persistent Chroma store
vector_store = ChromaVectorStore(
    collection_name="my_documents",
    persist_directory="./chroma_db"
)

# Create pipeline
pipeline = TemporalSpinIngestionPipeline(
    embedding_client=embedding_client,
    vector_store=vector_store
)

# Ingest
pipeline.ingest_document(
    text="Important document text...",
    timestamp=None  # Will be extracted from text
)
```

## Example 6: Batch Ingestion from Files

Process multiple documents efficiently:

```python
from pathlib import Path
from ingestion import TemporalSpinIngestionPipeline

# Assuming pipeline is initialized

# Ingest all text files from directory
docs_dir = Path("./documents")
file_paths = list(docs_dir.glob("*.txt"))

documents = pipeline.ingest_from_files(
    file_paths=[str(p) for p in file_paths],
    extract_timestamp_from_filename=True
)

print(f"Ingested {len(documents)} documents")

# Show spin phases
for doc in documents[:5]:
    print(f"{doc.doc_id}: φ = {doc.phi:.4f} rad")
```

## Example 7: PGVector Production Setup

Use PostgreSQL with pgvector for production:

```python
from vector_store import PGVectorStore
from ingestion import create_ingestion_pipeline

# Connect to PostgreSQL with pgvector
vector_store = PGVectorStore(
    connection_string="postgresql://user:pass@localhost:5432/vectordb",
    table_name="temporal_docs",
    embedding_dim=386  # 384 semantic + 2 spin
)

# Create pipeline with real embeddings
pipeline = create_ingestion_pipeline(
    vector_store=vector_store,
    llamastack_url="http://localhost:8000",
    model_name="text-embedding-v1",
    use_mock_embeddings=False
)

# Ingest (scales to millions of documents)
texts = ["doc1...", "doc2...", ...]
timestamps = [...]
pipeline.ingest_batch(texts=texts, timestamps=timestamps)
```

## Example 8: Result Explanation

Get detailed explanation of a result:

```python
from retrieval import TemporalSpinRetriever

# Assuming retriever is initialized
results = retriever.search(
    query_text="IBM AI",
    query_timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc),
    beta=5.0,
    top_k_final=1
)

# Get detailed explanation
explanation = retriever.explain_result(results[0])
print(explanation)
```

**Output:**

```
Rank #1
Document ID: ibm-report-2020
Timestamp: 2020-12-31T00:00:00+00:00

Scores:
  Semantic Similarity: 0.8723
  Temporal Alignment:  0.9845
  Combined Score:      0.8588

Phase Information:
  Document Phase (φ_doc):   3.9270 rad (225.0°)
  Query Phase (φ_query):    3.8950 rad (223.2°)
  Phase Difference (Δφ):    0.0320 rad (1.8°)

Text Preview:
  IBM Corporation Annual Financial Report
  For the period ended December 31, 2020...
```

## Example 9: API Beta Sweep

Compare multiple β values via API:

```bash
curl -X POST "http://localhost:8080/beta_sweep" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "IBM hybrid cloud",
    "query_timestamp": "2019-12-31T00:00:00Z",
    "beta_values": [0, 5, 10, 20],
    "top_k": 3
  }'
```

## Example 10: Custom Period Configuration

Adjust temporal period for different domains:

```python
from temporal_spin import T0_SECONDS, PERIOD_SECONDS
from datetime import datetime, timezone

# For daily cycles (e.g., trading hours)
PERIOD_DAILY = 24 * 3600  # 24 hours

# For yearly cycles (e.g., seasonal patterns)
PERIOD_YEARLY = 365.25 * 24 * 3600  # 1 year

# Use in pipeline
from ingestion import TemporalSpinIngestionPipeline

pipeline = TemporalSpinIngestionPipeline(
    embedding_client=...,
    vector_store=...,
    period_seconds=PERIOD_YEARLY  # Custom period
)
```

## Tips for Best Results

### 1. Choose Appropriate β
- Start with β=5 for balanced results
- Use β=0-1 for broad semantic search
- Use β=10-20 for strict temporal filtering

### 2. Timestamp Quality Matters
- Provide explicit timestamps when available
- Add custom regex patterns for your date formats
- Verify timestamps are being extracted correctly

### 3. Adjust Period for Your Domain
- Financial reports: 1-10 year period
- News articles: 1 month - 1 year period  
- Log files: 1 day - 1 week period
- Medical records: 1-5 year period

### 4. Batch Operations
- Use `ingest_batch()` for multiple documents
- Much faster than individual `ingest_document()` calls
- Amortizes embedding API overhead

### 5. Vector Store Selection
- Development: `InMemoryVectorStore`
- Small production (< 100k docs): `ChromaVectorStore`
- Large production (> 100k docs): `PGVectorStore`

### 6. Monitor Phase Distributions
```python
# Check document phase distribution
docs = [vector_store.get_document(doc_id) for doc_id in doc_ids]
phases = [doc.phi for doc in docs if doc]

import matplotlib.pyplot as plt
plt.hist(phases, bins=50)
plt.xlabel("Phase (radians)")
plt.ylabel("Document Count")
plt.title("Temporal Phase Distribution")
plt.show()
```

Ideally, phases should be distributed across the full [0, 2π) range for best temporal zoom effect.

---

For more examples, see the demo script: `python demo.py --help`

