# Deployment Guide

Production deployment guide for Temporal-Phase Spin Retrieval System.

## 🏢 Red Hat AI 3 (LlamaStack) Deployment

### Prerequisites

1. Red Hat AI 3 environment with LlamaStack
2. PostgreSQL with pgvector extension
3. Python 3.8+

### Step 1: Install Dependencies

```bash
# Clone repository
git clone <your-repo>
cd embeddingspin

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# For PGVector support
pip install psycopg2-binary
```

### Step 2: Configure PostgreSQL with pgvector

```sql
-- Connect to PostgreSQL
psql -U postgres

-- Create database
CREATE DATABASE temporal_spin_db;

-- Connect to database
\c temporal_spin_db

-- Enable pgvector extension
CREATE EXTENSION vector;

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE temporal_spin_db TO your_user;
```

### Step 3: Configure Environment

Create `.env` file:

```bash
# LlamaStack Configuration
USE_MOCK_EMBEDDINGS=false
LLAMASTACK_URL=http://llamastack-service:8000
LLAMASTACK_API_KEY=your_api_key_here
EMBEDDING_MODEL=text-embedding-v1

# Vector Store Configuration
VECTOR_STORE=pgvector
DATABASE_URL=postgresql://user:password@postgres-host:5432/temporal_spin_db

# Application Configuration
PORT=8080
HOST=0.0.0.0
LOAD_DEMO_DATA=false

# Optional: Temporal encoding parameters
# T0_EPOCH=2010-01-01
# PERIOD_YEARS=10
```

### Step 4: Initialize Database

```python
# init_db.py
from vector_store import PGVectorStore
import os

vector_store = PGVectorStore(
    connection_string=os.getenv('DATABASE_URL'),
    table_name='spin_documents',
    embedding_dim=386  # Adjust based on your model
)

print("✓ Database initialized")
print(f"✓ Documents: {vector_store.count()}")
```

Run:
```bash
python init_db.py
```

### Step 5: Deploy API Server

#### Option A: Direct Python

```bash
# Load environment
source .env

# Run server
python api.py
```

#### Option B: systemd Service

Create `/etc/systemd/system/temporal-spin-api.service`:

```ini
[Unit]
Description=Temporal Spin Retrieval API
After=network.target postgresql.service

[Service]
Type=simple
User=temporal-spin
WorkingDirectory=/opt/temporal-spin
EnvironmentFile=/opt/temporal-spin/.env
ExecStart=/opt/temporal-spin/venv/bin/python api.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable temporal-spin-api
sudo systemctl start temporal-spin-api
sudo systemctl status temporal-spin-api
```

#### Option C: Docker

Create `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8080

# Run
CMD ["python", "api.py"]
```

Build and run:
```bash
docker build -t temporal-spin-api .

docker run -d \
  --name temporal-spin \
  -p 8080:8080 \
  --env-file .env \
  temporal-spin-api
```

#### Option D: Kubernetes

Create `k8s-deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: temporal-spin-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: temporal-spin-api
  template:
    metadata:
      labels:
        app: temporal-spin-api
    spec:
      containers:
      - name: api
        image: temporal-spin-api:latest
        ports:
        - containerPort: 8080
        env:
        - name: USE_MOCK_EMBEDDINGS
          value: "false"
        - name: LLAMASTACK_URL
          value: "http://llamastack-service:8000"
        - name: VECTOR_STORE
          value: "pgvector"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: connection-string
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: temporal-spin-service
spec:
  selector:
    app: temporal-spin-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
  type: LoadBalancer
```

Deploy:
```bash
kubectl apply -f k8s-deployment.yaml
```

### Step 6: Load Initial Data

```python
# load_data.py
from ingestion import create_ingestion_pipeline
from vector_store import PGVectorStore
import os

# Initialize
vector_store = PGVectorStore(
    connection_string=os.getenv('DATABASE_URL'),
    table_name='spin_documents'
)

pipeline = create_ingestion_pipeline(
    vector_store=vector_store,
    llamastack_url=os.getenv('LLAMASTACK_URL'),
    model_name=os.getenv('EMBEDDING_MODEL'),
    use_mock_embeddings=False
)

# Load your documents
texts = [...]  # Your document texts
timestamps = [...]  # Corresponding timestamps
doc_ids = [...]  # Document IDs

docs = pipeline.ingest_batch(
    texts=texts,
    timestamps=timestamps,
    doc_ids=doc_ids
)

print(f"✓ Loaded {len(docs)} documents")
```

## 🔒 Security Considerations

### 1. API Key Management

Use environment variables or secrets management:

```bash
# Never commit API keys!
export LLAMASTACK_API_KEY=$(vault kv get -field=api_key secret/llamastack)
```

### 2. Database Security

- Use SSL/TLS for database connections
- Restrict network access with firewall rules
- Use strong passwords and rotate regularly

```python
# SSL connection
DATABASE_URL=postgresql://user:pass@host:5432/db?sslmode=require
```

### 3. API Authentication

Add authentication middleware to FastAPI:

```python
from fastapi import Security, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    token = credentials.credentials
    # Verify token against your auth system
    if not is_valid_token(token):
        raise HTTPException(status_code=401, detail="Invalid token")
    return token

# Apply to endpoints
@app.post("/temporal_search", dependencies=[Security(verify_token)])
async def temporal_search(request: TemporalSearchRequest):
    ...
```

### 4. Rate Limiting

Install and configure:

```bash
pip install slowapi
```

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/temporal_search")
@limiter.limit("10/minute")
async def temporal_search(request: Request, search_req: TemporalSearchRequest):
    ...
```

## 📊 Monitoring

### 1. Prometheus Metrics

```python
from prometheus_client import Counter, Histogram, generate_latest
from fastapi import Response

# Metrics
search_requests = Counter('temporal_search_requests_total', 'Total search requests')
search_duration = Histogram('temporal_search_duration_seconds', 'Search duration')

@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type="text/plain")

@app.post("/temporal_search")
async def temporal_search(request: TemporalSearchRequest):
    search_requests.inc()
    with search_duration.time():
        # ... search logic
        pass
```

### 2. Logging

```python
import logging
from logging.handlers import RotatingFileHandler

# Configure logging
handler = RotatingFileHandler(
    'temporal_spin.log',
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
handler.setFormatter(formatter)

logger = logging.getLogger('temporal_spin')
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Use in code
logger.info(f"Search query: {query_text}, beta: {beta}")
```

### 3. Health Checks

Enhanced health check:

```python
@app.get("/health")
async def health_check():
    checks = {
        "api": "healthy",
        "vector_store": "unknown",
        "embedding_client": "unknown"
    }
    
    # Check vector store
    try:
        count = vector_store.count()
        checks["vector_store"] = "healthy"
        checks["document_count"] = count
    except Exception as e:
        checks["vector_store"] = "unhealthy"
        checks["vector_store_error"] = str(e)
    
    # Check embedding client
    try:
        test_emb = embedding_client.embed_single("test")
        checks["embedding_client"] = "healthy"
        checks["embedding_dim"] = len(test_emb)
    except Exception as e:
        checks["embedding_client"] = "unhealthy"
        checks["embedding_error"] = str(e)
    
    # Overall status
    is_healthy = all(
        v == "healthy" for k, v in checks.items() 
        if k.endswith(('_store', '_client', 'api'))
    )
    
    status_code = 200 if is_healthy else 503
    return JSONResponse(content=checks, status_code=status_code)
```

## 🚀 Performance Optimization

### 1. Connection Pooling

```python
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=10,
    pool_pre_ping=True
)
```

### 2. Caching

```python
from functools import lru_cache
from datetime import datetime

@lru_cache(maxsize=1000)
def get_spin_vector_cached(timestamp_iso: str):
    """Cache spin vectors for common timestamps."""
    timestamp = datetime.fromisoformat(timestamp_iso)
    return compute_spin_vector(timestamp.timestamp())
```

### 3. Batch Processing

```python
# Process ingestion in batches
BATCH_SIZE = 100

for i in range(0, len(documents), BATCH_SIZE):
    batch = documents[i:i+BATCH_SIZE]
    pipeline.ingest_batch(
        texts=[d['text'] for d in batch],
        timestamps=[d['timestamp'] for d in batch]
    )
    print(f"Processed {i+BATCH_SIZE}/{len(documents)}")
```

### 4. Database Indexing

```sql
-- Optimize PGVector performance
CREATE INDEX CONCURRENTLY ON spin_documents 
USING ivfflat (embedding vector_cosine_ops) 
WITH (lists = 1000);  -- Adjust based on dataset size

-- Index for timestamp queries
CREATE INDEX idx_timestamp ON spin_documents(timestamp);

-- Index for metadata queries
CREATE INDEX idx_metadata ON spin_documents USING gin(metadata);
```

## 🔄 Backup and Recovery

### Database Backup

```bash
#!/bin/bash
# backup.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/temporal_spin"

# Backup PostgreSQL
pg_dump -h $DB_HOST -U $DB_USER $DB_NAME | gzip > \
  $BACKUP_DIR/db_backup_$DATE.sql.gz

# Backup Chroma (if used)
tar -czf $BACKUP_DIR/chroma_backup_$DATE.tar.gz ./chroma_db/

# Cleanup old backups (keep 30 days)
find $BACKUP_DIR -name "*.gz" -mtime +30 -delete

echo "✓ Backup completed: $DATE"
```

Schedule with cron:
```cron
0 2 * * * /opt/temporal-spin/backup.sh
```

## 📈 Scaling

### Horizontal Scaling

The API is stateless and can be scaled horizontally:

```yaml
# k8s-hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: temporal-spin-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: temporal-spin-api
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Database Scaling

For large datasets:

1. **PGVector with read replicas**
2. **Sharding by time range** (e.g., yearly shards)
3. **Distributed vector stores** (e.g., Milvus, Qdrant)

## 🧪 Testing in Production

```python
# health_check_test.py
import requests
import time

API_URL = "http://your-api-url"

def test_health():
    response = requests.get(f"{API_URL}/health")
    assert response.status_code == 200
    data = response.json()
    assert data['api'] == 'healthy'
    print("✓ Health check passed")

def test_search():
    response = requests.post(
        f"{API_URL}/temporal_search",
        json={
            "query": "test query",
            "beta": 5.0,
            "top_k": 5
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert 'results' in data
    print(f"✓ Search returned {len(data['results'])} results")

def test_performance():
    start = time.time()
    for _ in range(10):
        requests.post(
            f"{API_URL}/temporal_search",
            json={"query": "test", "beta": 5.0, "top_k": 10}
        )
    duration = time.time() - start
    avg = duration / 10
    print(f"✓ Average search time: {avg*1000:.2f}ms")

if __name__ == "__main__":
    test_health()
    test_search()
    test_performance()
```

## 📞 Support

For production issues:
- Check logs: `journalctl -u temporal-spin-api -f`
- Monitor metrics: Prometheus/Grafana dashboards
- Database status: `psql -c "SELECT count(*) FROM spin_documents;"`

---

**Production Checklist:**

- [ ] PostgreSQL with pgvector installed and configured
- [ ] Environment variables set securely
- [ ] Database backups scheduled
- [ ] Monitoring and alerts configured
- [ ] API authentication enabled
- [ ] Rate limiting configured
- [ ] Health checks passing
- [ ] Load testing completed
- [ ] Documentation updated
- [ ] Team trained on operations

Ready for production! 🚀

