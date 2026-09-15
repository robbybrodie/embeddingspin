# Prerequisites & Dependencies

This document outlines all prerequisites and dependencies needed to run the Temporal-Phase Spin Retrieval System.

## System Requirements

### Operating System
- **Linux** (Ubuntu 20.04+, RHEL 8+, etc.) - Recommended
- **macOS** (10.15+)
- **Windows** (with WSL2 recommended)

### Python Version
- **Python 3.9 or higher** (Required; the test suite runs on 3.9)
- Python 3.10 or 3.11 recommended

**Check your Python version:**
```bash
python3 --version
# Should output: Python 3.9.x or higher
```

**Install Python if needed:**
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3 python3-pip python3-venv

# macOS (with Homebrew)
brew install python@3.10

# RHEL/CentOS
sudo dnf install python3 python3-pip
```

## Python Dependencies

All Python dependencies are listed in `requirements.txt`. Install them with:

```bash
pip install -r requirements.txt
```

### Core Dependencies (Required)

| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | ≥1.21.0 | Numerical operations |
| `python-dateutil` | ≥2.8.2 | Timestamp parsing |
| `httpx` | ≥0.24.0 | LlamaStack API calls |
| `fastapi` | ≥0.104.0 | REST API server |
| `uvicorn` | ≥0.24.0 | ASGI server |
| `pydantic` | ≥2.0.0 | Data validation |

### Optional Dependencies

#### For OpenAI Embeddings
```bash
pip install openai>=1.0.0
export OPENAI_API_KEY=sk-...
```

Used when: `USE_OPENAI_EMBEDDINGS=true` (the default in `api.py`). Set
`USE_MOCK_EMBEDDINGS=true` to skip it entirely.

#### For Chroma Vector Store
```bash
pip install chromadb>=0.4.0
```

Used when: `VECTOR_STORE=chroma`

#### For PGVector Support
```bash
pip install psycopg2-binary>=2.9.0
```

Used when: `VECTOR_STORE=pgvector`

#### For Development
```bash
pip install pytest>=7.4.0 pytest-asyncio>=0.21.0
```

#### For Enhanced CLI
```bash
pip install rich>=13.0.0
```

## External Services (Optional)

### 1. LlamaStack Model Gateway

**Required for production use** (optional for demo with mock embeddings)

- **Service:** Red Hat AI 3 / LlamaStack
- **Endpoint:** Model Gateway API (usually port 8000)
- **Authentication:** API key (optional, depends on setup)

**Configuration:**
```bash
export LLAMASTACK_URL=http://localhost:8000
export LLAMASTACK_API_KEY=your_api_key_here
export EMBEDDING_MODEL=text-embedding-v1
```

**Demo mode (no LlamaStack needed):**
```bash
export USE_MOCK_EMBEDDINGS=true
```

### 2. Vector Database (Optional)

#### Option A: In-Memory (Default)
- **No setup needed**
- Good for: Development, testing, < 10k documents
- Configuration: `VECTOR_STORE=memory`

#### Option B: Chroma DB
- **No external service needed** (embedded database)
- Good for: Small to medium datasets (< 1M documents)
- Persists to disk
- Configuration: `VECTOR_STORE=chroma`

**Setup:**
```bash
pip install chromadb
export VECTOR_STORE=chroma
export CHROMA_PERSIST_DIR=./chroma_db
```

#### Option C: PostgreSQL with pgvector
- **Requires PostgreSQL 12+** with pgvector extension
- Good for: Production, large datasets (10M+ documents)
- Configuration: `VECTOR_STORE=pgvector`

**Setup:**

1. **Install PostgreSQL:**
```bash
# Ubuntu/Debian
sudo apt install postgresql postgresql-contrib

# macOS
brew install postgresql

# RHEL/CentOS
sudo dnf install postgresql-server postgresql-contrib
```

2. **Install pgvector extension:**
```bash
# Ubuntu/Debian
sudo apt install postgresql-14-pgvector

# Or build from source
git clone https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install
```

3. **Configure database:**
```sql
-- Connect to PostgreSQL
sudo -u postgres psql

-- Create database
CREATE DATABASE temporal_spin_db;

-- Connect to database
\c temporal_spin_db

-- Enable pgvector
CREATE EXTENSION vector;

-- Create user and grant permissions
CREATE USER temporal_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE temporal_spin_db TO temporal_user;
```

4. **Set connection string:**
```bash
export DATABASE_URL="postgresql://temporal_user:your_password@localhost:5432/temporal_spin_db"
```

## Installation Steps

### Quick Installation (Recommended)

```bash
# 1. Clone or extract the project
cd embeddingspin

# 2. Run quickstart script (does everything)
./quickstart.sh
```

### Manual Installation

```bash
# 1. Create virtual environment
python3 -m venv venv

# 2. Activate virtual environment
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows

# 3. Upgrade pip
pip install --upgrade pip

# 4. Install core dependencies
pip install -r requirements.txt

# 5. (Optional) Install vector store dependencies
pip install chromadb  # For Chroma
pip install psycopg2-binary  # For PGVector

# 6. Verify installation
python -c "import fastapi; import httpx; import numpy; print('✓ All core dependencies installed')"
```

## Environment Variables

Create a `.env` file or export these variables:

### Minimal Configuration (Demo Mode)
```bash
USE_OPENAI_EMBEDDINGS=false
USE_MOCK_EMBEDDINGS=true
VECTOR_STORE=memory
LOAD_DEMO_DATA=true
```

### Production Configuration
```bash
# Embeddings
USE_OPENAI_EMBEDDINGS=false
USE_MOCK_EMBEDDINGS=false
LLAMASTACK_URL=http://llamastack-service:8000
LLAMASTACK_API_KEY=your_api_key
EMBEDDING_MODEL=text-embedding-v1

# Vector Store
VECTOR_STORE=pgvector
DATABASE_URL=postgresql://user:pass@host:5432/db

# Application
PORT=8080
HOST=0.0.0.0
LOAD_DEMO_DATA=false
DEFAULT_BETA=0.5
```

### Temporal Encoding

These two change how time itself is encoded, and both are read **once, at import**.
Changing either on a populated corpus requires a re-index — see
[CHANGELOG.md](CHANGELOG.md) and the epoch section of the README.

```bash
EMBEDDINGSPIN_EPOCH=1900-01-01          # base epoch t₀; coverage runs to t₀ + 256y
EMBEDDINGSPIN_YEAR_CONVENTION=calendar  # 'calendar' (actual 365/366d) or 'linear'
```

Leave both unset unless you know why you are changing them. The defaults are what
the corpus, the tests and the published worked example assume.

## Verification

### Check Prerequisites

```bash
# Check Python version
python3 --version  # Should be 3.8+

# Check pip
pip --version

# Check virtual environment is activated
which python  # Should point to venv/bin/python
```

### Test Installation

```bash
# Test core imports
python -c "
import temporal_config
import temporal_encoding
import temporal_spin
import llamastack_client
import vector_store
import ingestion
import retrieval
import query_decomposition
print('✓ All modules imported successfully')
"

# Confirm the active hierarchy is the expected one
python -c "
from temporal_config import DEFAULT_HIERARCHY as h
print(h.fingerprint(), '| coverage through', h.coverage_end_year)
"
# → v2|1900-01-01|calendar|quarter:1:4+decade:16:16+century:256:16 | coverage through 2156

# Run the test suite (no network, no external services)
pytest

# Run demo
python demo.py

# Test API server
USE_OPENAI_EMBEDDINGS=false USE_MOCK_EMBEDDINGS=true python api.py &
curl http://localhost:8080/health
curl http://localhost:8080/stats
```

## Common Issues & Solutions

### Issue: "No module named 'dateutil'"
**Solution:**
```bash
pip install python-dateutil
```

### Issue: "No module named 'chromadb'"
**Solution:**
```bash
pip install chromadb
```

### Issue: "Could not connect to LlamaStack"
**Solution:**
```bash
# Use mock embeddings for testing
export USE_OPENAI_EMBEDDINGS=false USE_MOCK_EMBEDDINGS=true
python demo.py
```

### Issue: "No module named 'openai'"
**Solution:** `api.py` defaults to OpenAI embeddings. Either install the client and
set `OPENAI_API_KEY`, or opt out:
```bash
export USE_OPENAI_EMBEDDINGS=false USE_MOCK_EMBEDDINGS=true
```

### Issue: "retriever hierarchy is incompatible with the vector store's"
**Solution:** The corpus was written under a different epoch, year convention or
scale set than the one now configured. Either restore the old configuration or
re-index. `describe_epoch_migration(old, new)` will tell you which it is.

### Issue: "beta must be in [0, 1]"
**Solution:** β changed meaning in 2.0 — it is an interpolation weight now, not a
Gaussian width. Start at 0.5. See [CHANGELOG.md](CHANGELOG.md).

### Issue: "pg_config executable not found"
**Solution:**
```bash
# Install PostgreSQL development headers
sudo apt install libpq-dev  # Ubuntu/Debian
sudo dnf install postgresql-devel  # RHEL/CentOS
brew install postgresql  # macOS
```

### Issue: Python version too old
**Solution:**
```bash
# Install Python 3.10
sudo apt install python3.10 python3.10-venv
python3.10 -m venv venv
```

## Minimum vs. Recommended Setup

### Minimum Setup (Demo/Testing)
- Python 3.8+
- Core dependencies from requirements.txt
- No external services
- **Runs completely standalone!**

```bash
python3 -m venv venv
source venv/bin/activate
pip install fastapi uvicorn httpx pydantic python-dateutil numpy pytest
export USE_OPENAI_EMBEDDINGS=false USE_MOCK_EMBEDDINGS=true
python demo.py
```

### Recommended Setup (Development)
- Python 3.10+
- All dependencies including optional
- Chroma DB for persistence
- Mock embeddings or local LlamaStack

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export VECTOR_STORE=chroma
python demo.py
```

### Production Setup
- Python 3.10+
- All dependencies
- PostgreSQL with pgvector
- LlamaStack Model Gateway
- Monitoring and logging

See the Configuration section of [README.md](README.md) for the full variable list.

## Hardware Requirements

### Minimum
- **CPU:** 2 cores
- **RAM:** 4 GB
- **Disk:** 1 GB free space

### Recommended
- **CPU:** 4+ cores
- **RAM:** 8+ GB
- **Disk:** 10+ GB (for vector store data)

### Production
- **CPU:** 8+ cores
- **RAM:** 16+ GB
- **Disk:** 100+ GB SSD
- **Network:** Low latency to LlamaStack service

## Quick Start Checklist

- [ ] Python 3.9+ installed
- [ ] `pip` and `venv` available
- [ ] Virtual environment created and activated
- [ ] Dependencies installed: `pip install -r requirements.txt`
- [ ] (Optional) PostgreSQL + pgvector if using PGVector
- [ ] (Optional) LlamaStack URL configured if not using mock
- [ ] Environment variables set (or using defaults)
- [ ] Tests pass: `pytest`
- [ ] Demo runs successfully: `python demo.py`
- [ ] API starts successfully: `python api.py`

## Getting Help

If you encounter issues:

1. **Check prerequisites:** Ensure Python 3.9+ and pip are installed
2. **Read error messages:** Most issues are dependency-related
3. **Use mock mode:** Test without external dependencies
4. **Check documentation:** [README.md](README.md),
   [PATENT_ALIGNMENT.md](PATENT_ALIGNMENT.md), [CHANGELOG.md](CHANGELOG.md)
5. **Verify environment:** `pip list` to see installed packages

## Next Steps

Once prerequisites are met:

1. Run quickstart: `./quickstart.sh`
2. Try demo: `python demo.py`, then `python arc_demo.py` for the geometry
3. Explore API: `python api.py` → http://localhost:8080/docs
4. Read the design: [README.md](README.md)
5. Read the claim map: [PATENT_ALIGNMENT.md](PATENT_ALIGNMENT.md)

---

**TL;DR for Quick Start:**

```bash
# Minimal setup (no external dependencies)
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export USE_OPENAI_EMBEDDINGS=false USE_MOCK_EMBEDDINGS=true
python demo.py
```

That's it! The system runs completely standalone with mock embeddings.

