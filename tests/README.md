# Test Suite for Temporal-Phase Spin Retrieval System

Comprehensive test suite covering all aspects of the temporal spin encoding and retrieval system.

## Test Structure

```text
tests/
├── __init__.py              # Test package initialization
├── conftest.py              # Shared fixtures and utilities
├── test_temporal_spin.py    # Unit tests for core encoding functions
├── test_retrieval.py        # Integration tests for retrieval algorithm
└── test_ingestion.py        # Tests for document ingestion pipeline
```

## Running Tests

### Run All Tests

```bash
pytest
```

### Run Specific Test File

```bash
pytest tests/test_temporal_spin.py
pytest tests/test_retrieval.py
pytest tests/test_ingestion.py
```

### Run Specific Test Class or Function

```bash
pytest tests/test_temporal_spin.py::TestComputeSpinVectorPoint
pytest tests/test_retrieval.py::TestBetaParameterEffects::test_beta_zero_pure_semantic
```

### Run Tests with Coverage

```bash
pytest --cov=. --cov-report=html
open htmlcov/index.html
```

### Run Tests by Marker

```bash
pytest -m unit              # Only unit tests
pytest -m integration       # Only integration tests
pytest -m "not slow"        # Skip slow tests
pytest -m property          # Only property-based tests
```

### Verbose Output

```bash
pytest -v                   # Verbose
pytest -vv                  # Extra verbose
pytest -s                   # Show print statements
```

## Test Categories

### Unit Tests (`test_temporal_spin.py`)

Tests for core temporal spin encoding functions:

- ✅ `compute_spin_vector` - Point and arc modes
- ✅ `angular_difference` - Circular math with wrapping
- ✅ `arc_overlap` - Arc intersection calculations
- ✅ `jaccard_similarity_arcs` - Arc similarity metrics
- ✅ `extract_timestamp_from_text` - Timestamp parsing
- ✅ `cosine_similarity` - Vector similarity
- ✅ `normalize_vector` - Vector normalization
- ✅ Property-based tests for circular math edge cases

### Integration Tests (`test_retrieval.py`)

Tests for the complete retrieval pipeline:

- ✅ Two-pass retrieval algorithm (coarse recall + temporal zoom)
- ✅ Beta parameter effects on temporal focus
- ✅ Arc-to-arc, point-to-arc, and arc-to-point queries
- ✅ Multi-scale temporal alignment (quarter/decade/century)
- ✅ Hard boundary checking for arc queries
- ✅ Beta sweep functionality
- ✅ Result explanation and formatting

### Ingestion Tests (`test_ingestion.py`)

Tests for document ingestion:

- ✅ Single document ingestion (point and arc modes)
- ✅ Batch ingestion with mixed modes
- ✅ Timestamp extraction from text
- ✅ Embedding generation and concatenation
- ✅ Metadata handling and preservation
- ✅ Vector store integration
- ✅ Error handling and edge cases

## Test Fixtures

### Available Fixtures (from `conftest.py`)

#### Client Fixtures

- `mock_embedding_client` - Mock embedding client for testing
- `empty_vector_store` - Empty in-memory vector store
- `populated_vector_store` - Vector store with sample documents

#### Pipeline Fixtures

- `ingestion_pipeline` - Document ingestion pipeline
- `retriever` - Retriever with populated data

#### Data Fixtures

- `get_sample_documents()` - Documents with known temporal relationships
- `get_quarterly_reports()` - Quarterly financial reports as arcs
- `generate_quarterly_dates()` - Generate quarterly date ranges

#### Assertion Helpers

- `assert_temporal_ordering()` - Verify temporal ordering of results
- `assert_phase_alignment()` - Verify phase angle alignment
- `assert_arc_contains_point()` - Verify point is within arc

#### Validation Helpers

- `is_valid_phase()` - Check if phase angle is valid
- `is_valid_spin_vector()` - Check if spin vector is valid (9D)
- `is_unit_circle_point()` - Check if (x,y) is on unit circle

## Property-Based Testing

The test suite uses [Hypothesis](https://hypothesis.readthedocs.io/) for property-based testing of circular math functions. These tests automatically generate diverse inputs to verify mathematical properties hold across the entire input space.

Examples:

- Angular difference is symmetric: `Δφ(a,b) = Δφ(b,a)`
- Angular difference is bounded: `0 ≤ Δφ ≤ π`
- Arc overlap is non-negative
- Jaccard self-similarity equals 1

## Test Data

The test suite includes realistic sample data:

### Sample Documents

- 10 Apple-related documents spanning 2020-2023
- Documents at various temporal intervals (same day, week, month, quarter, year)
- Both point documents (instant) and arc documents (periods)

### Quarterly Reports

- 5 quarterly financial reports with arc encoding
- Q1-Q4 2020 + Q1 2021
- Realistic revenue figures and metadata

### Known Temporal Relationships

- Same day: Should cluster together (high temporal alignment)
- One week apart: Should be close (moderate alignment)
- One quarter apart: Same quarter discrimination
- One year apart: Should be distant (low alignment)
- Multiple years apart: Should be very distant

## Coverage Goals

Target coverage: **≥ 90%** for core modules

Current coverage (run `pytest --cov` to update):

- `temporal_spin.py`: Core encoding functions
- `retrieval.py`: Retrieval algorithm
- `ingestion.py`: Ingestion pipeline
- `vector_store.py`: Vector store operations

## Continuous Integration

For CI/CD pipelines, use:

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests with coverage
pytest --cov=. --cov-report=xml --cov-report=term

# Check coverage threshold (e.g., 90%)
pytest --cov=. --cov-fail-under=90
```

## Writing New Tests

### Test Naming Convention

- Test files: `test_*.py`
- Test classes: `Test*` (e.g., `TestComputeSpinVector`)
- Test methods: `test_*` (e.g., `test_point_mode_returns_9d_vector`)

### Test Structure

```python
def test_feature_description(fixture1, fixture2):
    """Clear docstring explaining what is tested."""
    # Arrange
    setup_code()

    # Act
    result = function_under_test()

    # Assert
    assert result == expected
```

### Using Fixtures

```python
def test_with_fixtures(ingestion_pipeline, empty_vector_store):
    """Test using pre-configured fixtures."""
    doc = ingestion_pipeline.ingest_document(...)
    assert empty_vector_store.count() == 1
```

### Property-Based Tests

```python
from hypothesis import given
import hypothesis.strategies as st

@given(
    phi1=st.floats(min_value=0, max_value=2*math.pi),
    phi2=st.floats(min_value=0, max_value=2*math.pi)
)
def test_property_holds(phi1, phi2):
    """Test that property holds for all inputs."""
    result = angular_difference(phi1, phi2)
    assert 0 <= result <= math.pi
```

## Debugging Tests

### Run Specific Failed Test

```bash
pytest tests/test_temporal_spin.py::TestComputeSpinVectorPoint::test_point_mode_returns_9d_vector
```

### Drop into Debugger on Failure

```bash
pytest --pdb
```

### Show Local Variables on Failure

```bash
pytest -l
```

### Run with More Output

```bash
pytest -vv -s
```

## Known Issues / TODO

- [ ] Add tests for OpenAI embedding client integration
- [ ] Add tests for Chroma vector store backend
- [ ] Add tests for PGVector backend
- [ ] Add performance benchmarks
- [ ] Add tests for concurrent ingestion
- [ ] Add tests for API endpoints (FastAPI)

## Contributing

When adding new functionality:

1. Write tests first (TDD approach)
2. Ensure all existing tests pass
3. Aim for ≥90% coverage of new code
4. Add docstrings explaining what is tested
5. Use descriptive test names
6. Group related tests in classes

## Questions?

See main README.md or contact: <robbytherobot@redhat.com>
