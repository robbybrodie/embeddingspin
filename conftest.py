"""
Pytest configuration.

The modules live flat at the repository root rather than inside a package
directory, so this file exists primarily to put that root on ``sys.path`` — pytest
prepends the directory containing the topmost ``conftest.py`` automatically.

It also pins the environment the tests run under. ``temporal_config`` reads the
epoch and year convention from ``EMBEDDINGSPIN_EPOCH`` and
``EMBEDDINGSPIN_YEAR_CONVENTION`` at import time, so a developer with those set in
their shell would otherwise see a different ``DEFAULT_HIERARCHY`` than CI does.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Must happen before temporal_config is imported anywhere.
os.environ.pop("EMBEDDINGSPIN_EPOCH", None)
os.environ.pop("EMBEDDINGSPIN_YEAR_CONVENTION", None)
os.environ.setdefault("USE_OPENAI_EMBEDDINGS", "false")
os.environ.setdefault("USE_MOCK_EMBEDDINGS", "true")

import pytest  # noqa: E402

from ingestion import TemporalSpinIngestionPipeline  # noqa: E402
from llamastack_client import MockEmbeddingClient  # noqa: E402
from retrieval import TemporalSpinRetriever  # noqa: E402
from temporal_config import DEFAULT_HIERARCHY  # noqa: E402
from vector_store import InMemoryVectorStore  # noqa: E402


@pytest.fixture
def hierarchy():
    return DEFAULT_HIERARCHY


@pytest.fixture
def embedding_client():
    """Deterministic embeddings — same text always yields the same vector."""
    return MockEmbeddingClient(dimension=64)


@pytest.fixture
def store(hierarchy):
    return InMemoryVectorStore(hierarchy=hierarchy)


@pytest.fixture
def pipeline(embedding_client, store, hierarchy):
    return TemporalSpinIngestionPipeline(
        embedding_client=embedding_client, vector_store=store, hierarchy=hierarchy
    )


@pytest.fixture
def retriever(embedding_client, store, hierarchy):
    return TemporalSpinRetriever(
        embedding_client=embedding_client,
        vector_store=store,
        hierarchy=hierarchy,
        default_beta=0.5,
    )
