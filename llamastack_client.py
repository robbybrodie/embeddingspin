"""
LlamaStack Embedding Client
============================

Wrapper for Red Hat AI 3 (LlamaStack) Model Gateway API to obtain embeddings.

This client provides a simple interface to any registered embedding model
in the LlamaStack environment (e.g., text-embedding-v1, nomic-embed-text, etc.).
"""

import os
import json
from typing import List, Optional
import httpx
from dataclasses import dataclass


@dataclass
class EmbeddingResponse:
    """Response from embedding API."""
    embeddings: List[List[float]]
    model: str
    dimension: int


class LlamaStackEmbeddingClient:
    """
    Client for LlamaStack Model Gateway embedding API.
    
    In Red Hat AI 3, the Model Gateway provides a unified interface to
    registered embedding models. This client handles authentication and
    batch embedding requests.
    
    Usage:
        client = LlamaStackEmbeddingClient(
            base_url="http://localhost:8000",
            model_name="text-embedding-v1"
        )
        embeddings = client.embed(["document 1", "document 2"])
    """
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        model_name: str = "text-embedding-v1",
        api_key: Optional[str] = None,
        timeout: float = 30.0
    ):
        """
        Initialize LlamaStack embedding client.
        
        Args:
            base_url: LlamaStack API base URL (defaults to env LLAMASTACK_URL)
            model_name: Name of registered embedding model
            api_key: Optional API key (defaults to env LLAMASTACK_API_KEY)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url or os.getenv(
            "LLAMASTACK_URL",
            "http://localhost:8000"
        )
        self.model_name = model_name
        self.api_key = api_key or os.getenv("LLAMASTACK_API_KEY")
        self.timeout = timeout
        
        # Remove trailing slash
        self.base_url = self.base_url.rstrip("/")
        
        # Construct embedding endpoint
        # Standard LlamaStack route: /v1/embeddings or /embeddings
        self.embeddings_url = f"{self.base_url}/v1/embeddings"
        
        # Build headers
        self.headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"
    
    def embed(
        self,
        texts: List[str],
        normalize: bool = True
    ) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            normalize: Whether to L2-normalize embeddings (recommended)
        
        Returns:
            List of embedding vectors (one per input text)
        
        Raises:
            httpx.HTTPError: If API request fails
            ValueError: If response is invalid
        """
        if not texts:
            return []
        
        # Prepare request payload
        payload = {
            "model": self.model_name,
            "input": texts,
        }
        
        # Add normalization flag if supported by your LlamaStack deployment
        # Some embedding models support this natively
        if normalize:
            payload["encoding_format"] = "float"
        
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    self.embeddings_url,
                    headers=self.headers,
                    json=payload
                )
                response.raise_for_status()
                
                data = response.json()
                
                # Extract embeddings from response
                # Standard OpenAI-compatible format:
                # {"data": [{"embedding": [...], "index": 0}, ...]}
                if "data" in data:
                    embeddings = [item["embedding"] for item in sorted(
                        data["data"], key=lambda x: x.get("index", 0)
                    )]
                elif "embeddings" in data:
                    embeddings = data["embeddings"]
                else:
                    raise ValueError(f"Unexpected response format: {data.keys()}")
                
                # Optional: L2-normalize if not done by server
                if normalize:
                    embeddings = [self._normalize(emb) for emb in embeddings]
                
                return embeddings
                
        except httpx.HTTPError as e:
            raise RuntimeError(
                f"Failed to get embeddings from {self.embeddings_url}: {e}"
            ) from e
    
    def embed_single(self, text: str, normalize: bool = True) -> List[float]:
        """
        Convenience method to embed a single text.
        
        Args:
            text: Text string to embed
            normalize: Whether to L2-normalize
        
        Returns:
            Single embedding vector
        """
        embeddings = self.embed([text], normalize=normalize)
        return embeddings[0] if embeddings else []
    
    @staticmethod
    def _normalize(vec: List[float]) -> List[float]:
        """L2-normalize a vector."""
        import math
        norm = math.sqrt(sum(x * x for x in vec))
        if norm == 0:
            return vec
        return [x / norm for x in vec]
    
    def get_embedding_dimension(self) -> int:
        """
        Determine embedding dimension by making a test request.
        
        Returns:
            Dimensionality of embeddings from this model
        """
        test_embedding = self.embed_single("test")
        return len(test_embedding)


class MockEmbeddingClient:
    """
    Mock embedding client for testing without LlamaStack.
    
    Generates random normalized embeddings of specified dimension.
    Useful for development and testing the spin retrieval logic.
    """
    
    def __init__(self, model_name: str = "mock-embed", dimension: int = 384):
        """
        Initialize mock client.
        
        Args:
            model_name: Mock model name
            dimension: Embedding vector dimension
        """
        self.model_name = model_name
        self.dimension = dimension
        import random
        self.rng = random.Random(42)  # Fixed seed for reproducibility
    
    def embed(self, texts: List[str], normalize: bool = True) -> List[List[float]]:
        """Generate mock embeddings."""
        embeddings = []
        for text in texts:
            # Generate deterministic embedding based on text hash
            seed = hash(text) % (2**32)
            rng = self.rng.__class__(seed)
            
            # Random vector
            vec = [rng.gauss(0, 1) for _ in range(self.dimension)]
            
            # Add small bias based on text content for semantic similarity
            # This makes similar texts have slightly more similar embeddings
            words = text.lower().split()
            for i, word in enumerate(words[:10]):
                idx = hash(word) % self.dimension
                vec[idx] += 0.5
            
            # Normalize
            if normalize:
                import math
                norm = math.sqrt(sum(x * x for x in vec))
                if norm > 0:
                    vec = [x / norm for x in vec]
            
            embeddings.append(vec)
        
        return embeddings
    
    def embed_single(self, text: str, normalize: bool = True) -> List[float]:
        """Embed a single text."""
        return self.embed([text], normalize=normalize)[0]
    
    def get_embedding_dimension(self) -> int:
        """Return embedding dimension."""
        return self.dimension

