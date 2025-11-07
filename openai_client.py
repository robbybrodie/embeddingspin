"""
OpenAI Embedding Client
========================

Drop-in replacement for LlamaStackEmbeddingClient using OpenAI's embedding API.
Compatible with the temporal-phase spin retrieval system.

Usage:
    client = OpenAIEmbeddingClient(model="text-embedding-3-small")
    embeddings = client.embed(["document 1", "document 2"])
"""

import os
from typing import List, Optional
from openai import OpenAI


class OpenAIEmbeddingClient:
    """
    OpenAI embedding client compatible with temporal-phase spin retrieval.
    
    Provides the same interface as LlamaStackEmbeddingClient but uses
    OpenAI's hosted embedding models.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-small"
    ):
        """
        Initialize OpenAI embedding client.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Embedding model to use:
                   - text-embedding-3-small (1536-dim, fast, $0.02/1M tokens)
                   - text-embedding-3-large (3072-dim, best quality, $0.13/1M tokens)
                   - text-embedding-ada-002 (1536-dim, legacy)
        
        Raises:
            ValueError: If API key is not provided
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        
        # Set dimension based on model
        if "large" in model:
            self.dimension = 3072
        elif "small" in model or "ada" in model:
            self.dimension = 1536
        else:
            # Default assumption
            self.dimension = 1536
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
        
        Returns:
            List of embedding vectors (one per input text)
        
        Raises:
            openai.OpenAIError: If API request fails
        
        Example:
            >>> client = OpenAIEmbeddingClient()
            >>> embeddings = client.embed(["Hello world", "Test document"])
            >>> len(embeddings)
            2
            >>> len(embeddings[0])
            1536
        """
        if not texts:
            return []
        
        # Call OpenAI API
        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
            encoding_format="float"  # Ensure we get floats, not base64
        )
        
        # Extract embeddings in order
        embeddings = [item.embedding for item in response.data]
        
        return embeddings
    
    def embed_single(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text string to embed
        
        Returns:
            Embedding vector
        
        Example:
            >>> client = OpenAIEmbeddingClient()
            >>> embedding = client.embed_single("Hello world")
            >>> len(embedding)
            1536
        """
        return self.embed([text])[0]
    
    def get_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this model.
        
        Returns:
            Embedding dimension
        """
        return self.dimension

