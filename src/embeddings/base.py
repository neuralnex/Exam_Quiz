from abc import ABC, abstractmethod
from typing import List


class BaseEmbeddingProvider(ABC):
    """Abstract Base Class for Embedding Providers."""

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding vector for a single query text."""
        pass

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embedding vectors for a list of document chunks."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimensionality of the embedding vectors."""
        pass

    def get_dimension(self) -> int:
        """Return the dimensionality of the embedding vectors."""
        return self.dimension
