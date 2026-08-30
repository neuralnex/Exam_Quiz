import hashlib
import logging
from typing import List, Optional
import numpy as np

from src.config import settings
from src.embeddings.base import BaseEmbeddingProvider

logger = logging.getLogger(__name__)

try:
    from fastembed import TextEmbedding
except ImportError:
    TextEmbedding = None  # type: ignore[misc, assignment]

_provider: Optional[BaseEmbeddingProvider] = None


class FastEmbedProvider(BaseEmbeddingProvider):
    """FastEmbed Provider using ONNX-runtime for high-performance CPU embeddings."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self._model = None
        self._dimension = settings.EMBEDDING_DIMENSION
        self._get_model()

    def _get_model(self):
        if self._model is None:
            try:
                if TextEmbedding is None:
                    raise ImportError("fastembed is not installed")
                logger.info(f"Loading FastEmbed model: {self.model_name}")
                self._model = TextEmbedding(model_name=self.model_name)
            except Exception as e:
                logger.warning(f"FastEmbed load failed ({e}). Falling back to SentenceTransformer or local.")
                raise e
        return self._model

    def embed_text(self, text: str) -> List[float]:
        try:
            model = self._get_model()
            embeddings = list(model.embed([text]))
            vec = embeddings[0]
            if hasattr(vec, "tolist"):
                return vec.tolist()
            return list(vec)
        except Exception as e:
            logger.warning(f"FastEmbed text embedding failed: {e}. Using fallback vectorizer.")
            return FallbackLocalEmbeddingProvider(dimension=self.dimension).embed_text(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        try:
            model = self._get_model()
            embeddings = list(model.embed(texts))
            return [
                e.tolist() if hasattr(e, "tolist") else list(e)
                for e in embeddings
            ]
        except Exception as e:
            logger.warning(f"FastEmbed doc embedding failed: {e}. Using fallback vectorizer.")
            return FallbackLocalEmbeddingProvider(dimension=self.dimension).embed_documents(texts)

    @property
    def dimension(self) -> int:
        return self._dimension


class SentenceTransformerProvider(BaseEmbeddingProvider):
    """SentenceTransformer Provider using HuggingFace models."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or "sentence-transformers/all-MiniLM-L6-v2"
        self._model = None
        self._dimension = 384

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading SentenceTransformer model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            self._dimension = self._model.get_sentence_embedding_dimension()
        return self._model

    def embed_text(self, text: str) -> List[float]:
        try:
            model = self._get_model()
            vec = model.encode(text, convert_to_numpy=True)
            return vec.tolist()
        except Exception as e:
            logger.warning(f"SentenceTransformer embedding error: {e}. Using fallback.")
            return FallbackLocalEmbeddingProvider(dimension=self.dimension).embed_text(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        try:
            model = self._get_model()
            vecs = model.encode(texts, convert_to_numpy=True, batch_size=32)
            return [v.tolist() for v in vecs]
        except Exception as e:
            logger.warning(f"SentenceTransformer doc embedding error: {e}. Using fallback.")
            return FallbackLocalEmbeddingProvider(dimension=self.dimension).embed_documents(texts)

    @property
    def dimension(self) -> int:
        return self._dimension


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """API-based Embedding Provider using OpenAI."""

    def __init__(self, model_name: str = "text-embedding-3-small", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or settings.OPENAI_API_KEY
        self._dimension = 1536

    def embed_text(self, text: str) -> List[float]:
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key)
        response = client.embeddings.create(input=text, model=self.model_name)
        return response.data[0].embedding

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key)
        response = client.embeddings.create(input=texts, model=self.model_name)
        return [item.embedding for item in response.data]

    @property
    def dimension(self) -> int:
        return self._dimension


class FallbackLocalEmbeddingProvider(BaseEmbeddingProvider):
    """
    Deterministic Bag-of-Features hashing embedding provider for offline / test environments
    or when heavy weights cannot be downloaded over network.
    Generates normalized 384-d vectors with high semantic hashing consistency.
    """

    def __init__(self, dimension: int = 384):
        self._dim = dimension

    def _hash_vector(self, text: str) -> List[float]:
        vec = np.zeros(self._dim, dtype=np.float32)
        words = text.lower().split()
        if not words:
            return vec.tolist()
        
        for w in words:
            # Deterministic hash feature mapping
            h = int(hashlib.md5(w.encode("utf-8")).hexdigest(), 16)
            idx = h % self._dim
            sign = 1.0 if ((h >> 8) & 1) else -1.0
            vec[idx] += sign

        # L2 Normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def embed_text(self, text: str) -> List[float]:
        return self._hash_vector(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._hash_vector(t) for t in texts]

    @property
    def dimension(self) -> int:
        return self._dim


def get_embedding_provider() -> BaseEmbeddingProvider:
    """Factory to instantiate the configured embedding provider."""
    global _provider
    if _provider is not None:
        return _provider

    provider_name = settings.EMBEDDING_PROVIDER.lower()
    
    if provider_name == "fastembed":
        try:
            _provider = FastEmbedProvider(model_name=settings.EMBEDDING_MODEL)
        except Exception as e:
            logger.warning(f"Failed to instantiate FastEmbedProvider: {e}. Trying SentenceTransformers.")
            try:
                _provider = SentenceTransformerProvider()
            except Exception:
                _provider = FallbackLocalEmbeddingProvider(dimension=settings.EMBEDDING_DIMENSION)

    elif provider_name == "sentence-transformers":
        try:
            _provider = SentenceTransformerProvider(model_name=settings.EMBEDDING_MODEL)
        except Exception as e:
            logger.warning(f"Failed to instantiate SentenceTransformerProvider: {e}. Fallback to local.")
            _provider = FallbackLocalEmbeddingProvider(dimension=settings.EMBEDDING_DIMENSION)

    elif provider_name == "openai":
        _provider = OpenAIEmbeddingProvider(model_name=settings.EMBEDDING_MODEL)

    elif provider_name == "local":
        _provider = FallbackLocalEmbeddingProvider(dimension=settings.EMBEDDING_DIMENSION)

    else:
        logger.warning(f"Unknown provider '{provider_name}'. Defaulting to FastEmbed.")
        _provider = FastEmbedProvider()

    return _provider
