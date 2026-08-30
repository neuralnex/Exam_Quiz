import logging
from typing import Any, Dict, List, Optional

try:
    from pinecone import Pinecone, ServerlessSpec
except ImportError:
    Pinecone = None  # type: ignore[misc, assignment]
    ServerlessSpec = None  # type: ignore[misc, assignment]

from src.config import settings
from src.vectorstore.base import BaseVectorStore
from src.vectorstore.inmemory_store import InMemoryVectorStore

logger = logging.getLogger(__name__)


class PineconeStore(BaseVectorStore):
    """
    Production-grade Pinecone vector store client with metadata filtering,
    batch upserts, namespace isolation, and automatic fallback.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        index_name: Optional[str] = None,
        dimension: Optional[int] = None,
    ):
        self.api_key = api_key or settings.PINECONE_API_KEY
        self.index_name = index_name or settings.PINECONE_INDEX
        self.dimension = dimension or settings.EMBEDDING_DIMENSION
        self._index = None
        self._fallback_store = InMemoryVectorStore()
        self.is_connected = False
        self._init_pinecone()

    def _init_pinecone(self):
        if not self.api_key or self.api_key.startswith("your_"):
            logger.info("Pinecone API key not configured. Operating in local in-memory vectorstore mode.")
            self.is_connected = False
            return

        try:
            if Pinecone is None or ServerlessSpec is None:
                raise ImportError("pinecone is not installed")

            pc = Pinecone(api_key=self.api_key)
            existing_indexes = [i.name for i in pc.list_indexes()]

            if self.index_name not in existing_indexes:
                logger.info(f"Creating Pinecone index: {self.index_name} with dim {self.dimension}")
                pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                )

            self._index = pc.Index(self.index_name)
            self.is_connected = True
            logger.info(f"Connected to Pinecone index: {self.index_name}")
        except Exception as e:
            logger.warning(f"Failed to connect to Pinecone ({e}). Falling back to local vector store.")
            self.is_connected = False

    def upsert_documents(
        self,
        ids: List[str],
        vectors: List[List[float]],
        metadatas: List[Dict[str, Any]],
        namespace: Optional[str] = None,
    ) -> int:
        # Also always sync to local store for offline redundancy
        self._fallback_store.upsert_documents(ids, vectors, metadatas, namespace)

        if not self.is_connected or self._index is None:
            return len(ids)

        try:
            batch_size = 100
            total_upserted = 0
            for i in range(0, len(ids), batch_size):
                batch_ids = ids[i : i + batch_size]
                batch_vecs = vectors[i : i + batch_size]
                batch_meta = metadatas[i : i + batch_size]

                # Pinecone vectors format: (id, vector, metadata)
                vectors_payload = [
                    (bid, bvec, bmeta)
                    for bid, bvec, bmeta in zip(batch_ids, batch_vecs, batch_meta)
                ]

                kwargs = {"vectors": vectors_payload}
                if namespace:
                    kwargs["namespace"] = namespace

                self._index.upsert(**kwargs)
                total_upserted += len(batch_ids)

            logger.info(f"Upserted {total_upserted} chunks into Pinecone.")
            return total_upserted
        except Exception as e:
            logger.error(f"Pinecone upsert failed: {e}. Defaulting to in-memory fallback store.")
            return len(ids)

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if not self.is_connected or self._index is None:
            return self._fallback_store.search(
                query_vector, top_k=top_k, filter_dict=filter_dict, namespace=namespace
            )

        try:
            kwargs = {
                "vector": query_vector,
                "top_k": top_k,
                "include_metadata": True,
            }
            if filter_dict:
                kwargs["filter"] = filter_dict
            if namespace:
                kwargs["namespace"] = namespace

            response = self._index.query(**kwargs)
            results = []
            for match in response.get("matches", []):
                meta = match.get("metadata", {})
                results.append({
                    "id": match.get("id"),
                    "score": match.get("score", 0.0),
                    "metadata": meta,
                    "text": meta.get("text", ""),
                })
            return results
        except Exception as e:
            logger.warning(f"Pinecone query error: {e}. Falling back to in-memory search.")
            return self._fallback_store.search(
                query_vector, top_k=top_k, filter_dict=filter_dict, namespace=namespace
            )

    def delete_course(self, course_code: str, namespace: Optional[str] = None) -> bool:
        self._fallback_store.delete_course(course_code, namespace)
        if not self.is_connected or self._index is None:
            return True

        try:
            filter_dict = {"course": course_code}
            kwargs = {"filter": filter_dict, "delete_all": False}
            if namespace:
                kwargs["namespace"] = namespace
            self._index.delete(**kwargs)
            return True
        except Exception as e:
            logger.warning(f"Pinecone delete_course error: {e}")
            return False

    def delete_document(self, course_code: str, filename: str, namespace: Optional[str] = None) -> bool:
        self._fallback_store.delete_document(course_code, filename, namespace)
        if not self.is_connected or self._index is None:
            return True

        try:
            filter_dict = {"course": course_code, "source": filename}
            kwargs = {"filter": filter_dict}
            if namespace:
                kwargs["namespace"] = namespace
            self._index.delete(**kwargs)
            return True
        except Exception as e:
            logger.warning(f"Pinecone delete_document error: {e}")
            return False

    def get_course_stats(self, course_code: str) -> Dict[str, Any]:
        return self._fallback_store.get_course_stats(course_code)


def get_vector_store() -> BaseVectorStore:
    """Factory to instantiate vector store with Pinecone / fallback."""
    return PineconeStore()
