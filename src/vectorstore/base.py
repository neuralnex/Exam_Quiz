from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseVectorStore(ABC):
    """Abstract Base Class for Vector Database Abstraction."""

    @abstractmethod
    def upsert_documents(
        self,
        ids: List[str],
        vectors: List[List[float]],
        metadatas: List[Dict[str, Any]],
        namespace: Optional[str] = None,
    ) -> int:
        """Upsert document chunks into vector database with metadata."""
        pass

    @abstractmethod
    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search nearest document chunks given a query vector and optional metadata filter."""
        pass

    @abstractmethod
    def delete_course(self, course_code: str, namespace: Optional[str] = None) -> bool:
        """Delete all indexed chunks belonging to a course."""
        pass

    @abstractmethod
    def delete_document(self, course_code: str, filename: str, namespace: Optional[str] = None) -> bool:
        """Delete all chunks for a specific document."""
        pass

    @abstractmethod
    def get_course_stats(self, course_code: str) -> Dict[str, Any]:
        """Get vector store statistics for a course."""
        pass
