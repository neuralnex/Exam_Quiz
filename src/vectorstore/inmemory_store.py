import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

from src.config import settings, BASE_DIR
from src.vectorstore.base import BaseVectorStore

logger = logging.getLogger(__name__)


class InMemoryVectorStore(BaseVectorStore):
    """
    Lightweight vector store with Cosine Similarity and JSON file persistence.
    Acts as zero-dependency local store / offline fallback for Pinecone.
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or (BASE_DIR / ".vectorstore_data.json")
        self.records: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self):
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    self.records = json.load(f)
                logger.info(f"Loaded {len(self.records)} records from local vector storage.")
            except Exception as e:
                logger.warning(f"Could not load local vector storage: {e}")
                self.records = {}

    def _save(self):
        try:
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(self.records, f)
        except Exception as e:
            logger.warning(f"Could not save local vector storage: {e}")

    def upsert_documents(
        self,
        ids: List[str],
        vectors: List[List[float]],
        metadatas: List[Dict[str, Any]],
        namespace: Optional[str] = None,
    ) -> int:
        count = 0
        for doc_id, vec, meta in zip(ids, vectors, metadatas):
            self.records[doc_id] = {
                "id": doc_id,
                "values": vec,
                "metadata": meta,
                "namespace": namespace or "",
            }
            count += 1
        self._save()
        return count

    def _match_filter(self, metadata: Dict[str, Any], filter_dict: Optional[Dict[str, Any]]) -> bool:
        if not filter_dict:
            return True
        for k, v in filter_dict.items():
            if isinstance(v, dict):
                # Handle $in operator
                if "$in" in v:
                    allowed = v["$in"]
                    curr = metadata.get(k)
                    if curr not in allowed:
                        return False
                elif "$eq" in v:
                    if metadata.get(k) != v["$eq"]:
                        return False
            else:
                if metadata.get(k) != v:
                    return False
        return True

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        q_vec = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            return []

        results = []
        for doc_id, item in self.records.items():
            item_namespace = item.get("namespace") or ""
            if namespace is not None and item_namespace != namespace:
                continue

            meta = item.get("metadata", {})
            if not self._match_filter(meta, filter_dict):
                continue
            
            doc_vec = np.array(item.get("values", []), dtype=np.float32)
            doc_norm = np.linalg.norm(doc_vec)
            if doc_norm == 0:
                continue

            similarity = float(np.dot(q_vec, doc_vec) / (q_norm * doc_norm))
            results.append({
                "id": doc_id,
                "score": similarity,
                "metadata": meta,
                "text": meta.get("text", ""),
            })

        # Sort descending by score
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def delete_course(self, course_code: str, namespace: Optional[str] = None) -> bool:
        to_delete = [
            doc_id
            for doc_id, item in self.records.items()
            if item.get("metadata", {}).get("course") == course_code
        ]
        for doc_id in to_delete:
            del self.records[doc_id]
        self._save()
        return True

    def delete_document(self, course_code: str, filename: str, namespace: Optional[str] = None) -> bool:
        to_delete = [
            doc_id
            for doc_id, item in self.records.items()
            if item.get("metadata", {}).get("course") == course_code
            and item.get("metadata", {}).get("source") == filename
        ]
        for doc_id in to_delete:
            del self.records[doc_id]
        self._save()
        return True

    def get_course_stats(self, course_code: str) -> Dict[str, Any]:
        course_records = [
            item for item in self.records.values()
            if item.get("metadata", {}).get("course") == course_code
        ]
        topics = set()
        sources = set()
        for item in course_records:
            meta = item.get("metadata", {})
            if "topic" in meta and meta["topic"]:
                topics.add(meta["topic"])
            if "source" in meta and meta["source"]:
                sources.add(meta["source"])
        return {
            "course": course_code,
            "chunks": len(course_records),
            "documents": len(sources),
            "total_chunks": len(course_records),
            "total_documents": len(sources),
            "topics": list(topics),
        }
