"""
Qdrant vector store.

Manages collections, point insertion, similarity search, and deletion.
Uses 3 collections: personal_knowledge, personal_memory, conversation_memory.
All 1024-dim vectors from bge-m3, Cosine distance.
"""

import uuid
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse
from loguru import logger

from app.config import settings


class QdrantStore:
    """Qdrant vector database manager."""

    def __init__(self):
        self._client: Optional[QdrantClient] = None

    def connect(self) -> None:
        """Initialize the Qdrant client."""
        try:
            self._client = QdrantClient(url=settings.qdrant_url, timeout=30)
            logger.info(f"Qdrant client connected to {settings.qdrant_url}")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise

    @property
    def client(self) -> QdrantClient:
        """Get the Qdrant client, connecting if needed."""
        if not self._client:
            self.connect()
        return self._client

    # ── Health Check ─────────────────────────────────────────────────────

    def health_check(self) -> bool:
        """Test Qdrant connectivity."""
        try:
            collections = self.client.get_collections()
            names = [c.name for c in collections.collections]
            logger.info(f"Qdrant health check passed. Collections: {names}")
            return True
        except Exception as e:
            logger.error(f"Qdrant health check failed: {e}")
            return False

    # ── Collection Management ────────────────────────────────────────────

    def _get_distance(self) -> qmodels.Distance:
        """Map distance string from config to Qdrant enum."""
        mapping = {
            "cosine": qmodels.Distance.COSINE,
            "euclid": qmodels.Distance.EUCLID,
            "dot": qmodels.Distance.DOT,
        }
        return mapping.get(settings.qdrant_distance.lower(), qmodels.Distance.COSINE)

    def ensure_collections(self) -> None:
        """Create all 3 required collections if they don't already exist."""
        collection_names = [
            settings.qdrant_knowledge_collection,
            settings.qdrant_memory_collection,
            settings.qdrant_chat_collection,
        ]

        existing = {c.name for c in self.client.get_collections().collections}

        for name in collection_names:
            if name in existing:
                logger.info(f"Qdrant collection '{name}' already exists")
                continue

            try:
                self.client.create_collection(
                    collection_name=name,
                    vectors_config=qmodels.VectorParams(
                        size=settings.qdrant_vector_size,
                        distance=self._get_distance(),
                    ),
                )
                logger.info(f"Created Qdrant collection '{name}'")
                
                # Create Full-Text index for Hybrid Search
                # We index the common text fields
                for field in ["text", "content", "memory", "chunk_text"]:
                    self.client.create_payload_index(
                        collection_name=name,
                        field_name=field,
                        field_schema=qmodels.TextIndexParams(
                            type="text",
                            tokenizer=qmodels.TokenizerType.WORD,
                            min_token_len=2,
                            max_token_len=20,
                            lowercase=True,
                        ),
                    )
                    logger.info(f"Created full-text index on {name}.{field}")
            except UnexpectedResponse as e:
                logger.warning(f"Collection '{name}' creation issue: {e}")

    # ── Point Operations ─────────────────────────────────────────────────

    def add_point(
        self,
        collection: str,
        vector: list[float],
        payload: dict,
        point_id: Optional[str] = None,
    ) -> str:
        """
        Upsert a single point into a collection.

        Args:
            collection: Collection name.
            vector: 1024-dim embedding vector.
            payload: Metadata dict (text, source, timestamp, etc.).
            point_id: Optional UUID string. Generated if not provided.

        Returns:
            The point UUID string.
        """
        if not point_id:
            point_id = str(uuid.uuid4())

        try:
            self.client.upsert(
                collection_name=collection,
                points=[
                    qmodels.PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=payload,
                    )
                ],
            )
            logger.debug(f"Added point {point_id} to '{collection}'")
            return point_id
        except Exception as e:
            logger.error(f"Failed to add point to '{collection}': {e}")
            raise

    def add_points_batch(
        self,
        collection: str,
        vectors: list[list[float]],
        payloads: list[dict],
        point_ids: Optional[list[str]] = None,
    ) -> list[str]:
        """Upsert multiple points at once."""
        if not point_ids:
            point_ids = [str(uuid.uuid4()) for _ in vectors]

        points = [
            qmodels.PointStruct(id=pid, vector=vec, payload=pay)
            for pid, vec, pay in zip(point_ids, vectors, payloads)
        ]

        try:
            self.client.upsert(collection_name=collection, points=points)
            logger.debug(f"Batch added {len(points)} points to '{collection}'")
            return point_ids
        except Exception as e:
            logger.error(f"Batch add to '{collection}' failed: {e}")
            raise

    # ── Search ───────────────────────────────────────────────────────────

    def search(
        self,
        collection: str,
        query_vector: list[float],
        query_text: Optional[str] = None,
        limit: Optional[int] = None,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[qmodels.Filter] = None,
    ) -> list[dict]:
        """
        Hybrid similarity search (Vector + Keyword).

        Args:
            collection: Collection name.
            query_vector: 1024-dim embedding vector.
            query_text: Optional raw text for keyword search.
            limit: Max results.
            score_threshold: Min score.
            filter_conditions: Optional Qdrant filter.

        Returns:
            List of dicts with 'id', 'score', and 'payload' keys.
        """
        limit = limit or settings.rag_max_results
        score_threshold = score_threshold or settings.rag_score_threshold

        try:
            # If query_text is provided, we perform a Hybrid query using prefetch.
            # Qdrant's query_points can combine multiple search signals.
            
            # 1. Vector prefetch
            prefetch_vector = qmodels.Prefetch(
                query=query_vector,
                limit=limit,
            )

            # 2. Keyword prefetch (if text provided)
            # We look for the text in 'text', 'content', or 'memory' fields
            prefetches = [prefetch_vector]
            
            if query_text:
                # We use a filter-based search for exact tokens/keywords
                # This ensures ".45" matches ".45" in the text
                keyword_filter = qmodels.Filter(
                    should=[
                        qmodels.FieldCondition(key="text", match=qmodels.MatchText(text=query_text)),
                        qmodels.FieldCondition(key="content", match=qmodels.MatchText(text=query_text)),
                        qmodels.FieldCondition(key="memory", match=qmodels.MatchText(text=query_text)),
                    ]
                )
                
                prefetch_keyword = qmodels.Prefetch(
                    filter=keyword_filter,
                    limit=limit,
                )
                prefetches.append(prefetch_keyword)

            # Perform the combined query
            results = self.client.query_points(
                collection_name=collection,
                prefetch=prefetches,
                query=query_vector, # Main query for ranking
                limit=limit,
                score_threshold=score_threshold,
                query_filter=filter_conditions,
            )

            hits = []
            for point in results.points:
                hits.append({
                    "id": str(point.id),
                    "score": point.score,
                    "payload": point.payload or {},
                })

            logger.debug(
                f"Hybrid Search '{collection}': {len(hits)} results "
                f"(query_text='{query_text}')"
            )
            return hits
        except Exception as e:
            logger.error(f"Search in '{collection}' failed: {e}")
            return []

    # ── Delete ───────────────────────────────────────────────────────────

    def delete_point(self, collection: str, point_id: str) -> bool:
        """Delete a single point by ID."""
        try:
            self.client.delete(
                collection_name=collection,
                points_selector=qmodels.PointIdsList(points=[point_id]),
            )
            logger.debug(f"Deleted point {point_id} from '{collection}'")
            return True
        except Exception as e:
            logger.error(f"Failed to delete point {point_id} from '{collection}': {e}")
            return False

    def delete_by_filter(self, collection: str, key: str, value: str) -> bool:
        """Delete points matching a payload filter."""
        try:
            self.client.delete(
                collection_name=collection,
                points_selector=qmodels.FilterSelector(
                    filter=qmodels.Filter(
                        must=[
                            qmodels.FieldCondition(
                                key=key,
                                match=qmodels.MatchValue(value=value),
                            )
                        ]
                    )
                ),
            )
            logger.debug(f"Deleted points where {key}={value} from '{collection}'")
            return True
        except Exception as e:
            logger.error(f"Filter delete from '{collection}' failed: {e}")
            return False

    # ── Info ─────────────────────────────────────────────────────────────

    def collection_info(self, collection: str) -> Optional[dict]:
        """Get collection stats."""
        try:
            info = self.client.get_collection(collection_name=collection)
            return {
                "name": collection,
                "points_count": info.points_count,
                "vectors_count": info.vectors_count,
                "status": str(info.status),
            }
        except Exception as e:
            logger.error(f"Failed to get info for '{collection}': {e}")
            return None


# Singleton instance
qdrant_store = QdrantStore()
