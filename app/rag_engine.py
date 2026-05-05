"""
RAG engine.

Flow:
1. Embed the user question using Ollama bge-m3.
2. Search Qdrant collections:
   - personal_memory
   - personal_knowledge
   - conversation_memory
3. Keep results above score threshold.
4. Build context.
5. Ask Ollama qwen3-8b-rag using only the retrieved context.
6. Return answer, sources, and response time.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests
from loguru import logger

from app.config import settings
from app.mysql_store import mysql_store


@dataclass
class RagSource:
    collection: str
    point_id: str
    score: float
    text: str
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RagResult:
    answer: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    response_time_ms: int = 0


class RagEngine:
    """RAG search and answer engine."""

    def __init__(self):
        self.ollama_url = settings.ollama_base_url.rstrip("/")
        self.qdrant_url = settings.qdrant_url.rstrip("/")
        self.embedding_model = settings.ollama_embedding_model
        self.main_model = settings.ollama_main_model
        self.fast_model = settings.ollama_fast_model
        self.max_results = int(settings.rag_max_results)
        self.score_threshold = float(settings.rag_score_threshold)

        self.collections = [
            settings.qdrant_memory_collection,
            settings.qdrant_knowledge_collection,
            settings.qdrant_chat_collection,
        ]

        # Important:
        # In proxy mode, global HTTP_PROXY/HTTPS_PROXY exists for WhatsApp.
        # For local Ollama/Qdrant calls, we do not want requests to use the proxy.
        self.http = requests.Session()
        self.http.trust_env = False

    def _post_json(self, url: str, payload: Dict[str, Any], timeout: int = 120) -> Dict[str, Any]:
        response = self.http.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()

    def _embed(self, text: str) -> List[float]:
        """Create embedding using Ollama."""
        data = self._post_json(
            f"{self.ollama_url}/api/embed",
            {
                "model": self.embedding_model,
                "input": text,
            },
            timeout=120,
        )

        embeddings = data.get("embeddings") or []
        if not embeddings:
            raise RuntimeError("Ollama embedding response did not contain embeddings")

        return embeddings[0]

    def _extract_points(self, qdrant_response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Support both Qdrant response shapes:

        New query API:
        {
          "result": {
            "points": [...]
          }
        }

        Older/search style:
        {
          "result": [...]
        }
        """
        result = qdrant_response.get("result")

        if isinstance(result, dict):
            points = result.get("points", [])
            if isinstance(points, list):
                return points

        if isinstance(result, list):
            return result

        return []

    def _payload_text(self, payload: Dict[str, Any]) -> str:
        """Extract text from possible payload keys."""
        for key in ["text", "content", "chunk_text", "memory", "document_text"]:
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    def _search_collection(self, collection: str, vector: List[float]) -> List[RagSource]:
        """Search one Qdrant collection."""
        try:
            data = self._post_json(
                f"{self.qdrant_url}/collections/{collection}/points/query",
                {
                    "query": vector,
                    "limit": self.max_results,
                    "with_payload": True,
                },
                timeout=60,
            )

            points = self._extract_points(data)
            sources: List[RagSource] = []

            for point in points:
                score = float(point.get("score", 0.0))
                payload = point.get("payload") or {}
                text = self._payload_text(payload)

                if not text:
                    continue

                if score < self.score_threshold:
                    logger.debug(
                        f"Skipping low-score result from {collection}: "
                        f"score={score}, threshold={self.score_threshold}"
                    )
                    continue

                sources.append(
                    RagSource(
                        collection=collection,
                        point_id=str(point.get("id", "")),
                        score=score,
                        text=text,
                        payload=payload,
                    )
                )

            logger.info(f"Qdrant search {collection}: {len(sources)} usable results")
            return sources

        except Exception as e:
            logger.error(f"Qdrant search failed for collection {collection}: {e}")
            return []

    def _search_all(self, question: str) -> List[RagSource]:
        """Embed question and search all configured Qdrant collections."""
        vector = self._embed(question)

        all_sources: List[RagSource] = []

        for collection in self.collections:
            results = self._search_collection(collection, vector)
            all_sources.extend(results)

        all_sources.sort(key=lambda item: item.score, reverse=True)
        return all_sources[: self.max_results]

    def _build_context(self, sources: List[RagSource]) -> str:
        """Build context text for the LLM."""
        blocks = []

        for idx, source in enumerate(sources, start=1):
            source_name = (
                source.payload.get("source")
                or source.payload.get("filename")
                or source.payload.get("source_name")
                or source.collection
            )

            blocks.append(
                f"[Source {idx}]\n"
                f"Collection: {source.collection}\n"
                f"Source: {source_name}\n"
                f"Score: {source.score:.4f}\n"
                f"Text:\n{source.text}"
            )

        return "\n\n---\n\n".join(blocks)

    def _ask_ollama(self, question: str, context: str) -> str:
        """Ask Ollama using retrieved context only."""
        prompt = f"""
You are a personal WhatsApp RAG assistant.

Answer the user question using only the context below.
If the answer is in the context, answer directly and briefly.
If the answer is not in the context, say:
Maaf, saya belum memiliki informasi yang cukup untuk menjawab pertanyaan ini.

Use Indonesian if the question is Indonesian.
Use English if the question is English.
Do not show reasoning or thinking.

Context:
{context}

Question:
{question}

Answer:
""".strip()

        data = self._post_json(
            f"{self.ollama_url}/api/chat",
            {
                "model": self.main_model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "think": False,
                "stream": False,
            },
            timeout=180,
        )

        message = data.get("message") or {}
        answer = message.get("content", "").strip()

        if not answer:
            return "Maaf, saya belum memiliki informasi yang cukup untuk menjawab pertanyaan ini."

        return answer

    def _log_failed_question(
        self,
        sender_number: str,
        chat_jid: str,
        question: str,
        best_score: Optional[float],
    ) -> None:
        """Log failed question if mysql_store supports it."""
        try:
            if hasattr(mysql_store, "log_failed_question"):
                mysql_store.log_failed_question(
                    sender_number=sender_number,
                    question_text=question,
                    chat_jid=chat_jid,
                    attempted_sources=None,
                    best_score=best_score,
                )
                return

            if hasattr(mysql_store, "add_failed_question"):
                mysql_store.add_failed_question(
                    sender_number=sender_number,
                    question_text=question,
                    chat_jid=chat_jid,
                    best_score=best_score,
                )
                return

            logger.debug("mysql_store has no failed-question method; skipping failed question log")

        except Exception as e:
            logger.warning(f"Could not log failed question: {e}")

    def answer(self, question: str, sender_number: str, chat_jid: str) -> RagResult:
        """Main RAG answer function."""
        start_time = time.time()
        question = question.strip()

        logger.info(f"RAG started for {sender_number}: {question}")

        try:
            sources = self._search_all(question)

            if not sources:
                elapsed_ms = int((time.time() - start_time) * 1000)
                self._log_failed_question(
                    sender_number=sender_number,
                    chat_jid=chat_jid,
                    question=question,
                    best_score=None,
                )

                return RagResult(
                    answer=(
                        "Maaf, saya belum memiliki informasi yang cukup untuk menjawab pertanyaan ini. "
                        "Pertanyaan ini sudah dicatat untuk ditinjau nanti."
                    ),
                    sources=[],
                    response_time_ms=elapsed_ms,
                )

            best_score = sources[0].score
            logger.info(
                f"RAG found {len(sources)} source(s). "
                f"Best score={best_score:.4f}, collection={sources[0].collection}"
            )

            context = self._build_context(sources)
            answer = self._ask_ollama(question, context)

            source_payloads = [
                {
                    "collection": source.collection,
                    "point_id": source.point_id,
                    "score": source.score,
                    "source": source.payload.get("source")
                    or source.payload.get("filename")
                    or source.payload.get("source_name")
                    or source.collection,
                    "text": source.text[:500],
                }
                for source in sources
            ]

            elapsed_ms = int((time.time() - start_time) * 1000)

            logger.info(f"RAG finished in {elapsed_ms} ms")

            return RagResult(
                answer=answer,
                sources=source_payloads,
                response_time_ms=elapsed_ms,
            )

        except Exception as e:
            logger.exception(f"RAG answer failed: {e}")

            elapsed_ms = int((time.time() - start_time) * 1000)

            return RagResult(
                answer="Maaf, terjadi error saat memproses pertanyaan.",
                sources=[],
                response_time_ms=elapsed_ms,
            )


rag_engine = RagEngine()