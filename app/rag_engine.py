"""
RAG engine.

Flow:
1. Embed the user question using Ollama bge-m3.
2. Search Qdrant collections:
   - personal_memory
   - personal_knowledge
   - conversation_memory
3. Keep results above collection-specific score thresholds.
4. Build context.
5. Ask Ollama qwen3-8b-rag using only the retrieved context.
6. Return answer, sources, and response time.
"""

import time
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests
from loguru import logger

from app.config import settings
from app.mysql_store import mysql_store
from app.prompts import RAG_SYSTEM_PROMPT, CLASSIFY_SYSTEM_PROMPT, GREETING_SYSTEM_PROMPT


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
        self.default_score_threshold = float(settings.rag_score_threshold)

        self.memory_collection = settings.qdrant_memory_collection
        self.knowledge_collection = settings.qdrant_knowledge_collection
        self.chat_collection = settings.qdrant_chat_collection

        self.collections = [
            self.memory_collection,
            self.knowledge_collection,
            self.chat_collection,
        ]

        # Different threshold per collection.
        # Personal memory should be more flexible because users often ask casually/slang.
        # Document knowledge should stay stricter.
        self.collection_thresholds = {
            self.memory_collection: 0.45,
            self.knowledge_collection: self.default_score_threshold,
            self.chat_collection: 0.55,
        }

        # Important:
        # In proxy mode, global HTTP_PROXY/HTTPS_PROXY exists for WhatsApp.
        # For local Ollama/Qdrant calls, do not use proxy.
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
        Support both Qdrant response shapes.

        Current query API:
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

    def _threshold_for_collection(self, collection: str) -> float:
        """Get score threshold for a specific collection."""
        return self.collection_thresholds.get(collection, self.default_score_threshold)

    def _search_collection(self, collection: str, vector: List[float]) -> List[RagSource]:
        """Search one Qdrant collection."""
        threshold = self._threshold_for_collection(collection)

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

                if score < threshold:
                    logger.debug(
                        f"Skipping low-score result from {collection}: "
                        f"score={score:.4f}, threshold={threshold:.4f}"
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

            logger.info(
                f"Qdrant search {collection}: {len(sources)} usable results "
                f"(threshold={threshold:.2f})"
            )

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

    def _ask_ollama(self, question: str, context: str, role: str) -> str:
        """Ask Ollama using retrieved context only."""
        # Use centralized prompt from prompts.py
        prompt = RAG_SYSTEM_PROMPT.format(
            role=role,
            context=context or "No specific context found.",
            memory_context="",
        )
        
        # Add the actual question as the final message
        prompt += f"\n\nQuestion:\n{question}\n\nAnswer:"

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

    def get_rejection_message(self, text: str) -> str:
        """Use the fast model to generate a translated rejection message."""
        prompt = f"Translate the exact phrase 'You don't have the permission to do this.' into the language used in the following text. Reply ONLY with the translated phrase, no quotes or extra words.\n\nText: {text}"
        try:
            data = self._post_json(
                f"{self.ollama_url}/api/chat",
                {
                    "model": self.fast_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                },
                timeout=30,
            )
            message = data.get("message", {}).get("content", "").strip()
            return message if message else "You don't have the permission to do this."
        except Exception as e:
            logger.error(f"Failed to translate rejection message: {e}")
            return "You don't have the permission to do this."

    # Minimal fallback keyword set used ONLY when LLM classification fails
    _GREETING_KEYWORDS = {
        "hi", "hii", "hiii", "hello", "helo", "halo", "hai", "hey", "heey",
        "p", "yo", "oi", "woi", "hei",
    }

    def _is_greeting_fallback(self, text: str) -> bool:
        """Simple keyword check used as fallback when LLM classification fails."""
        words = text.strip().lower().split()
        return len(words) <= 3 and bool(self._GREETING_KEYWORDS.intersection(words))

    def classify_message(self, text: str) -> str:
        """Use the fast model to classify the user's intent."""
        prompt = f"{CLASSIFY_SYSTEM_PROMPT}\n\nMessage: {text}"
        try:
            data = self._post_json(
                f"{self.ollama_url}/api/chat",
                {
                    "model": self.fast_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                },
                timeout=30,  # increased from 10s
            )
            message = data.get("message", {}).get("content", "").strip().lower()

            for category in ["greeting", "question", "command", "chitchat", "unclear"]:
                if category in message:
                    return category

            return "unclear"
        except Exception as e:
            logger.warning(f"LLM classification failed, using keyword fallback: {e}")
            # Fallback: check simple keywords so greetings still work
            if self._is_greeting_fallback(text):
                return "greeting"
            return "unclear"

    def generate_greeting(self, text: str, sender_name: str = "User") -> str:
        """Generate a time-aware greeting response using the fast model."""
        current_time = datetime.now().strftime("%H:%M")
        prompt = GREETING_SYSTEM_PROMPT.format(
            current_time=current_time,
            text=text,
            sender_name=sender_name or "User",
        )
        try:
            data = self._post_json(
                f"{self.ollama_url}/api/chat",
                {
                    "model": self.fast_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                },
                timeout=30,
            )
            message = data.get("message", {}).get("content", "").strip()
            return message if message else "Hello! How can I help you?"
        except Exception as e:
            logger.error(f"Failed to generate greeting: {e}")
            return "Hello! How can I help you?"

    def answer(self, question: str, sender_number: str, chat_jid: str, role: str = "user") -> RagResult:
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

                logger.info(f"RAG found no usable sources. Finished in {elapsed_ms} ms")

                return RagResult(
                    answer=(
                        "Maaf, saya belum memiliki informasi yang cukup untuk menjawab pertanyaan ini. "
                        "Pertanyaan ini sudah dicatat untuk ditinjau nanti."
                    ),
                    sources=[],
                    response_time_ms=elapsed_ms,
                )

            logger.info(
                f"RAG found {len(sources)} source(s). "
                f"Best score={sources[0].score:.4f}, collection={sources[0].collection}"
            )

            context = self._build_context(sources)
            answer = self._ask_ollama(question, context, role)

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