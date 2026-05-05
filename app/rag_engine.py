"""
RAG (Retrieval-Augmented Generation) engine.

Orchestrates the full pipeline:
1. Embed the user's question
2. Search Qdrant collections for relevant context
3. Build prompt with context
4. Call Ollama for the answer
5. Log failed questions if no good context found
"""

import time
from typing import Optional
from dataclasses import dataclass, field

from loguru import logger

from app.config import settings
from app.ollama_client import ollama_client
from app.qdrant_store import qdrant_store
from app.mysql_store import mysql_store
from app.prompts import (
    RAG_SYSTEM_PROMPT,
    CLASSIFY_SYSTEM_PROMPT,
    GREETING_SYSTEM_PROMPT,
    NO_CONTEXT_REPLY,
)


@dataclass
class RAGResult:
    """Result from the RAG pipeline."""
    answer: str
    sources: list[dict] = field(default_factory=list)
    best_score: float = 0.0
    intent: str = "question"
    response_time_ms: int = 0
    has_context: bool = False


class RAGEngine:
    """RAG answer pipeline using Qdrant + Ollama."""

    def __init__(self):
        self.max_results = settings.rag_max_results
        self.score_threshold = settings.rag_score_threshold

    def classify_intent(self, text: str) -> str:
        """Classify the user's message intent using the fast model."""
        result = ollama_client.classify(text, CLASSIFY_SYSTEM_PROMPT)
        intent = result.strip().lower()

        valid_intents = {"greeting", "question", "chitchat", "unclear"}
        if intent not in valid_intents:
            intent = "question"  # Default to question if unclear

        logger.debug(f"Intent classified: '{intent}' for: {text[:50]}...")
        return intent

    def search_context(self, query_vector: list[float]) -> tuple[list[dict], float]:
        """
        Search all relevant Qdrant collections for context.

        Returns:
            Tuple of (results list, best score).
        """
        all_results = []
        best_score = 0.0

        # Search knowledge base
        knowledge_hits = qdrant_store.search(
            collection=settings.qdrant_knowledge_collection,
            query_vector=query_vector,
            limit=self.max_results,
            score_threshold=self.score_threshold,
        )
        all_results.extend(knowledge_hits)

        # Search conversation memory
        memory_hits = qdrant_store.search(
            collection=settings.qdrant_chat_collection,
            query_vector=query_vector,
            limit=3,
            score_threshold=self.score_threshold,
        )
        all_results.extend(memory_hits)

        if all_results:
            best_score = max(r["score"] for r in all_results)

        return all_results, best_score

    def search_personal_memory(self, query_vector: list[float]) -> list[dict]:
        """Search the personal memory collection for user-stored facts."""
        return qdrant_store.search(
            collection=settings.qdrant_memory_collection,
            query_vector=query_vector,
            limit=3,
            score_threshold=self.score_threshold,
        )

    def build_context_string(self, results: list[dict]) -> str:
        """Format search results into a context string for the prompt."""
        if not results:
            return "No relevant context found."

        parts = []
        for i, r in enumerate(results, 1):
            payload = r.get("payload", {})
            text = payload.get("text", payload.get("content", ""))
            source = payload.get("source", "unknown")
            score = r.get("score", 0)
            parts.append(f"[{i}] (score: {score:.2f}, source: {source})\n{text}")

        return "\n\n".join(parts)

    def answer(
        self,
        question: str,
        sender_number: Optional[str] = None,
        chat_jid: Optional[str] = None,
    ) -> RAGResult:
        """
        Full RAG pipeline: classify → embed → search → generate → log.

        Args:
            question: The user's message text.
            sender_number: For logging failed questions.
            chat_jid: For logging.

        Returns:
            RAGResult with the answer, sources, and metadata.
        """
        start_time = time.time()

        # Step 1: Classify intent
        intent = self.classify_intent(question)

        # Handle greetings without RAG
        if intent == "greeting":
            reply = ollama_client.chat(
                messages=[{"role": "user", "content": question}],
                system=GREETING_SYSTEM_PROMPT,
                temperature=0.8,
            )
            elapsed = int((time.time() - start_time) * 1000)
            return RAGResult(
                answer=reply or "Halo! Ada yang bisa saya bantu?",
                intent="greeting",
                response_time_ms=elapsed,
                has_context=False,
            )

        # Step 2: Embed the question
        query_vector = ollama_client.embed(question)
        if not query_vector:
            elapsed = int((time.time() - start_time) * 1000)
            return RAGResult(
                answer="Maaf, terjadi kesalahan saat memproses pertanyaan.",
                response_time_ms=elapsed,
            )

        # Step 3: Search context
        context_results, best_score = self.search_context(query_vector)
        memory_results = self.search_personal_memory(query_vector)

        # Step 4: Build prompt and generate
        has_context = len(context_results) > 0 and best_score >= self.score_threshold

        if has_context or intent == "chitchat":
            context_str = self.build_context_string(context_results)
            memory_str = self.build_context_string(memory_results) if memory_results else "No personal memories found."

            system_prompt = RAG_SYSTEM_PROMPT.format(
                context=context_str,
                memory_context=memory_str,
            )

            reply = ollama_client.chat(
                messages=[{"role": "user", "content": question}],
                system=system_prompt,
            )
        else:
            reply = ""

        # Step 5: Handle no-answer case
        if not reply or not has_context:
            reply = reply or NO_CONTEXT_REPLY

            # Log the failed question
            if sender_number:
                sources_info = [
                    {"collection": r.get("payload", {}).get("source", "unknown"), "score": r.get("score", 0)}
                    for r in context_results
                ]
                mysql_store.log_failed_question(
                    sender_number=sender_number,
                    question_text=question,
                    chat_jid=chat_jid,
                    attempted_sources=sources_info,
                    best_score=best_score if best_score > 0 else None,
                )
                logger.info(f"Logged failed question from {sender_number}: {question[:50]}...")

        elapsed = int((time.time() - start_time) * 1000)

        sources = [
            {
                "id": r["id"],
                "score": r["score"],
                "source": r.get("payload", {}).get("source", "unknown"),
            }
            for r in context_results
        ]

        return RAGResult(
            answer=reply,
            sources=sources,
            best_score=best_score,
            intent=intent,
            response_time_ms=elapsed,
            has_context=has_context,
        )


# Singleton instance
rag_engine = RAGEngine()
