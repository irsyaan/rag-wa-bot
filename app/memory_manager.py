"""
Memory manager.

Handles /remember, /forget, and /memory search commands.
Stores durable user memories in the personal_memory Qdrant collection.
"""

import time
from datetime import datetime, timezone
from typing import Optional
from dataclasses import dataclass

from loguru import logger

from app.config import settings
from app.ollama_client import ollama_client
from app.qdrant_store import qdrant_store


@dataclass
class MemoryResult:
    """Result from a memory operation."""
    success: bool
    message: str


class MemoryManager:
    """Manages personal memory storage and retrieval in Qdrant."""

    def __init__(self):
        self.collection = settings.qdrant_memory_collection

    def is_memory_command(self, text: str) -> bool:
        """Check if the text is a memory command."""
        lower = text.strip().lower()
        return (
            lower.startswith("/remember")
            or lower.startswith("/forget")
            or lower.startswith("/memory")
        )

    def handle_command(self, text: str, sender_number: str) -> MemoryResult:
        """
        Route and execute a memory command.

        Args:
            text: Full message text.
            sender_number: The sender's WhatsApp number.

        Returns:
            MemoryResult with the outcome.
        """
        lower = text.strip().lower()

        if lower.startswith("/remember"):
            fact = text.strip()[len("/remember"):].strip()
            return self.remember(fact, sender_number)

        elif lower.startswith("/memory search"):
            query = text.strip()[len("/memory search"):].strip()
            return self.search(query, sender_number)

        elif lower.startswith("/forget"):
            keyword = text.strip()[len("/forget"):].strip()
            return self.forget(keyword, sender_number)

        return MemoryResult(success=False, message="❌ Unknown memory command.")

    def remember(self, fact: str, sender_number: str) -> MemoryResult:
        """
        Store a fact in personal memory.

        Args:
            fact: The text to remember.
            sender_number: Who stored it.
        """
        if not fact:
            return MemoryResult(
                success=False,
                message="❌ Usage: /remember <fact to remember>",
            )

        # Embed the fact
        vector = ollama_client.embed(fact)
        if not vector:
            return MemoryResult(
                success=False,
                message="❌ Failed to process the memory. Try again.",
            )

        # Store in Qdrant
        payload = {
            "text": fact,
            "source": "user_memory",
            "stored_by": sender_number,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": "personal_memory",
        }

        try:
            point_id = qdrant_store.add_point(
                collection=self.collection,
                vector=vector,
                payload=payload,
            )
            logger.info(f"Memory stored: {point_id} by {sender_number}: {fact[:50]}...")
            return MemoryResult(
                success=True,
                message=f"✅ Remembered!\n\n📝 _{fact}_\n\nID: `{point_id[:8]}...`",
            )
        except Exception as e:
            logger.error(f"Failed to store memory: {e}")
            return MemoryResult(
                success=False,
                message="❌ Failed to save the memory. Try again.",
            )

    def search(self, query: str, sender_number: str) -> MemoryResult:
        """
        Search personal memories.

        Args:
            query: Search query text.
            sender_number: Who is searching.
        """
        if not query:
            return MemoryResult(
                success=False,
                message="❌ Usage: /memory search <query>",
            )

        # Embed the query
        vector = ollama_client.embed(query)
        if not vector:
            return MemoryResult(
                success=False,
                message="❌ Failed to process the search. Try again.",
            )

        # Search Qdrant
        results = qdrant_store.search(
            collection=self.collection,
            query_vector=vector,
            limit=5,
            score_threshold=0.5,  # Lower threshold for memory search
        )

        if not results:
            return MemoryResult(
                success=True,
                message="🔍 No matching memories found.",
            )

        lines = ["🔍 *Matching Memories:*\n"]
        for i, r in enumerate(results, 1):
            payload = r.get("payload", {})
            text = payload.get("text", "—")
            score = r.get("score", 0)
            point_id = r.get("id", "—")
            timestamp = payload.get("timestamp", "—")

            # Format timestamp
            if timestamp != "—":
                try:
                    dt = datetime.fromisoformat(timestamp)
                    timestamp = dt.strftime("%Y-%m-%d %H:%M")
                except (ValueError, TypeError):
                    pass

            lines.append(
                f"*{i}.* (score: {score:.2f})\n"
                f"   📝 {text}\n"
                f"   📅 {timestamp}\n"
                f"   🆔 `{str(point_id)[:8]}...`"
            )

        return MemoryResult(success=True, message="\n\n".join(lines))

    def forget(self, keyword: str, sender_number: str) -> MemoryResult:
        """
        Delete a memory by ID prefix or by keyword search.

        Args:
            keyword: Memory ID prefix or search keyword.
            sender_number: Who is deleting.
        """
        if not keyword:
            return MemoryResult(
                success=False,
                message="❌ Usage: /forget <memory_id or keyword>",
            )

        # First try to find by keyword search
        vector = ollama_client.embed(keyword)
        if not vector:
            return MemoryResult(
                success=False,
                message="❌ Failed to process. Try again.",
            )

        results = qdrant_store.search(
            collection=self.collection,
            query_vector=vector,
            limit=1,
            score_threshold=0.7,  # Higher threshold for deletion
        )

        if not results:
            return MemoryResult(
                success=True,
                message="🔍 No matching memory found to forget.",
            )

        # Delete the best match
        target = results[0]
        point_id = target["id"]
        text = target.get("payload", {}).get("text", "—")

        success = qdrant_store.delete_point(self.collection, point_id)
        if success:
            logger.info(f"Memory forgotten by {sender_number}: {point_id}")
            return MemoryResult(
                success=True,
                message=f"🗑️ Forgotten!\n\n📝 _{text}_",
            )
        else:
            return MemoryResult(
                success=False,
                message="❌ Failed to delete the memory. Try again.",
            )

    def store_conversation_summary(
        self,
        summary: str,
        sender_number: str,
        chat_jid: str,
    ) -> Optional[str]:
        """
        Store a conversation summary in the conversation_memory collection.
        Used by the RAG engine for long-term conversational context.

        Returns:
            The point ID if successful, None otherwise.
        """
        vector = ollama_client.embed(summary)
        if not vector:
            return None

        payload = {
            "text": summary,
            "source": "conversation_summary",
            "sender": sender_number,
            "chat_jid": chat_jid,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": "conversation_memory",
        }

        try:
            point_id = qdrant_store.add_point(
                collection=settings.qdrant_chat_collection,
                vector=vector,
                payload=payload,
            )
            return point_id
        except Exception as e:
            logger.error(f"Failed to store conversation summary: {e}")
            return None


# Singleton instance
memory_manager = MemoryManager()
