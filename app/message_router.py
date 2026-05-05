"""
Message router.

Routes incoming WhatsApp messages to the appropriate handler:
- /admin → admin_commands
- /remember, /forget, /memory → memory_manager
- Everything else → rag_engine

Logs all conversations to MySQL.
"""

import time
from typing import Optional

from loguru import logger

from app.config import settings
from app.mysql_store import mysql_store
from app.admin_commands import admin_commands
from app.memory_manager import memory_manager
from app.rag_engine import rag_engine
from app.whatsapp_client import whatsapp_client


class MessageRouter:
    """Routes incoming messages to appropriate handlers."""

    def handle_message(
        self,
        sender_number: str,
        sender_name: Optional[str],
        chat_jid: str,
        text: str,
        is_group: bool,
        mentioned_jids: list[str] = None,
        raw_message=None,
    ) -> None:
        """
        Main message handler called by the WhatsApp client.

        Flow per spec Section 8:
        1. Extract sender info (done by WhatsApp client)
        2. Ignore self-messages (done by WhatsApp client)
        3. Check sender in MySQL users table
        4. If not allowed, ignore or reply with rejection
        5. Route to /admin, /remember, /forget, /memory, or RAG
        6. Reply via WhatsApp
        7. Log conversation to MySQL
        """
        start_time = time.time()

        # ── Step 1: Check user permissions ───────────────────────────────
        user = mysql_store.get_user(sender_number)

        if not user:
            # Auto-register unknown users with 'user' role
            # The owner can later block them if needed
            mysql_store.add_user(sender_number, sender_name or "Unknown", "user")
            user = mysql_store.get_user(sender_number)
            logger.info(f"Auto-registered new user: {sender_number} ({sender_name})")

        if not mysql_store.is_allowed(sender_number):
            logger.info(f"Blocked user attempted message: {sender_number}")
            return  # Silently ignore blocked users

        # ── Step 1.5: Handle Group Logic ─────────────────────────────────
        bot_jid = f"{settings.whatsapp_bot_number}@s.whatsapp.net"
        reply_chat_jid = chat_jid  # Default: reply to the chat where msg came from

        if is_group:
            mentioned_jids = mentioned_jids or []
            if bot_jid not in mentioned_jids:
                return  # Ignore group messages where bot is not tagged
            
            # If tagged in group, reply via personal chat (DM) instead
            reply_chat_jid = f"{sender_number}@s.whatsapp.net"
            logger.info(f"Bot tagged in group, replying personally to {sender_number}")

        # ── Step 2: Route the message ────────────────────────────────────
        reply = ""
        text_stripped = text.strip()

        if admin_commands.is_admin_command(text_stripped):
            # Admin commands
            result = admin_commands.execute(text_stripped, sender_number)
            reply = result.message

        elif memory_manager.is_memory_command(text_stripped):
            # Memory commands
            result = memory_manager.handle_command(text_stripped, sender_number)
            reply = result.message

        else:
            # RAG pipeline
            rag_result = rag_engine.answer(
                question=text_stripped,
                sender_number=sender_number,
                chat_jid=chat_jid,
            )
            reply = rag_result.answer

            # Log response time from RAG
            elapsed_ms = rag_result.response_time_ms

        # ── Step 3: Send reply ───────────────────────────────────────────
        if reply:
            whatsapp_client.send_reply(reply_chat_jid, reply)

        # ── Step 4: Log conversation ─────────────────────────────────────
        elapsed_ms = int((time.time() - start_time) * 1000)

        rag_sources = None
        if not admin_commands.is_admin_command(text_stripped) and not memory_manager.is_memory_command(text_stripped):
            # Only log RAG sources for regular messages
            if hasattr(rag_result, "sources"):
                rag_sources = rag_result.sources

        mysql_store.log_conversation(
            sender_number=sender_number,
            sender_name=sender_name,
            chat_jid=chat_jid,
            message_text=text,
            bot_reply=reply,
            is_group=is_group,
            rag_sources=rag_sources,
            response_time_ms=elapsed_ms,
        )


# Singleton instance
message_router = MessageRouter()
