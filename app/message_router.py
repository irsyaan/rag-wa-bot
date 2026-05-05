"""
Message router..

Routes incoming WhatsApp messages to the appropriate handler:
- /admin -> admin_commands
- /remember, /forget, /memory -> memory_manager
- Everything else -> rag_engine

Logs all conversations to MySQL.
"""

import time
import re
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

        Flow:
        1. Check sender in MySQL users table.
        2. Auto-register unknown users as user.
        3. Ignore blocked users.
        4. If group message, only respond when bot is tagged.
        5. Route admin commands, memory commands, or RAG.
        6. Send reply.
        7. Log conversation to MySQL.
        """
        start_time = time.time()
        logger.info(
            f"Router started: sender_number={sender_number}, "
            f"chat_jid={chat_jid}, is_group={is_group}, text={text}"
        )

        text_stripped = text.strip()
        reply = ""
        rag_sources = None
        rag_result = None

        # Step 1: Check user permissions
        user = mysql_store.get_user(sender_number)

        if not user:
            mysql_store.add_user(sender_number, sender_name or "Unknown", "user")
            user = mysql_store.get_user(sender_number)
            logger.info(f"Auto-registered new user: {sender_number} ({sender_name})")

        role = user.get("role", "user") if user else "user"

        if not mysql_store.is_allowed(sender_number):
            logger.info(f"Blocked user attempted message: {sender_number}")
            return

        # Step 2: Group logic
        bot_jid = f"{settings.whatsapp_bot_number}@s.whatsapp.net"
        bot_number = settings.whatsapp_bot_number
        reply_chat_jid = chat_jid

        if is_group:
            mentioned_jids = mentioned_jids or []
            
            # Resolve mentioned JIDs to phone numbers (to handle @lid mentions)
            resolved_mentions = [
                whatsapp_client._resolve_phone_number(jid) 
                for jid in mentioned_jids
            ]

            is_mentioned = (
                bot_jid in mentioned_jids or 
                bot_number in resolved_mentions or
                (whatsapp_client.bot_jid and whatsapp_client.bot_jid in mentioned_jids)
            )

            if not is_mentioned:
                logger.debug(f"Ignoring group message because bot was not mentioned. Mentions: {mentioned_jids}, Resolved: {resolved_mentions}")
                return

            # If tagged in group, reply in the group by default
            # (unless it's an admin command, which we will handle below)
            logger.info(f"Bot tagged in group, processing message from {sender_number}")

        # Step 3: Route message
        elif text_stripped.lower() == "/help":
            help_lines = ["📋 *Available Commands*\n"]
            
            help_lines.append("*Memory Commands:*")
            help_lines.append("/memory search <query> — Search memories")
            
            if role in ["admin", "owner"]:
                help_lines.append("/remember <fact> — Save a memory")
                help_lines.append("/forget <keyword> — Delete a memory")
                
                help_lines.append("\n*Admin Commands:*")
                help_lines.append("/admin listusers — List all registered users")
                help_lines.append("/admin adduser <number> <name> <role> — Add a user")
                help_lines.append("/admin setrole <number> <role> — Change user role")
                help_lines.append("/admin blockuser <number> — Block a user")
                help_lines.append("/admin unblockuser <number> — Unblock a user")
                
            reply = "\n".join(help_lines)

        elif admin_commands.is_admin_command(text_stripped):
            result = admin_commands.execute(text_stripped, sender_number)
            
            # If denied due to role, reply in the same chat (group or private) with translated message
            if not result.success and "admin permissions" in result.message:
                reply = rag_engine.get_rejection_message(text_stripped)
                logger.info(f"Admin command denied for {sender_number}, replied with: {reply}")
            else:
                # If allowed (or normal error), force private reply to protect sensitive data
                if is_group:
                    reply_chat_jid = f"{sender_number}@s.whatsapp.net"
                    logger.info(f"Admin command used in group, replying personally to {sender_number}")
                reply = result.message

        elif memory_manager.is_memory_command(text_stripped):
            # Restrict /remember and /forget to admin/owner
            is_write_cmd = text_stripped.lower().startswith(("/remember", "/forget"))
            
            if is_write_cmd and role not in ["admin", "owner"]:
                reply = rag_engine.get_rejection_message(text_stripped)
            else:
                result = memory_manager.handle_command(text_stripped, sender_number)
                reply = result.message

        elif rag_engine.classify_message(text_stripped) == "greeting":
            reply = rag_engine.generate_greeting(text_stripped, sender_name=sender_name)
            logger.info(f"Greeting detected, replied with: {reply[:80]}")

        else:
            rag_result = rag_engine.answer(
                question=text_stripped,
                sender_number=sender_number,
                chat_jid=chat_jid,
                role=role,
            )
            reply = rag_result.answer

            if hasattr(rag_result, "sources"):
                rag_sources = rag_result.sources

        # Step 4: Send reply
        if reply:
            whatsapp_client.send_reply(reply_chat_jid, reply)
        else:
            logger.warning("No reply generated for message")

        # Step 5: Log conversation
        elapsed_ms = int((time.time() - start_time) * 1000)

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

        logger.info(f"Router finished in {elapsed_ms} ms")

        logger.info(f"Router finished in {elapsed_ms} ms")


# Singleton instance
message_router = MessageRouter()
