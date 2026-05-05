"""
Message router..

Routes incoming WhatsApp messages to the appropriate handler:
- /admin -> admin_commands
- /remember, /forget, /memory -> memory_manager
- Everything else -> rag_engine

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
from app.document_processor import document_processor
from app.qdrant_store import qdrant_store


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
        document_msg=None,
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

            # If tagged in group, strip the mentions to get the clean command/question
            # We remove both @number and @name tags if possible
            text_stripped = text
            for jid in mentioned_jids:
                if jid in [bot_jid, whatsapp_client.bot_jid]:
                    # Remove the @jid part (neonize often passes it as @number in the text)
                    prefix = jid.split('@')[0]
                    text_stripped = text_stripped.replace(f"@{prefix}", "").strip()
            
            # Fallback: if there are still any @ mention patterns, clean them up
            import re
            text_stripped = re.sub(r'@[0-9]+', '', text_stripped).strip()

            logger.info(f"Bot tagged in group, processing cleaned message: '{text_stripped}'")

        # Step 3: Route message
        if text_stripped.lower() == "/help":
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

        elif text_stripped.lower().startswith("/forgetdoc "):
            if role not in ["admin", "owner"]:
                reply = rag_engine.get_rejection_message(text_stripped)
            else:
                filename = text_stripped[11:].strip()
                deleted = qdrant_store.delete_by_filter(
                    settings.qdrant_knowledge_collection, "filename", filename
                )
                if deleted:
                    reply = f"✅ Deleted all knowledge chunks for document '{filename}' from vector database."
                else:
                    reply = f"❌ Failed to delete document '{filename}' or it was not found."

        elif document_msg:
            # Handle PDF uploads
            if is_group:
                reply = "⚠️ Document uploads are only supported in private chats."
            elif role not in ["admin", "owner"]:
                reply = "⚠️ You do not have permission to upload documents to the knowledge base."
            elif document_msg.mimetype != "application/pdf":
                reply = "⚠️ Only PDF documents are supported for upload."
            else:
                # Tell the user we're processing
                whatsapp_client.send_reply(reply_chat_jid, f"⏳ Downloading and processing '{document_msg.fileName}'...")
                
                # Download
                try:
                    import os
                    media_bytes = whatsapp_client.client.download_any(raw_message.Message)
                    
                    filename = document_msg.fileName or "uploaded.pdf"
                    file_path = os.path.join(settings.rag_upload_dir, filename)
                    
                    # Ensure directory exists
                    os.makedirs(settings.rag_upload_dir, exist_ok=True)
                    
                    with open(file_path, "wb") as f:
                        f.write(media_bytes)
                    
                    # Process
                    success, msg = document_processor.process_pdf(file_path, filename, sender_number)
                    reply = msg
                except Exception as e:
                    logger.exception(f"Failed to download/process PDF: {e}")
                    reply = f"❌ Failed to process document: {str(e)}"

        else:
            # Send everything to RAG — the main model handles greetings too
            rag_result = rag_engine.answer(
                question=text_stripped,
                sender_number=sender_number,
                chat_jid=chat_jid,
                role=role,
                sender_name=sender_name,
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
