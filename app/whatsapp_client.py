"""
WhatsApp client using neonize..

Handles connection, message receiving, QR code display,
and sending replies. Session is persisted to avoid re-scanning QR.
"""

from pathlib import Path
from typing import Callable, Optional
import sqlite3

import segno
from loguru import logger

from neonize.client import NewClient
from neonize.events import (
    MessageEv,
    ConnectedEv,
    PairStatusEv,
)
from neonize.proto.Neonize_pb2 import Message
from neonize.utils.jid import JIDToNonAD

from app.config import settings


class WhatsAppClient:
    """Neonize WhatsApp client wrapper."""

    def __init__(self):
        self.client: Optional[NewClient] = None
        self.bot_jid: Optional[str] = None
        self._message_handler: Optional[Callable] = None

    def initialize(self) -> None:
        """Create and configure the neonize client."""
        logger.info(f"Initializing neonize client with session: {settings.neonize_session_path}")

        self.client = NewClient(settings.neonize_session_path)

        # Register event handlers
        self.client.event(ConnectedEv)(self._on_connected)
        self.client.event(PairStatusEv)(self._on_pair_status)
        self.client.event(MessageEv)(self._on_message)

        # Save QR as image instead of relying only on terminal QR.
        # Terminal QR can wrap/crop and become unscannable.
        self.client.qr(self._on_qr)

    def set_message_handler(self, handler: Callable) -> None:
        """Set the callback for incoming messages."""
        self._message_handler = handler

    def _jid_to_str(self, jid_obj) -> str:
        """Convert Neonize/WhatsApp JID object into clean user@server string."""
        try:
            jid = JIDToNonAD(jid_obj)
            user = getattr(jid, "User", "")
            server = getattr(jid, "Server", "")

            if user and server:
                return f"{user}@{server}"

            return str(jid)
        except Exception:
            return str(jid_obj)

    def _resolve_phone_number(self, jid: str) -> str:
        """
        Resolve WhatsApp LID to real phone number using Neonize/Whatsmeow SQLite session.

        Example:
        22982370033670@lid -> 6287877904270
        6287877904270@s.whatsapp.net -> 6287877904270
        """
        if not jid:
            return ""

        user_part = jid.split("@")[0]

        # Normal phone-number JID
        if jid.endswith("@s.whatsapp.net"):
            return user_part

        # LID JID, try SQLite lid map
        if jid.endswith("@lid"):
            try:
                conn = sqlite3.connect(settings.neonize_session_path)
                cur = conn.cursor()
                cur.execute(
                    "SELECT pn FROM whatsmeow_lid_map WHERE lid = ? LIMIT 1",
                    (user_part,),
                )
                row = cur.fetchone()
                conn.close()

                if row and row[0]:
                    logger.debug(f"Resolved LID {user_part} to phone number {row[0]}")
                    return str(row[0])

            except Exception as e:
                logger.warning(f"Could not resolve LID {user_part}: {e}")

            # Fallback if no mapping exists
            return user_part

        # Fallback for groups or unknown formats
        return user_part

    def _on_connected(self, client: NewClient, event: ConnectedEv):
        """Called when the client connects to WhatsApp."""
        self.bot_jid = self._jid_to_str(client.get_me().JID)
        logger.info(f"WhatsApp connected! Bot JID: {self.bot_jid}")

    def _on_pair_status(self, client: NewClient, event: PairStatusEv):
        """Called during QR pairing."""
        logger.info(f"Pairing status: {event}")

    def _on_qr(self, client: NewClient, qr_data: bytes):
        """Save WhatsApp QR code as PNG for easier scanning."""
        try:
            qr_path = Path(settings.rag_log_dir) / "whatsapp_qr.png"
            qr_path.parent.mkdir(parents=True, exist_ok=True)

            segno.make_qr(qr_data).save(str(qr_path), scale=12, border=4)

            logger.info("=" * 60)
            logger.info("New WhatsApp QR generated.")
            logger.info(f"QR image saved to: {qr_path}")
            logger.info("Download/open this PNG and scan it from WhatsApp Linked Devices.")
            logger.info("=" * 60)
        except Exception as e:
            logger.error(f"Failed to save QR image: {e}")

    def _on_message(self, client: NewClient, event: MessageEv):
        """Handle incoming messages."""
        # In this neonize version, event itself is the full Neonize_pb2.Message.
        # event.Info contains sender/chat metadata.
        # event.Message contains the actual WhatsApp text/media payload.
        msg = event

        # Some WhatsApp events are history sync / internal events and do not have Info.
        # Ignore those events so they do not create callback errors.
        if msg is None or not hasattr(msg, "Info"):
            logger.debug("Ignoring non-message event without Info")
            return

        # Extract message info
        info = msg.Info

        try:
            source = info.MessageSource
            sender_jid = self._jid_to_str(source.Sender)
            chat_jid = self._jid_to_str(source.Chat)
            sender_number = self._resolve_phone_number(sender_jid)

            is_group = source.IsGroup
            is_from_me = source.IsFromMe
            push_name = info.Pushname
        except Exception as e:
            logger.exception(f"Failed to parse message source: {e}")
            return

        # Ignore messages from self
        if is_from_me:
            return

        # Check for document message
        document_msg = getattr(msg.Message, "documentMessage", None)
        if document_msg and not getattr(document_msg, "mimetype", None):
            document_msg = None

        # Extract text content (could be caption)
        text = self._extract_text(msg) or ""
        
        if not text and not document_msg:
            return

        logger.info(
            f"Message from {push_name} ({sender_jid}) "
            f"{'[group]' if is_group else '[private]'}: {text[:80]}... "
            f"[{'Document: ' + document_msg.mimetype if document_msg else 'No Doc'}]"
        )

        # Extract mentioned JIDs
        mentioned_jids = self._extract_mentions(msg)

        # Call the message handler
        if self._message_handler:
            try:
                logger.info(
                    f"Calling message router with sender_number={sender_number}, "
                    f"chat_jid={chat_jid}, is_group={is_group}"
                )
                self._message_handler(
                    sender_number=sender_number,
                    sender_name=push_name,
                    chat_jid=chat_jid,
                    text=text,
                    is_group=is_group,
                    mentioned_jids=mentioned_jids,
                    raw_message=msg,
                    document_msg=document_msg,
                )
                logger.info("Message router finished successfully")
            except Exception as e:
                logger.exception(f"Message router failed: {e}")
        else:
            logger.warning("No message handler registered")

    def _extract_text(self, msg: Message) -> Optional[str]:
        """Extract text content from a message (or its caption)."""
        try:
            message = msg.Message

            # Regular text message
            if getattr(message, "conversation", None):
                return message.conversation

            # Extended text message
            if getattr(message, "extendedTextMessage", None) and message.extendedTextMessage.text:
                return message.extendedTextMessage.text

            # Document message caption
            if getattr(message, "documentMessage", None) and message.documentMessage.caption:
                return message.documentMessage.caption

            return None
        except Exception as e:
            logger.debug(f"Could not extract text: {e}")
            return None

    def _extract_mentions(self, msg: Message) -> list[str]:
        """Extract mentioned JIDs from a message safely."""
        try:
            message = msg.Message

            if not message.extendedTextMessage:
                return []

            context_info = getattr(message.extendedTextMessage, "contextInfo", None)
            if not context_info:
                return []

            mentioned = getattr(context_info, "mentionedJid", None)
            if mentioned is None:
                mentioned = getattr(context_info, "mentionedJID", None)

            if mentioned is None:
                return []

            return list(mentioned)

        except Exception as e:
            logger.debug(f"Could not extract mentions: {e}")
            return []

    def send_reply(self, chat_jid: str, text: str, quoted_message=None) -> bool:
        """Send a text reply to a chat."""
        if not self.client:
            logger.error("Client not initialized")
            return False

        try:
            from neonize.utils.jid import build_jid

            parts = chat_jid.split("@")
            if len(parts) == 2:
                user, server = parts
            else:
                user = chat_jid
                server = "s.whatsapp.net"

            jid = build_jid(user, server)
            self.client.send_message(jid, text)
            logger.info(f"Reply sent to {chat_jid}: {text[:80]}...")
            return True
        except Exception as e:
            logger.error(f"Failed to send reply to {chat_jid}: {e}")
            return False

    def start(self) -> None:
        """Start the WhatsApp client."""
        if not self.client:
            self.initialize()

        logger.info("Starting WhatsApp client...")
        self.client.connect()


# Singleton instance
whatsapp_client = WhatsAppClient()