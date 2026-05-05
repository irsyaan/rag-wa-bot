"""
WhatsApp client using neonize.

Handles connection, message receiving, QR code display,
and sending replies. Session is persisted to avoid re-scanning QR.
"""

from typing import Callable, Optional

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

    def set_message_handler(self, handler: Callable) -> None:
        """Set the callback for incoming messages."""
        self._message_handler = handler

    def _on_connected(self, client: NewClient, event: ConnectedEv):
        """Called when the client connects to WhatsApp."""
        self.bot_jid = str(JIDToNonAD(client.get_me().JID))
        logger.info(f"WhatsApp connected! Bot JID: {self.bot_jid}")

    def _on_pair_status(self, client: NewClient, event: PairStatusEv):
        """Called during QR pairing."""
        logger.info(f"Pairing status: {event}")

    def _on_message(self, client: NewClient, event: MessageEv):
        """Handle incoming messages."""
        msg = event.Message

        # Extract message info
        info = msg.Info
        sender_jid = str(JIDToNonAD(info.MessageSource.Sender))
        chat_jid = str(JIDToNonAD(info.MessageSource.Chat))
        is_group = info.MessageSource.IsGroup
        is_from_me = info.MessageSource.IsFromMe
        push_name = info.Pushname

        # Ignore messages from self
        if is_from_me:
            return

        # Extract text content
        text = self._extract_text(msg)
        if not text:
            return  # Skip non-text messages for now

        logger.info(
            f"Message from {push_name} ({sender_jid}) "
            f"{'[group]' if is_group else '[private]'}: {text[:80]}..."
        )

        # Extract mentioned JIDs
        mentioned_jids = self._extract_mentions(msg)

        # Call the message handler
        if self._message_handler:
            self._message_handler(
                sender_number=sender_jid.split("@")[0],
                sender_name=push_name,
                chat_jid=chat_jid,
                text=text,
                is_group=is_group,
                mentioned_jids=mentioned_jids,
                raw_message=msg,
            )

    def _extract_text(self, msg: Message) -> Optional[str]:
        """Extract text content from a message."""
        message = msg.Message

        # Regular text message
        if message.conversation:
            return message.conversation

        # Extended text message (quoted, linked, etc.)
        if message.extendedTextMessage and message.extendedTextMessage.text:
            return message.extendedTextMessage.text

        return None

    def _extract_mentions(self, msg: Message) -> list[str]:
        """Extract mentioned JIDs from a message."""
        message = msg.Message
        if message.extendedTextMessage and message.extendedTextMessage.contextInfo:
            return list(message.extendedTextMessage.contextInfo.mentionedJid)
        return []

    def send_reply(self, chat_jid: str, text: str, quoted_message=None) -> bool:
        """Send a text reply to a chat."""
        if not self.client:
            logger.error("Client not initialized")
            return False

        try:
            from neonize.utils.jid import build_jid

            # Parse the JID
            parts = chat_jid.split("@")
            if len(parts) == 2:
                user, server = parts
            else:
                user = chat_jid
                server = "s.whatsapp.net"

            jid = build_jid(user, server)
            self.client.send_message(jid, text)
            logger.debug(f"Reply sent to {chat_jid}: {text[:50]}...")
            return True
        except Exception as e:
            logger.error(f"Failed to send reply to {chat_jid}: {e}")
            return False

    def start(self) -> None:
        """Start the WhatsApp client (blocking)."""
        if not self.client:
            self.initialize()

        logger.info("Starting WhatsApp client...")
        self.client.connect()


# Singleton instance
whatsapp_client = WhatsAppClient()
