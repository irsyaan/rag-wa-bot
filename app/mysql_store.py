"""
MySQL database store.

Handles connection pooling, user CRUD, conversation logging,
audit logging, and health checks.

All connection params come from config.py / .env.
"""

import json
from datetime import datetime
from typing import Optional

import mysql.connector
from mysql.connector import pooling, Error as MySQLError
from loguru import logger

from app.config import settings


class MySQLStore:
    """MySQL connection pool and data access layer."""

    def __init__(self):
        self._pool: Optional[pooling.MySQLConnectionPool] = None

    def connect(self) -> None:
        """Initialize the connection pool."""
        try:
            self._pool = pooling.MySQLConnectionPool(
                pool_name="wa_rag_pool",
                pool_size=5,
                pool_reset_session=True,
                host=settings.mysql_host,
                port=settings.mysql_port,
                user=settings.mysql_user,
                password=settings.mysql_password,
                database=settings.mysql_database,
                charset="utf8mb4",
                collation="utf8mb4_unicode_ci",
                autocommit=True,
            )
            logger.info("MySQL connection pool created successfully")
        except MySQLError as e:
            logger.error(f"Failed to create MySQL connection pool: {e}")
            raise

    def _get_conn(self):
        """Get a connection from the pool."""
        if not self._pool:
            self.connect()
        return self._pool.get_connection()

    # ── Health Check ─────────────────────────────────────────────────────

    def health_check(self) -> bool:
        """Test MySQL connectivity."""
        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            cursor.close()
            conn.close()
            logger.info("MySQL health check passed")
            return True
        except MySQLError as e:
            logger.error(f"MySQL health check failed: {e}")
            return False

    # ── User Operations ──────────────────────────────────────────────────

    def get_user(self, whatsapp_number: str) -> Optional[dict]:
        """Get user by WhatsApp number."""
        conn = self._get_conn()
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(
                "SELECT * FROM users WHERE whatsapp_number = %s",
                (whatsapp_number,),
            )
            return cursor.fetchone()
        finally:
            conn.close()

    def add_user(self, whatsapp_number: str, display_name: str, role: str = "user") -> bool:
        """Add a new user. Returns True if inserted, False if already exists."""
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO users (whatsapp_number, display_name, role, is_active)
                   VALUES (%s, %s, %s, 1)
                   ON DUPLICATE KEY UPDATE display_name = VALUES(display_name)""",
                (whatsapp_number, display_name, role),
            )
            conn.commit()
            return cursor.rowcount > 0
        except MySQLError as e:
            logger.error(f"Failed to add user {whatsapp_number}: {e}")
            return False
        finally:
            conn.close()

    def set_role(self, whatsapp_number: str, role: str) -> bool:
        """Update a user's role."""
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE users SET role = %s WHERE whatsapp_number = %s",
                (role, whatsapp_number),
            )
            conn.commit()
            return cursor.rowcount > 0
        except MySQLError as e:
            logger.error(f"Failed to set role for {whatsapp_number}: {e}")
            return False
        finally:
            conn.close()

    def block_user(self, whatsapp_number: str) -> bool:
        """Block a user by setting role to 'blocked'."""
        return self.set_role(whatsapp_number, "blocked")

    def unblock_user(self, whatsapp_number: str) -> bool:
        """Unblock a user by setting role back to 'user'."""
        return self.set_role(whatsapp_number, "user")

    def list_users(self) -> list[dict]:
        """List all users."""
        conn = self._get_conn()
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(
                "SELECT id, whatsapp_number, display_name, role, is_active, created_at FROM users ORDER BY id"
            )
            return cursor.fetchall()
        finally:
            conn.close()

    def is_admin(self, whatsapp_number: str) -> bool:
        """Check if a user is an admin or owner."""
        user = self.get_user(whatsapp_number)
        if not user:
            return False
        return user["role"] in ("admin", "owner") and user["is_active"] == 1

    def is_allowed(self, whatsapp_number: str) -> bool:
        """Check if a user is allowed to interact (not blocked)."""
        user = self.get_user(whatsapp_number)
        if not user:
            return False
        return user["role"] != "blocked" and user["is_active"] == 1

    # ── Conversation Logging ─────────────────────────────────────────────

    def log_conversation(
        self,
        sender_number: str,
        chat_jid: str,
        message_text: str,
        bot_reply: str,
        sender_name: Optional[str] = None,
        is_group: bool = False,
        message_type: str = "text",
        rag_sources: Optional[list] = None,
        response_time_ms: Optional[int] = None,
    ) -> Optional[int]:
        """Log a conversation exchange. Returns the log ID."""
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO conversation_logs
                   (sender_number, sender_name, chat_jid, is_group,
                    message_text, bot_reply, message_type, rag_sources, response_time_ms)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    sender_number,
                    sender_name,
                    chat_jid,
                    1 if is_group else 0,
                    message_text,
                    bot_reply,
                    message_type,
                    json.dumps(rag_sources) if rag_sources else None,
                    response_time_ms,
                ),
            )
            conn.commit()
            return cursor.lastrowid
        except MySQLError as e:
            logger.error(f"Failed to log conversation: {e}")
            return None
        finally:
            conn.close()

    # ── Audit Logging ────────────────────────────────────────────────────

    def log_audit(
        self,
        actor_number: str,
        action: str,
        target: Optional[str] = None,
        details: Optional[dict] = None,
        status: str = "success",
    ) -> Optional[int]:
        """Write an audit log entry. Returns the log ID."""
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO audit_logs (actor_number, action, target, details, status)
                   VALUES (%s, %s, %s, %s, %s)""",
                (
                    actor_number,
                    action,
                    target,
                    json.dumps(details) if details else None,
                    status,
                ),
            )
            conn.commit()
            return cursor.lastrowid
        except MySQLError as e:
            logger.error(f"Failed to log audit: {e}")
            return None
        finally:
            conn.close()

    # ── Failed Questions ─────────────────────────────────────────────────

    def log_failed_question(
        self,
        sender_number: str,
        question_text: str,
        chat_jid: Optional[str] = None,
        attempted_sources: Optional[list] = None,
        best_score: Optional[float] = None,
    ) -> Optional[int]:
        """Log a question the bot could not answer."""
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO failed_questions
                   (sender_number, question_text, chat_jid, attempted_sources, best_score)
                   VALUES (%s, %s, %s, %s, %s)""",
                (
                    sender_number,
                    question_text,
                    chat_jid,
                    json.dumps(attempted_sources) if attempted_sources else None,
                    best_score,
                ),
            )
            conn.commit()
            return cursor.lastrowid
        except MySQLError as e:
            logger.error(f"Failed to log failed question: {e}")
            return None
        finally:
            conn.close()

    # ── Document Tracking ────────────────────────────────────────────────

    def add_document(
        self,
        filename: str,
        file_path: Optional[str] = None,
        file_type: Optional[str] = None,
        file_size: Optional[int] = None,
        uploaded_by: Optional[int] = None,
    ) -> Optional[int]:
        """Register a new document. Returns the document ID."""
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO documents (filename, file_path, file_type, file_size, uploaded_by)
                   VALUES (%s, %s, %s, %s, %s)""",
                (filename, file_path, file_type, file_size, uploaded_by),
            )
            conn.commit()
            return cursor.lastrowid
        except MySQLError as e:
            logger.error(f"Failed to add document {filename}: {e}")
            return None
        finally:
            conn.close()

    def add_chunk(
        self,
        document_id: int,
        chunk_index: int,
        chunk_text: str,
        qdrant_point_id: Optional[str] = None,
        collection_name: str = "personal_knowledge",
    ) -> Optional[int]:
        """Add a document chunk. Returns the chunk ID."""
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO document_chunks
                   (document_id, chunk_index, chunk_text, qdrant_point_id, collection_name)
                   VALUES (%s, %s, %s, %s, %s)""",
                (document_id, chunk_index, chunk_text, qdrant_point_id, collection_name),
            )
            conn.commit()
            return cursor.lastrowid
        except MySQLError as e:
            logger.error(f"Failed to add chunk for doc {document_id}: {e}")
            return None
        finally:
            conn.close()

    def update_document_status(self, document_id: int, status: str, chunk_count: int = 0) -> bool:
        """Update document processing status."""
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE documents SET status = %s, chunk_count = %s WHERE id = %s",
                (status, chunk_count, document_id),
            )
            conn.commit()
            return cursor.rowcount > 0
        except MySQLError as e:
            logger.error(f"Failed to update document {document_id} status: {e}")
            return False
        finally:
            conn.close()


# Singleton instance
mysql_store = MySQLStore()
