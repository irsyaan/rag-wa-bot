"""
Configuration loader using Pydantic Settings.

All connection values come from .env file.
Nothing is hardcoded — switch between server Docker, server host, or local Mac
by changing .env values only.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from .env file."""

    # ── General ──────────────────────────────────────────────────────────
    app_env: str = Field(default="dev", alias="APP_ENV")
    log_level: str = Field(default="info", alias="LOG_LEVEL")

    # ── WhatsApp ─────────────────────────────────────────────────────────
    whatsapp_bot_number: str = Field(default="6281994039680", alias="WHATSAPP_BOT_NUMBER")
    owner_whatsapp_number: str = Field(default="6287877904270", alias="OWNER_WHATSAPP_NUMBER")

    # ── MySQL ────────────────────────────────────────────────────────────
    mysql_host: str = Field(default="host.docker.internal", alias="MYSQL_HOST")
    mysql_port: int = Field(default=3306, alias="MYSQL_PORT")
    mysql_user: str = Field(default="rag", alias="MYSQL_USER")
    mysql_password: str = Field(default="P@ssw0rd", alias="MYSQL_PASSWORD")
    mysql_database: str = Field(default="whatsapp_rag", alias="MYSQL_DATABASE")

    # ── Ollama ───────────────────────────────────────────────────────────
    ollama_base_url: str = Field(default="http://host.docker.internal:11434", alias="OLLAMA_BASE_URL")
    ollama_main_model: str = Field(default="qwen3-8b-rag", alias="OLLAMA_MAIN_MODEL")
    ollama_fast_model: str = Field(default="qwen3-4b-fast", alias="OLLAMA_FAST_MODEL")
    ollama_embedding_model: str = Field(default="bge-m3", alias="OLLAMA_EMBEDDING_MODEL")
    ollama_think: bool = Field(default=False, alias="OLLAMA_THINK")

    # ── Qdrant ───────────────────────────────────────────────────────────
    qdrant_url: str = Field(default="http://qdrant:6333", alias="QDRANT_URL")
    qdrant_knowledge_collection: str = Field(default="personal_knowledge", alias="QDRANT_KNOWLEDGE_COLLECTION")
    qdrant_memory_collection: str = Field(default="personal_memory", alias="QDRANT_MEMORY_COLLECTION")
    qdrant_chat_collection: str = Field(default="conversation_memory", alias="QDRANT_CHAT_COLLECTION")
    qdrant_vector_size: int = Field(default=1024, alias="QDRANT_VECTOR_SIZE")
    qdrant_distance: str = Field(default="Cosine", alias="QDRANT_DISTANCE")

    # ── RAG ──────────────────────────────────────────────────────────────
    rag_upload_dir: str = Field(default="/srv/rag/personal/uploads", alias="RAG_UPLOAD_DIR")
    rag_processed_dir: str = Field(default="/srv/rag/personal/processed", alias="RAG_PROCESSED_DIR")
    rag_log_dir: str = Field(default="/srv/rag/personal/logs", alias="RAG_LOG_DIR")
    rag_max_results: int = Field(default=5, alias="RAG_MAX_RESULTS")
    rag_score_threshold: float = Field(default=0.65, alias="RAG_SCORE_THRESHOLD")

    # ── Neonize ──────────────────────────────────────────────────────────
    neonize_session_path: str = Field(
        default="/srv/rag/personal/neonize-session/bot.sqlite3",
        alias="NEONIZE_SESSION_PATH",
    )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
        "populate_by_name": True,
    }

    def ensure_directories(self) -> None:
        """Create data directories if they don't exist (for server deployment)."""
        dirs = [
            self.rag_upload_dir,
            self.rag_processed_dir,
            self.rag_log_dir,
            os.path.dirname(self.neonize_session_path),
        ]
        for d in dirs:
            Path(d).mkdir(parents=True, exist_ok=True)


# Singleton settings instance
settings = Settings()
