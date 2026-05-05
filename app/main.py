"""
Main entry point for the Personal WhatsApp RAG Assistant.

Startup sequence:
1. Load config from .env
2. Create data directories
3. Health check: MySQL
4. Health check: Ollama
5. Health check: Qdrant + ensure collections
6. Initialize WhatsApp client
7. Register message handler
8. Start WhatsApp (blocking)
"""

import sys

from loguru import logger

from app.config import settings
from app.mysql_store import mysql_store
from app.ollama_client import ollama_client
from app.qdrant_store import qdrant_store
from app.whatsapp_client import whatsapp_client
from app.message_router import message_router


def setup_logging() -> None:
    """Configure loguru logging."""
    logger.remove()  # Remove default handler

    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )

    # Console output
    logger.add(
        sys.stderr,
        format=log_format,
        level=settings.log_level.upper(),
        colorize=True,
    )

    # File output (if log dir is available)
    try:
        logger.add(
            f"{settings.rag_log_dir}/bot.log",
            format=log_format,
            level="DEBUG",
            rotation="10 MB",
            retention="7 days",
            compression="gz",
        )
    except Exception:
        logger.warning("Could not set up file logging (directory may not exist)")


def run_health_checks() -> bool:
    """Run health checks for all services. Returns True if all pass."""
    logger.info("=" * 60)
    logger.info("Running health checks...")
    logger.info("=" * 60)

    all_ok = True

    # MySQL
    logger.info("Checking MySQL...")
    if mysql_store.health_check():
        logger.info("✅ MySQL: OK")
    else:
        logger.error("❌ MySQL: FAILED")
        all_ok = False

    # Ollama
    logger.info("Checking Ollama...")
    if ollama_client.health_check():
        logger.info("✅ Ollama: OK")
    else:
        logger.error("❌ Ollama: FAILED")
        all_ok = False

    # Qdrant
    logger.info("Checking Qdrant...")
    if qdrant_store.health_check():
        logger.info("✅ Qdrant: OK")
        # Ensure collections exist
        qdrant_store.ensure_collections()
        logger.info("✅ Qdrant collections: ensured")
    else:
        logger.error("❌ Qdrant: FAILED")
        all_ok = False

    logger.info("=" * 60)
    if all_ok:
        logger.info("All health checks passed! ✅")
    else:
        logger.error("Some health checks failed! ❌")
    logger.info("=" * 60)

    return all_ok


def main() -> None:
    """Main entry point."""
    # Setup
    setup_logging()

    logger.info("=" * 60)
    logger.info("Personal WhatsApp RAG Assistant")
    logger.info(f"Environment: {settings.app_env}")
    logger.info("=" * 60)

    # Create data directories
    try:
        settings.ensure_directories()
        logger.info("Data directories ensured")
    except Exception as e:
        logger.warning(f"Could not create data directories: {e}")

    # Health checks
    if not run_health_checks():
        logger.error("Aborting due to failed health checks.")
        logger.error("Please check your .env configuration and service status.")
        sys.exit(1)

    # Initialize and start WhatsApp
    logger.info("Initializing WhatsApp client...")
    whatsapp_client.initialize()
    whatsapp_client.set_message_handler(message_router.handle_message)

    logger.info("Starting WhatsApp client (scan QR code if prompted)...")
    try:
        whatsapp_client.start()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
