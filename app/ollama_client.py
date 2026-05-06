"""
Ollama HTTP client.

Handles chat completions, embeddings, and health checks.
Always calls chat with think=false as per spec.
"""

from typing import Optional

import requests
from loguru import logger

from app.config import settings


class OllamaClient:
    """HTTP client for Ollama API."""

    def __init__(self):
        self.base_url = settings.ollama_base_url.rstrip("/")
        self.main_model = settings.ollama_main_model
        self.embedding_model = settings.ollama_embedding_model
        self.think = settings.ollama_think
        self.timeout = 120  # seconds

    # ── Health Check ─────────────────────────────────────────────────────

    def health_check(self) -> bool:
        """Ping Ollama to verify it's running."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if resp.status_code == 200:
                models = [m["name"] for m in resp.json().get("models", [])]
                logger.info(f"Ollama health check passed. Models: {models}")
                return True
            logger.warning(f"Ollama returned status {resp.status_code}")
            return False
        except requests.RequestException as e:
            logger.error(f"Ollama health check failed: {e}")
            return False

    # ── Chat Completion ──────────────────────────────────────────────────

    def chat(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        system: Optional[str] = None,
        temperature: float = 0.7,
    ) -> str:
        """
        Send a chat completion request to Ollama.

        Args:
            messages: List of {"role": "user"|"assistant", "content": "..."} dicts.
            model: Model name. Defaults to main_model (qwen3-8b-rag).
            system: Optional system prompt (prepended as a system message).
            temperature: Sampling temperature.

        Returns:
            The assistant's reply text.
        """
        model = model or self.main_model

        # Prepend system message if provided
        full_messages = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)

        payload = {
            "model": model,
            "messages": full_messages,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
            "think": self.think,  # Always false per spec
        }

        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            content = data.get("message", {}).get("content", "")
            logger.debug(f"Ollama chat ({model}): {len(content)} chars response")
            return content.strip()
        except requests.RequestException as e:
            logger.error(f"Ollama chat failed ({model}): {e}")
            return ""

    # ── Classification ───────────────────────────────────────────────────

    def classify(self, text: str, system_prompt: str) -> str:
        """
        Classification using the main model.
        Used for routing/intent detection.
        """
        return self.chat(
            messages=[{"role": "user", "content": text}],
            model=self.main_model,
            system=system_prompt,
            temperature=0.1,
        )

    # ── Embeddings ───────────────────────────────────────────────────────

    def embed(self, text: str) -> list[float]:
        """
        Generate an embedding vector using bge-m3.

        Returns:
            1024-dimensional float vector.
        """
        payload = {
            "model": self.embedding_model,
            "input": text,
        }

        try:
            resp = requests.post(
                f"{self.base_url}/api/embed",
                json=payload,
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()

            # Ollama returns {"embeddings": [[...]]} for /api/embed
            embeddings = data.get("embeddings", [])
            if embeddings and len(embeddings) > 0:
                vector = embeddings[0]
                logger.debug(f"Embedding generated: {len(vector)} dimensions")
                return vector

            logger.error("Empty embedding response from Ollama")
            return []
        except requests.RequestException as e:
            logger.error(f"Ollama embed failed: {e}")
            return []

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        return [self.embed(text) for text in texts]


# Singleton instance
ollama_client = OllamaClient()
