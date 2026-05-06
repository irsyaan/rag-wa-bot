"""
RAG engine.

Flow:
1. Embed the user question using Ollama bge-m3.
2. Search Qdrant collections (hybrid: vector + keyword).
3. Keep results above collection-specific score thresholds.
4. Build context.
5. Ask Ollama main model using only the retrieved context.
6. Clean answer (strip <think> tags, whitespace).
7. Return answer, sources, and response time.
"""

import json
import re
import time
from collections import defaultdict, deque
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import requests
from loguru import logger

from app.config import settings
from app.mysql_store import mysql_store
from app.prompts import NO_CONTEXT_REPLY, NO_CONTEXT_REPLY_EN, RAG_SYSTEM_PROMPT
from app.qdrant_store import qdrant_store


# ── Shared Ollama options for deterministic, fast output ─────────────────────

OLLAMA_CHAT_OPTIONS = {
    "temperature": 0,
    "top_p": 0.8,
    "top_k": 20,
    "repeat_penalty": 1.1,
    "num_ctx": 2048,
    "num_predict": 160,
}

OLLAMA_PARSE_OPTIONS = {
    "temperature": 0,
    "top_p": 0.8,
    "top_k": 20,
    "repeat_penalty": 1.05,
    "num_ctx": 512,
    "num_predict": 80,
}

OLLAMA_IP_EXTRACT_OPTIONS = {
    "temperature": 0,
    "top_p": 0.8,
    "top_k": 20,
    "repeat_penalty": 1.05,
    "num_ctx": 2048,
    "num_predict": 220,
}

IP_LOOKUP_SCORE_THRESHOLD = 0.45
MIN_RAG_BEST_SCORE = 0.62
IP_RELEVANCE_STOPWORDS = {
    "ada",
    "alamat",
    "all",
    "apa",
    "apakah",
    "berapa",
    "bot",
    "butuh",
    "cari",
    "coba",
    "dan",
    "dari",
    "di",
    "for",
    "give",
    "ip",
    "ipv4",
    "is",
    "itu",
    "list",
    "lookup",
    "mana",
    "me",
    "minta",
    "nya",
    "saja",
    "saya",
    "semua",
    "server",
    "show",
    "the",
    "tolong",
    "untuk",
    "what",
    "whats",
    "yang",
}
IP_PATTERN = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}"
    r"(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"
)


@dataclass
class RagSource:
    collection: str
    point_id: str
    score: float
    text: str
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RagResult:
    answer: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    response_time_ms: int = 0


@dataclass
class ParsedIntent:
    intent: str = "unknown"
    entity: str = ""
    suffix: str = ""
    language: str = "id"
    used_llm: bool = False


class RagEngine:
    """RAG search and answer engine."""

    def __init__(self):
        self.ollama_url = settings.ollama_base_url.rstrip("/")
        self.qdrant_url = settings.qdrant_url.rstrip("/")

        self.embedding_model = settings.ollama_embedding_model
        self.main_model = settings.ollama_main_model
        self.fast_model = settings.ollama_fast_model

        self.max_results = int(settings.rag_max_results)
        self.default_score_threshold = float(settings.rag_score_threshold)

        self.memory_collection = settings.qdrant_memory_collection
        self.knowledge_collection = settings.qdrant_knowledge_collection
        self.chat_collection = settings.qdrant_chat_collection

        self.collections = [
            self.memory_collection,
            self.knowledge_collection,
            self.chat_collection,
        ]

        # Different threshold per collection.
        # Personal memory should be more flexible because users often ask casually/slang.
        # Document knowledge also needs flexibility for cross-lingual or brief queries.
        self.collection_thresholds = {
            self.memory_collection: 0.62,
            self.knowledge_collection: 0.62,
            self.chat_collection: 0.68,
        }

        # Per-sender rolling conversation buffer: last 3 (user, bot) pairs
        # key: sender_number -> deque of (user_msg, bot_reply) tuples
        self._history: Dict[str, Deque[Tuple[str, str]]] = defaultdict(lambda: deque(maxlen=3))

        # Important:
        # In proxy mode, global HTTP_PROXY/HTTPS_PROXY exists for WhatsApp.
        # For local Ollama/Qdrant calls, do not use proxy.
        self.http = requests.Session()
        self.http.trust_env = False

    def _post_json(self, url: str, payload: Dict[str, Any], timeout: int = 120) -> Dict[str, Any]:
        response = self.http.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()

    # ── Answer cleaner ───────────────────────────────────────────────────

    @staticmethod
    def _clean_model_answer(text: str) -> str:
        """
        Strip <think>...</think> blocks and leftover tags from model output.
        Safety layer even when using models that shouldn't produce them.
        """
        # Remove full <think>...</think> blocks (greedy, multiline)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        # Remove leftover orphan tags
        text = re.sub(r"</?think>", "", text)
        text = text.strip()

        if not text:
            return "Maaf, saya belum bisa menjawab pertanyaan ini."

        return text

    # ── Embedding ────────────────────────────────────────────────────────

    def _embed(self, text: str) -> List[float]:
        """Create embedding using Ollama."""
        data = self._post_json(
            f"{self.ollama_url}/api/embed",
            {
                "model": self.embedding_model,
                "input": text,
                "keep_alive": "10m",
            },
            timeout=120,
        )

        embeddings = data.get("embeddings") or []
        if not embeddings:
            raise RuntimeError("Ollama embedding response did not contain embeddings")

        return embeddings[0]

    # ── Qdrant helpers ───────────────────────────────────────────────────

    def _extract_points(self, qdrant_response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Support both Qdrant response shapes.

        Current query API:
        {
          "result": {
            "points": [...]
          }
        }

        Older/search style:
        {
          "result": [...]
        }
        """
        result = qdrant_response.get("result")

        if isinstance(result, dict):
            points = result.get("points", [])
            if isinstance(points, list):
                return points

        if isinstance(result, list):
            return result

        return []

    def _payload_text(self, payload: Dict[str, Any]) -> str:
        """Extract text from possible payload keys."""
        for key in ["text", "content", "chunk_text", "memory", "document_text"]:
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        return ""

    def _threshold_for_collection(self, collection: str) -> float:
        """Get score threshold for a specific collection."""
        return self.collection_thresholds.get(collection, self.default_score_threshold)

    def _search_collection(
        self,
        collection: str,
        vector: List[float],
        question: Optional[str] = None,
        threshold: Optional[float] = None,
        limit: Optional[int] = None,
    ) -> List[RagSource]:
        """Search one Qdrant collection with Hybrid (Vector + Keyword)."""
        threshold = threshold if threshold is not None else self._threshold_for_collection(collection)
        limit = limit or self.max_results

        try:
            results = qdrant_store.search(
                collection=collection,
                query_vector=vector,
                query_text=question,
                limit=limit,
                score_threshold=threshold,
            )

            sources: List[RagSource] = []
            for point in results:
                score = float(point.get("score", 0.0))
                payload = point.get("payload") or {}
                text = self._payload_text(payload)

                sources.append(
                    RagSource(
                        collection=collection,
                        point_id=str(point.get("id", "")),
                        score=score,
                        text=text,
                        payload=payload,
                    )
                )

            logger.info(f"Qdrant search {collection}: {len(sources)} results (threshold={threshold})")
            return sources

        except Exception as e:
            logger.error(f"Qdrant search failed for collection {collection}: {e}")
            return []

    def _search_all(
        self,
        question: str,
        threshold: Optional[float] = None,
        limit: Optional[int] = None,
    ) -> Tuple[List[RagSource], int, int]:
        """
        Embed question and search all configured Qdrant collections.

        Returns:
            (sources, embed_ms, search_ms)
        """
        t0 = time.time()
        vector = self._embed(question)
        embed_ms = int((time.time() - t0) * 1000)

        t1 = time.time()
        all_sources: List[RagSource] = []

        limit = limit or self.max_results

        for collection in self.collections:
            results = self._search_collection(
                collection,
                vector,
                question,
                threshold=threshold,
                limit=limit,
            )
            all_sources.extend(results)

        all_sources.sort(key=lambda item: item.score, reverse=True)
        search_ms = int((time.time() - t1) * 1000)

        return all_sources[:limit], embed_ms, search_ms

    def _build_context(self, sources: List[RagSource]) -> str:
        """Build context text for the LLM."""
        blocks = []

        for idx, source in enumerate(sources, start=1):
            source_name = (
                source.payload.get("source")
                or source.payload.get("filename")
                or source.payload.get("source_name")
                or source.collection
            )

            blocks.append(
                f"[Source {idx}]\n"
                f"Collection: {source.collection}\n"
                f"Source: {source_name}\n"
                f"Score: {source.score:.4f}\n"
                f"Text:\n{source.text}"
            )

        return "\n\n---\n\n".join(blocks)

    # ── LLM call ─────────────────────────────────────────────────────────

    def _ask_ollama(
        self,
        question: str,
        context: str,
        role: str,
        sender_name: str = "User",
        recent_history: str = "",
        parsed_intent: Optional[ParsedIntent] = None,
    ) -> Tuple[str, int]:
        """
        Ask Ollama using retrieved context.

        Returns:
            (answer_text, llm_ms)
        """
        # Use configured timezone offset (default +7 for WIB)
        tz = timezone(timedelta(hours=settings.timezone_offset))
        now = datetime.now(tz)
        current_time = now.strftime("%H:%M")
        hour = now.hour

        if 5 <= hour < 12:
            time_period = "morning (pagi)"
        elif 12 <= hour < 15:
            time_period = "afternoon (siang)"
        elif 15 <= hour < 18:
            time_period = "afternoon (sore)"
        elif 18 <= hour < 21:
            time_period = "evening (malam)"
        else:
            time_period = "night (malam)"

        prompt = RAG_SYSTEM_PROMPT.format(
            current_time=current_time,
            time_period=time_period,
            sender_name=sender_name or "User",
            role=role,
            context=context or "No context available.",
            recent_history=recent_history or "(no recent conversation)",
        )

        parsed_lines = []
        if parsed_intent:
            parsed_lines.append(f"Parsed intent: {parsed_intent.intent}")
            if parsed_intent.entity:
                parsed_lines.append(f"Parsed requested target/filter: {parsed_intent.entity}")
            if parsed_intent.suffix:
                parsed_lines.append(f"Parsed IP suffix filter: {parsed_intent.suffix}")

        if parsed_lines:
            prompt += "\n\nParsed request:\n" + "\n".join(parsed_lines)

        prompt += f"\n\nMessage:\n{question}\n\nAnswer:"

        t0 = time.time()

        data = self._post_json(
            f"{self.ollama_url}/api/chat",
            {
                "model": self.main_model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "think": False,
                "stream": False,
                "keep_alive": "10m",
                "options": OLLAMA_CHAT_OPTIONS,
            },
            timeout=180,
        )

        llm_ms = int((time.time() - t0) * 1000)

        message = data.get("message") or {}
        answer = message.get("content", "").strip()

        if not answer:
            return "Maaf, saya belum memiliki informasi yang cukup untuk menjawab pertanyaan ini.", llm_ms

        # Clean any <think> leaks
        answer = self._clean_model_answer(answer)

        return answer, llm_ms

    # ── Utility methods ──────────────────────────────────────────────────

    def _log_failed_question(
        self,
        sender_number: str,
        chat_jid: str,
        question: str,
        best_score: Optional[float],
    ) -> None:
        """Log failed question if mysql_store supports it."""
        try:
            if hasattr(mysql_store, "log_failed_question"):
                mysql_store.log_failed_question(
                    sender_number=sender_number,
                    question_text=question,
                    chat_jid=chat_jid,
                    attempted_sources=None,
                    best_score=best_score,
                )
                return

            if hasattr(mysql_store, "add_failed_question"):
                mysql_store.add_failed_question(
                    sender_number=sender_number,
                    question_text=question,
                    chat_jid=chat_jid,
                    best_score=best_score,
                )
                return

            logger.debug("mysql_store has no failed-question method; skipping failed question log")

        except Exception as e:
            logger.warning(f"Could not log failed question: {e}")

    @staticmethod
    def _no_context_reply(question: str) -> str:
        """Return no-context reply in the user's likely language."""
        lowered = question.lower()
        english_markers = {"what", "where", "which", "who", "when", "how", "show", "list", "give"}
        if any(marker in lowered.split() for marker in english_markers):
            return NO_CONTEXT_REPLY_EN
        return NO_CONTEXT_REPLY

    @staticmethod
    def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
        """Extract the first JSON object from a small LLM parser response."""
        if not text:
            return None

        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None

        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

        return parsed if isinstance(parsed, dict) else None

    @staticmethod
    def _looks_like_greeting_typo(text: str) -> bool:
        """Catch short stretched greeting typos after the fast parser has had first pass."""
        normalized = re.sub(r"[^a-zA-Z]", "", (text or "").lower())
        if not normalized or len(normalized) > 16:
            return False

        greeting_shapes = [
            r"h+i+",
            r"h+y+",
            r"h+e+l+[oa]*w*",
            r"h+a+l+[oa]*w*",
            r"h+a+i+w*",
            r"p+a+g+i+",
            r"s+i+a+n+g+",
            r"s+o+r+e+",
            r"m+a+l+a+m+",
        ]
        return any(re.fullmatch(pattern, normalized) for pattern in greeting_shapes)

    def parse_message_intent(self, text: str) -> ParsedIntent:
        """
        Use the fast model only as an intent/entity parser.

        The parser must not answer facts. Python still executes retrieval and formatting.
        """
        text = (text or "").strip()
        if not text:
            return ParsedIntent()

        system_prompt = (
            "You classify short WhatsApp messages for an IT assistant. "
            "Return ONLY compact JSON with keys: intent, entity, suffix, language. "
            "intent must be one of: greeting, ip_lookup, other. "
            "Treat greeting typos and casual variants as greeting, for example "
            "helo, heloo, hy, haii, hallaww, hellaww, hallo, halo, pagi, siang, sore, malam. "
            "For ip_lookup, entity is the target system/location/device only, "
            "without filler words like gw, lupa, berapa, brp, ini, sih. "
            "suffix is only the last IP octet if the user asks by suffix, otherwise empty. "
            "language is id or en. Do not answer the user. Do not invent IPs."
        )

        try:
            data = self._post_json(
                f"{self.ollama_url}/api/chat",
                {
                    "model": self.fast_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": text},
                    ],
                    "think": False,
                    "stream": False,
                    "keep_alive": "10m",
                    "options": OLLAMA_PARSE_OPTIONS,
                },
                timeout=20,
            )
            content = data.get("message", {}).get("content", "")
            parsed = self._extract_json_object(content)
            if parsed:
                intent = str(parsed.get("intent", "unknown")).strip().lower()
                if intent not in {"greeting", "ip_lookup", "other"}:
                    intent = "unknown"
                if intent in {"unknown", "other"} and self._looks_like_greeting_typo(text):
                    intent = "greeting"

                entity = str(parsed.get("entity", "") or "").strip().lower()
                suffix = str(parsed.get("suffix", "") or "").strip().lstrip(".")
                language = str(parsed.get("language", "id") or "id").strip().lower()
                if language not in {"id", "en"}:
                    language = "id"

                return ParsedIntent(
                    intent=intent,
                    entity=entity,
                    suffix=suffix if suffix.isdigit() else "",
                    language=language,
                    used_llm=True,
                )

            logger.warning(f"Intent parser returned non-JSON response: {content[:120]}")

        except Exception as e:
            logger.warning(f"Fast intent parser failed, using fallback parser: {e}")

        if self._looks_like_greeting_typo(text):
            return ParsedIntent(intent="greeting", language="id", used_llm=False)

        return self._fallback_parse_message_intent(text)

    def _fallback_parse_message_intent(self, text: str) -> ParsedIntent:
        """Deterministic backup when the fast parser is unavailable."""
        lowered = text.lower()
        suffixes = self._requested_ip_suffixes(text)

        if IP_PATTERN.search(text) or "ip" in lowered or suffixes:
            terms = self._query_terms_for_ip(text)
            return ParsedIntent(
                intent="ip_lookup",
                entity=" ".join(sorted(terms)),
                suffix=next(iter(suffixes), ""),
                language="en" if any(word in lowered.split() for word in {"what", "which", "where"}) else "id",
                used_llm=False,
            )

        return ParsedIntent(
            intent="other",
            language="en" if any(word in lowered.split() for word in {"hello", "hi", "hey"}) else "id",
            used_llm=False,
        )

    @staticmethod
    def _query_terms_for_ip(question: str) -> set[str]:
        """Extract meaningful entity/location tokens for deterministic IP filtering."""
        terms = set()
        for token in re.findall(r"[a-zA-Z0-9][a-zA-Z0-9_-]{1,}", question.lower()):
            if token in IP_RELEVANCE_STOPWORDS:
                continue
            if token.isdigit():
                continue
            terms.add(token)
        return terms

    def _query_terms_from_entity(self, entity: str) -> set[str]:
        """Convert parser entity into lookup tokens."""
        if not entity:
            return set()
        return self._query_terms_for_ip(entity)

    @staticmethod
    def _requested_ip_suffixes(question: str) -> set[str]:
        """Detect suffix lookups like '.38' or 'ending 38'."""
        suffixes = set()
        lowered = question.lower()
        for match in re.findall(r"(?:\.|ending\s+|akhiran\s+|suffix\s+)(\d{1,3})\b", lowered):
            suffixes.add(match)
        return suffixes

    @staticmethod
    def _label_from_text(text: str) -> Optional[str]:
        """Extract a compact server/host label from simple memory text."""
        patterns = [
            r"\b(?:adalah|is|name is|hostname is)\s+([a-zA-Z0-9][a-zA-Z0-9_.-]{1,})",
            r"\b([a-zA-Z0-9][a-zA-Z0-9_.-]{1,})\s+(?:dan\s+)?(?:ip|ipnya|ip-nya)\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                label = match.group(1).strip(".,:; ")
                if label.lower() not in IP_RELEVANCE_STOPWORDS:
                    return label
        return None

    @staticmethod
    def _ip_context(text: str, ip: str) -> str:
        """Return the smallest useful text window around a matched IP."""
        for line in text.splitlines():
            if ip in line:
                return line.strip()

        index = text.find(ip)
        if index < 0:
            return text[:300]

        start = max(0, index - 180)
        end = min(len(text), index + len(ip) + 180)
        return text[start:end].strip()

    @staticmethod
    def _source_label(source: RagSource) -> str:
        payload_label = (
            source.payload.get("name")
            or source.payload.get("hostname")
        )
        if payload_label:
            return payload_label

        text_label = RagEngine._label_from_text(source.text)
        if text_label:
            return text_label

        return (
            source.payload.get("source")
            or source.payload.get("filename")
            or source.payload.get("source_name")
            or source.collection
        )

    @staticmethod
    def _is_all_ip_request(question: str, parsed_intent: Optional[ParsedIntent]) -> bool:
        """Return True when the user asks for every IP, not a specific target."""
        lowered = question.lower()
        entity = (parsed_intent.entity if parsed_intent else "").strip().lower()
        if entity in {"", "all", "all ip", "all ips", "semua", "semua ip"}:
            return True
        return any(phrase in lowered for phrase in ["all ip", "all ips", "semua ip", "list all ip"])

    @staticmethod
    def _title_from_entity(entity: str) -> str:
        """Create a compact answer heading from the parsed requested target."""
        cleaned = re.sub(r"[^a-zA-Z0-9_. -]+", " ", entity or "").strip()
        if not cleaned:
            return ""

        words = []
        for word in cleaned.split():
            if len(word) <= 3:
                words.append(word.upper())
            else:
                words.append(word[:1].upper() + word[1:])
        return " ".join(words)

    def _align_ip_answer_heading(self, answer: str, parsed_intent: Optional[ParsedIntent]) -> str:
        """
        Keep LLM content but correct misleading infrastructure headings for targeted IP answers.

        Example: user asks FreshFactory, model returns "Zstack e1:" because those rows live
        under that infra heading. Rewrite only the heading; leave extracted IP rows unchanged.
        """
        if not parsed_intent or parsed_intent.intent != "ip_lookup":
            return answer
        if self._is_all_ip_request("", parsed_intent):
            return answer
        if not parsed_intent.entity or not IP_PATTERN.search(answer or ""):
            return answer

        target_terms = self._query_terms_from_entity(parsed_intent.entity)
        if not target_terms:
            return answer

        lines = answer.splitlines()
        first_content_idx = next((idx for idx, line in enumerate(lines) if line.strip()), None)
        if first_content_idx is None:
            return answer

        first_line = lines[first_content_idx].strip()
        is_heading = first_line.endswith(":") and not IP_PATTERN.search(first_line)
        if not is_heading:
            return answer

        heading_terms = set(re.findall(r"[a-zA-Z0-9][a-zA-Z0-9_-]{1,}", first_line.lower()))
        if target_terms & heading_terms:
            return answer

        title = self._title_from_entity(parsed_intent.entity)
        if not title:
            return answer

        lines[first_content_idx] = f"{title}:"
        return "\n".join(lines)

    def _format_ip_matches_from_json(
        self,
        raw_text: str,
        context: str,
        question: str,
        parsed_intent: Optional[ParsedIntent],
    ) -> str:
        """Validate structured LLM IP extraction and format WhatsApp-safe bullets."""
        no_answer = (
            NO_CONTEXT_REPLY_EN
            if parsed_intent and parsed_intent.language == "en"
            else NO_CONTEXT_REPLY
        )
        parsed = self._extract_json_object(raw_text)
        if not parsed:
            logger.warning(f"IP extractor returned non-JSON response: {raw_text[:160]}")
            return no_answer

        matches = parsed.get("matches")
        if not isinstance(matches, list):
            return no_answer

        all_request = self._is_all_ip_request(question, parsed_intent)
        target_terms = set()
        if parsed_intent and parsed_intent.entity and not all_request:
            target_terms = self._query_terms_from_entity(parsed_intent.entity)

        context_lower = context.lower()
        suffix = parsed_intent.suffix if parsed_intent else ""
        lines = []
        seen = set()

        for item in matches:
            if not isinstance(item, dict):
                continue

            ip = str(item.get("ip", "") or "").strip()
            name = str(item.get("name", "") or "").strip()
            evidence = str(item.get("evidence", "") or "").strip()

            if not IP_PATTERN.fullmatch(ip):
                continue
            if ip not in context:
                continue
            if suffix and ip.rsplit(".", 1)[-1] != suffix:
                continue
            if not evidence or ip not in evidence or evidence.lower() not in context_lower:
                continue
            if target_terms and not any(term in evidence.lower() for term in target_terms):
                continue

            label = name if name else "Unknown"
            key = (ip, label)
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"- {ip} - {label}")

        return "\n".join(lines) if lines else no_answer

    def _ask_ollama_ip_lookup(
        self,
        question: str,
        context: str,
        parsed_intent: Optional[ParsedIntent] = None,
    ) -> Tuple[str, int]:
        """Use the LLM to extract hostname/IP pairs from retrieved context only."""
        language = parsed_intent.language if parsed_intent else "id"
        entity = parsed_intent.entity if parsed_intent else ""
        suffix = parsed_intent.suffix if parsed_intent else ""
        all_request = self._is_all_ip_request(question, parsed_intent)
        target_rule = (
            "This is an all-IP request, so include every IP row present in Context."
            if all_request
            else "This is a targeted request. Include only matches whose evidence explicitly mentions the parsed target."
        )

        prompt = f"""You extract IP information from retrieved IT document context.

Rules:
- Answer ONLY from Context.
- Do not use outside knowledge.
- Do not invent IPs, hostnames, products, locations, or relationships.
- {target_rule}
- Each match must include a short evidence string copied from Context that contains the IP.
- For targeted requests, the evidence must also contain the parsed target word.
- If no matching IP is in Context, return {{"matches":[]}}.
- Return ONLY JSON in this shape:
{{"matches":[{{"ip":"172.22.255.38","name":"HOSTNAME_OR_NAME","evidence":"copied context text containing target and IP"}}]}}

User message: {question}
Parsed target: {entity or "(none)"}
Parsed suffix: {suffix or "(none)"}

Context:
{context}
"""

        t0 = time.time()
        data = self._post_json(
            f"{self.ollama_url}/api/chat",
            {
                "model": self.main_model,
                "messages": [{"role": "user", "content": prompt}],
                "think": False,
                "stream": False,
                "keep_alive": "10m",
                "options": OLLAMA_IP_EXTRACT_OPTIONS,
            },
            timeout=120,
        )
        llm_ms = int((time.time() - t0) * 1000)
        raw_answer = data.get("message", {}).get("content", "").strip()
        raw_answer = self._clean_model_answer(raw_answer) if raw_answer else ""
        answer = self._format_ip_matches_from_json(raw_answer, context, question, parsed_intent)

        return answer, llm_ms

    def answer_ip_lookup(
        self,
        question: str,
        sender_number: str,
        chat_jid: str,
        parsed_intent: Optional[ParsedIntent] = None,
    ) -> RagResult:
        """Answer IP lookups by retrieving context, then asking LLM to extract IP rows."""
        start_time = time.time()
        question = question.strip()
        logger.info(f"Direct IP lookup started for {sender_number}: {question}")

        try:
            parsed_intent = parsed_intent or self._fallback_parse_message_intent(question)
            search_text = question
            if parsed_intent.entity and parsed_intent.entity not in search_text.lower():
                search_text = f"{question} {parsed_intent.entity}"
            if parsed_intent.suffix and parsed_intent.suffix not in search_text:
                search_text = f"{search_text} {parsed_intent.suffix}".strip()

            sources, embed_ms, search_ms = self._search_all(
                search_text,
                threshold=IP_LOOKUP_SCORE_THRESHOLD,
                limit=max(self.max_results, 8),
            )

            sources_with_ip = [source for source in sources if IP_PATTERN.search(source.text)]
            if not sources_with_ip:
                total_ms = int((time.time() - start_time) * 1000)
                best_score = sources[0].score if sources else None
                self._log_failed_question(sender_number, chat_jid, question, best_score)
                logger.info(
                    f"IP lookup found no retrieved source containing an IP. "
                    f"embed={embed_ms}ms, search={search_ms}ms, total={total_ms}ms"
                )
                return RagResult(
                    answer=self._no_context_reply(question),
                    sources=[],
                    response_time_ms=total_ms,
                )

            context = self._build_context(sources_with_ip)
            answer, llm_ms = self._ask_ollama_ip_lookup(question, context, parsed_intent)
            total_ms = int((time.time() - start_time) * 1000)

            if answer in {NO_CONTEXT_REPLY, NO_CONTEXT_REPLY_EN}:
                best_score = sources[0].score if sources else None
                self._log_failed_question(sender_number, chat_jid, question, best_score)

            source_payloads = [
                {
                    "collection": source.collection,
                    "point_id": source.point_id,
                    "score": source.score,
                    "source": self._source_label(source),
                    "text": source.text[:500],
                }
                for source in sources_with_ip
            ]

            logger.info(
                f"LLM IP lookup finished. embed={embed_ms}ms, search={search_ms}ms, "
                f"llm={llm_ms}ms, total={total_ms}ms"
            )
            return RagResult(
                answer=answer,
                sources=source_payloads,
                response_time_ms=total_ms,
            )

        except Exception as e:
            logger.exception(f"Direct IP lookup failed: {e}")
            elapsed_ms = int((time.time() - start_time) * 1000)
            return RagResult(
                answer="Maaf, terjadi error saat memproses pertanyaan.",
                sources=[],
                response_time_ms=elapsed_ms,
            )

    def get_rejection_message(self, text: str) -> str:
        """Use the fast model to generate a translated rejection message."""
        prompt = f"Translate the exact phrase 'You don't have the permission to do this.' into the language used in the following text. Reply ONLY with the translated phrase, no quotes or extra words.\n\nText: {text}"
        try:
            data = self._post_json(
                f"{self.ollama_url}/api/chat",
                {
                    "model": self.fast_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": OLLAMA_CHAT_OPTIONS,
                },
                timeout=30,
            )
            message = data.get("message", {}).get("content", "").strip()
            answer = self._clean_model_answer(message) if message else ""
            return answer if answer else "You don't have the permission to do this."
        except Exception as e:
            logger.error(f"Failed to translate rejection message: {e}")
            return "You don't have the permission to do this."

    # ── Main answer method ───────────────────────────────────────────────

    def answer(
        self,
        question: str,
        sender_number: str,
        chat_jid: str,
        role: str = "user",
        sender_name: str = "User",
        parsed_intent: Optional[ParsedIntent] = None,
    ) -> RagResult:
        """Main RAG answer function for complex questions after direct handlers."""
        start_time = time.time()
        question = question.strip()

        logger.info(f"RAG started for {sender_number}: {question}")

        # Build recent conversation history string
        history_deque = self._history[sender_number]
        if history_deque:
            history_lines = []
            for user_msg, bot_reply in history_deque:
                history_lines.append(f"User: {user_msg}")
                history_lines.append(f"Bot: {bot_reply}")
            recent_history = "\n".join(history_lines)
        else:
            recent_history = "(no recent conversation)"

        try:
            # Step 1: Embed + Search
            search_query = question
            search_threshold = None
            search_limit = self.max_results

            if parsed_intent and parsed_intent.intent == "ip_lookup":
                search_threshold = IP_LOOKUP_SCORE_THRESHOLD
                search_limit = max(self.max_results, 8)
                search_terms = [question]
                if parsed_intent.entity and parsed_intent.entity not in question.lower():
                    search_terms.append(parsed_intent.entity)
                if parsed_intent.suffix and parsed_intent.suffix not in question:
                    search_terms.append(parsed_intent.suffix)
                search_query = " ".join(search_terms)

            sources, embed_ms, search_ms = self._search_all(
                search_query,
                threshold=search_threshold,
                limit=search_limit,
            )

            best_score = sources[0].score if sources else None
            min_best_score = (
                IP_LOOKUP_SCORE_THRESHOLD
                if parsed_intent and parsed_intent.intent == "ip_lookup"
                else MIN_RAG_BEST_SCORE
            )
            if not sources or best_score < min_best_score:
                self._log_failed_question(sender_number, chat_jid, question, best_score)
                total_ms = int((time.time() - start_time) * 1000)
                logger.info(
                    f"RAG stopped before LLM due weak/no retrieval. "
                    f"best_score={best_score}, embed={embed_ms}ms, "
                    f"search={search_ms}ms, total={total_ms}ms"
                )
                return RagResult(
                    answer=self._no_context_reply(question),
                    sources=[],
                    response_time_ms=total_ms,
                )

            logger.info(
                f"RAG found {len(sources)} source(s). "
                f"Best score={sources[0].score:.4f}, collection={sources[0].collection}"
            )

            # Step 2: Build context
            context = self._build_context(sources)
            context_len = len(context)

            # Step 3: Ask LLM
            answer, llm_ms = self._ask_ollama(
                question,
                context,
                role,
                sender_name=sender_name,
                recent_history=recent_history,
                parsed_intent=parsed_intent,
            )
            answer = self._align_ip_answer_heading(answer, parsed_intent)

            # Store this exchange in the rolling buffer
            self._history[sender_number].append((question, answer))

            # Step 4: Timing logs
            total_ms = int((time.time() - start_time) * 1000)
            logger.info(
                f"RAG timing: embed={embed_ms}ms, search={search_ms}ms, "
                f"context={context_len}chars, llm={llm_ms}ms, total={total_ms}ms"
            )

            source_payloads = [
                {
                    "collection": source.collection,
                    "point_id": source.point_id,
                    "score": source.score,
                    "source": self._source_label(source),
                    "text": source.text[:500],
                }
                for source in sources
            ]

            return RagResult(
                answer=answer,
                sources=source_payloads,
                response_time_ms=total_ms,
            )

        except Exception as e:
            logger.exception(f"RAG answer failed: {e}")

            elapsed_ms = int((time.time() - start_time) * 1000)

            return RagResult(
                answer="Maaf, terjadi error saat memproses pertanyaan.",
                sources=[],
                response_time_ms=elapsed_ms,
            )


rag_engine = RagEngine()
