"""
semantic_cache.py
-----------------
Core SemanticSupportCache class.

Supports three LLM providers:
  • "ollama"     – local Ollama server (free)
  • "anthropic"  – Anthropic API directly (sk-ant-... key)
  • "openrouter" – OpenRouter / OpenAI-compatible endpoint
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import os
import time
import logging
from datetime import datetime, timezone
from typing import Optional

import httpx
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Embedding model (singleton)
# ---------------------------------------------------------------------------

_EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
_embedding_model: Optional[TextEmbedding] = None


def get_embedding_model() -> TextEmbedding:
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading embedding model '%s'…", _EMBED_MODEL_NAME)
        _embedding_model = TextEmbedding(model_name=_EMBED_MODEL_NAME)
    return _embedding_model


def embed(text: str) -> list[float]:
    """Return a unit-normalised dense vector for *text*."""
    model = get_embedding_model()
    vectors = list(model.embed([text]))
    return vectors[0].tolist()


# ---------------------------------------------------------------------------
# LLM provider wrappers
# ---------------------------------------------------------------------------

DEFAULT_OLLAMA_BASE  = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3.2"
DEFAULT_OR_BASE      = "https://openrouter.ai/api/v1"
DEFAULT_OR_MODEL     = "openai/gpt-4o-mini"
DEFAULT_ANT_MODEL    = "claude-3-haiku-20240307"


def _call_ollama(
    prompt: str,
    model: str,
    base_url: str,
    system_prompt: str = "",
) -> dict:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    url = f"{base_url.rstrip('/')}/api/chat"
    payload = {"model": model, "messages": messages, "stream": False}

    t0 = time.perf_counter()
    resp = httpx.post(url, json=payload, timeout=120)
    latency_ms = (time.perf_counter() - t0) * 1000
    resp.raise_for_status()

    data = resp.json()
    text = data.get("message", {}).get("content", "")
    input_tokens  = data.get("prompt_eval_count", 0)
    output_tokens = data.get("eval_count", 0)

    return {
        "text":          text,
        "input_tokens":  input_tokens,
        "output_tokens": output_tokens,
        "total_tokens":  input_tokens + output_tokens,
        "latency_ms":    latency_ms,
    }


def _call_anthropic(
    prompt: str,
    model: str,
    api_key: str,
    system_prompt: str = "",
) -> dict:
    """Call Anthropic API directly using an sk-ant-... key."""
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key":         api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type":      "application/json",
    }
    payload = {
        "model":      model,
        "max_tokens": 1024,
        "messages":   [{"role": "user", "content": prompt}],
    }
    if system_prompt:
        payload["system"] = system_prompt

    t0 = time.perf_counter()
    resp = httpx.post(url, json=payload, headers=headers, timeout=120)
    latency_ms = (time.perf_counter() - t0) * 1000
    resp.raise_for_status()

    data          = resp.json()
    text          = data["content"][0]["text"]
    usage         = data.get("usage", {})
    input_tokens  = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)

    return {
        "text":          text,
        "input_tokens":  input_tokens,
        "output_tokens": output_tokens,
        "total_tokens":  input_tokens + output_tokens,
        "latency_ms":    latency_ms,
    }


def _call_openrouter(
    prompt: str,
    model: str,
    base_url: str,
    api_key: str,
    system_prompt: str = "",
) -> dict:
    """Call any OpenAI-compatible endpoint (OpenRouter, OpenAI, etc.)."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }
    payload = {"model": model, "messages": messages}

    t0 = time.perf_counter()
    resp = httpx.post(url, json=payload, headers=headers, timeout=120)
    latency_ms = (time.perf_counter() - t0) * 1000
    resp.raise_for_status()

    data          = resp.json()
    choice        = data["choices"][0]
    text          = choice["message"]["content"]
    usage         = data.get("usage", {})
    input_tokens  = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)
    total_tokens  = usage.get("total_tokens", input_tokens + output_tokens)

    return {
        "text":          text,
        "input_tokens":  input_tokens,
        "output_tokens": output_tokens,
        "total_tokens":  total_tokens,
        "latency_ms":    latency_ms,
    }


# ---------------------------------------------------------------------------
# SemanticSupportCache
# ---------------------------------------------------------------------------

COLLECTION_NAME      = "support_cache"
VECTOR_SIZE          = 384
SIMILARITY_THRESHOLD = 0.92


class SemanticSupportCache:
    """
    Semantic cache backed by Qdrant.

    llm_provider: "ollama" | "anthropic" | "openrouter"
    """

    def __init__(
        self,
        *,
        qdrant_url: str = ":memory:",
        qdrant_api_key: Optional[str] = None,
        threshold: float = SIMILARITY_THRESHOLD,
        llm_provider: str = "ollama",
        llm_model: Optional[str] = None,
        llm_base_url: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        system_prompt: str = "",
        collection_name: str = COLLECTION_NAME,
    ) -> None:
        self.threshold       = threshold
        self.llm_provider    = llm_provider.lower()
        self.system_prompt   = system_prompt
        self.collection_name = collection_name

        # ── Resolve LLM settings ──────────────────────────────────────────
        if self.llm_provider == "ollama":
            self.llm_model    = llm_model    or os.getenv("OLLAMA_MODEL",    DEFAULT_OLLAMA_MODEL)
            self.llm_base_url = llm_base_url or os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE)
            self.llm_api_key  = ""

        elif self.llm_provider == "anthropic":
            self.llm_model    = llm_model or os.getenv("ANTHROPIC_MODEL", DEFAULT_ANT_MODEL)
            self.llm_base_url = ""   # not used
            self.llm_api_key  = (
                llm_api_key
                or os.getenv("ANTHROPIC_API_KEY")
                or ""
            )
            if not self.llm_api_key:
                raise ValueError(
                    "An API key is required for the 'anthropic' provider. "
                    "Set ANTHROPIC_API_KEY in your .env file."
                )

        else:  # openrouter
            self.llm_model    = llm_model    or os.getenv("OPENROUTER_MODEL",    DEFAULT_OR_MODEL)
            self.llm_base_url = llm_base_url or os.getenv("OPENROUTER_BASE_URL", DEFAULT_OR_BASE)
            self.llm_api_key  = (
                llm_api_key
                or os.getenv("OPENROUTER_API_KEY")
                or os.getenv("OPENAI_API_KEY")
                or ""
            )
            if not self.llm_api_key:
                raise ValueError(
                    "An API key is required for the 'openrouter' provider. "
                    "Set OPENROUTER_API_KEY in your .env file."
                )

        # ── Qdrant client ─────────────────────────────────────────────────
        if qdrant_url == ":memory:":
            self.client = QdrantClient(location=":memory:")
        else:
            self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

        self._ensure_collection()

    def _ensure_collection(self) -> None:
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qdrant_models.VectorParams(
                    size=VECTOR_SIZE,
                    distance=qdrant_models.Distance.COSINE,
                ),
            )
            logger.info("Created Qdrant collection '%s'.", self.collection_name)

    def check_cache(self, query: str) -> Optional[str]:
        vector = embed(query)
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            limit=1,
            score_threshold=self.threshold,
            with_payload=True,
        )
        results = response.points
        if results:
            hit = results[0]
            logger.debug("Cache HIT (score=%.4f) for query: %s", hit.score, query[:80])
            return hit.payload.get("cached_response", "")
        logger.debug("Cache MISS for query: %s", query[:80])
        return None

    def update_cache(self, query: str, response: str, category: str = "general") -> None:
        import uuid
        vector = embed(query)
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                qdrant_models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        "original_prompt": query,
                        "cached_response": response,
                        "category":        category,
                        "timestamp":       datetime.now(timezone.utc).isoformat(),
                    },
                )
            ],
        )
        logger.debug("Stored new cache entry (category=%s).", category)

    def query(self, user_query: str, category: str = "general", use_cache: bool = True) -> dict:
        t_start = time.perf_counter()

        if use_cache:
            cached = self.check_cache(user_query)
            if cached is not None:
                return {
                    "answer":        cached,
                    "cache_hit":     True,
                    "latency_ms":    (time.perf_counter() - t_start) * 1000,
                    "input_tokens":  0,
                    "output_tokens": 0,
                    "total_tokens":  0,
                }

        llm_result = self._call_llm(user_query)

        if use_cache:
            self.update_cache(user_query, llm_result["text"], category=category)

        return {
            "answer":        llm_result["text"],
            "cache_hit":     False,
            "latency_ms":    (time.perf_counter() - t_start) * 1000,
            "input_tokens":  llm_result["input_tokens"],
            "output_tokens": llm_result["output_tokens"],
            "total_tokens":  llm_result["total_tokens"],
        }

    def invalidate_by_category(self, category: str) -> int:
        result = self.client.delete(
            collection_name=self.collection_name,
            points_selector=qdrant_models.FilterSelector(
                filter=qdrant_models.Filter(
                    must=[
                        qdrant_models.FieldCondition(
                            key="category",
                            match=qdrant_models.MatchValue(value=category),
                        )
                    ]
                )
            ),
        )
        logger.info("Invalidated category '%s' (%s).", category, result.status)
        return 0

    def invalidate_by_ttl(self, max_age_seconds: int) -> None:
        cutoff = datetime.now(timezone.utc).timestamp() - max_age_seconds
        scroll_result = self.client.scroll(
            collection_name=self.collection_name,
            with_payload=True,
            limit=100,
        )
        ids_to_delete: list[str] = []
        while True:
            points, next_offset = scroll_result
            for pt in points:
                ts_str = pt.payload.get("timestamp", "")
                try:
                    ts = datetime.fromisoformat(ts_str).timestamp()
                    if ts < cutoff:
                        ids_to_delete.append(pt.id)
                except ValueError:
                    pass
            if next_offset is None:
                break
            scroll_result = self.client.scroll(
                collection_name=self.collection_name,
                with_payload=True,
                limit=100,
                offset=next_offset,
            )

        if ids_to_delete:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=qdrant_models.PointIdsList(points=ids_to_delete),
            )
            logger.info("TTL invalidation removed %d stale points.", len(ids_to_delete))

    def count(self) -> int:
        return self.client.count(collection_name=self.collection_name).count

    def _call_llm(self, prompt: str) -> dict:
        if self.llm_provider == "ollama":
            return _call_ollama(
                prompt=prompt,
                model=self.llm_model,
                base_url=self.llm_base_url,
                system_prompt=self.system_prompt,
            )
        elif self.llm_provider == "anthropic":
            return _call_anthropic(
                prompt=prompt,
                model=self.llm_model,
                api_key=self.llm_api_key,
                system_prompt=self.system_prompt,
            )
        else:
            return _call_openrouter(
                prompt=prompt,
                model=self.llm_model,
                base_url=self.llm_base_url,
                api_key=self.llm_api_key,
                system_prompt=self.system_prompt,
            )
