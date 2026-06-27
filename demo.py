"""
demo.py
-------
Interactive walkthrough of the SemanticSupportCache, mirroring the
step-by-step tutorial described in the article.

Run with:
    python demo.py                     # Ollama (default)
    python demo.py --provider openrouter
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import argparse
import os
import time

from semantic_cache import SemanticSupportCache

SYSTEM_PROMPT = (
    "You are a helpful customer support assistant for an e-commerce store. "
    "Answer in Markdown. Be concise (3-5 sentences max)."
)


def hr(char: str = "─", width: int = 60) -> None:
    print(char * width)


def section(title: str) -> None:
    print()
    hr("═")
    print(f"  {title}")
    hr("═")


def demo_query(cache: SemanticSupportCache, query: str, category: str = "general") -> None:
    t0 = time.perf_counter()
    result = cache.query(query, category=category, use_cache=True)
    elapsed = (time.perf_counter() - t0) * 1000

    status = "⚡ CACHE HIT" if result["cache_hit"] else "🔵 CACHE MISS (LLM called)"
    print(f"\nQuery : {query}")
    print(f"Status: {status}")
    print(f"Tokens: {result['total_tokens']}  |  Latency: {elapsed:.0f} ms")
    print(f"Answer: {result['answer'][:200]}…" if len(result["answer"]) > 200 else f"Answer: {result['answer']}")
    hr()


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic cache demo")
    parser.add_argument("--provider", default=os.getenv("LLM_PROVIDER", "ollama"),
                        choices=["ollama", "openrouter"])
    parser.add_argument("--model",   default=None)
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args()

    section("1. Initialise SemanticSupportCache")
    print(f"Provider : {args.provider}")
    print("Qdrant   : in-memory (no server needed for this demo)")
    print("Embedding: BAAI/bge-small-en-v1.5 (local, via fastembed)")

    cache = SemanticSupportCache(
        qdrant_url    = ":memory:",
        llm_provider  = args.provider,
        llm_model     = args.model,
        llm_api_key   = args.api_key,
        system_prompt = SYSTEM_PROMPT,
        threshold     = 0.92,
    )

    section("2. Seed the cache (first queries → all MISS → stored in Qdrant)")

    seed_queries = [
        ("Where is my order?",          "order_tracking"),
        ("How do I return an item?",    "return_policy"),
        ("I forgot my password",        "account"),
        ("Do you offer free shipping?", "shipping"),
    ]
    for q, cat in seed_queries:
        demo_query(cache, q, category=cat)

    print(f"\nCache size: {cache.count()} entries")

    section("3. Semantic variations — should all be CACHE HITS")

    similar_queries = [
        ("Track my package",                   "order_tracking"),
        ("I want to send back my purchase",    "return_policy"),
        ("How do I reset my login?",           "account"),
        ("Is there free delivery available?",  "shipping"),
        ("What's the status of my delivery?",  "order_tracking"),
    ]
    for q, cat in similar_queries:
        demo_query(cache, q, category=cat)

    section("4. Genuinely new query — should be a CACHE MISS")

    demo_query(cache, "What payment methods do you accept?", "payment")

    section("5. Cache invalidation demonstration")

    print("Simulating a return policy change → invalidating 'return_policy' category…")
    cache.invalidate_by_category("return_policy")
    print(f"Cache size after invalidation: {cache.count()} entries")
    print()
    print("Re-querying with the same return question → should be a CACHE MISS again:")
    demo_query(cache, "How do I return an item?", "return_policy")

    section("6. Summary")
    print("✓ Semantic cache demo complete.")
    print("  Run `python benchmark.py` for a full token/cost comparison report.")


if __name__ == "__main__":
    main()
