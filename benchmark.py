"""
benchmark.py
------------
Benchmark the monetary impact of semantic caching.

Two runs are executed against the same query workload:
  Run A – NO cache  : every query hits the LLM.
  Run B – WITH cache: cache is checked first; LLM only called on miss.

A summary is printed to stdout and saved as both CSV and JSON under
the results/ directory.

Usage
-----
    python benchmark.py                         # Ollama, default settings
    python benchmark.py --provider openrouter   # OpenRouter (needs API key)
    python benchmark.py --queries data/queries.json --threshold 0.90

All CLI options are documented in `parse_args()` below.
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import argparse
import csv
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from semantic_cache import SemanticSupportCache

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


# ---------------------------------------------------------------------------
# Pricing registry  (per 1 000 000 tokens, USD)
# ---------------------------------------------------------------------------

PRICING_PER_MILLION: dict[str, dict[str, float]] = {
    # OpenAI
    "gpt-4o":              {"input": 2.50,  "output": 10.00},
    "gpt-4o-mini":         {"input": 0.15,  "output": 0.60},
    "openai/gpt-4o-mini":  {"input": 0.15,  "output": 0.60},
    "openai/gpt-4o":       {"input": 2.50,  "output": 10.00},
    "gpt-3.5-turbo":       {"input": 0.50,  "output": 1.50},
    # Anthropic (via OpenRouter)
    "anthropic/claude-3-haiku":   {"input": 0.25,  "output": 1.25},
    "anthropic/claude-3-sonnet":  {"input": 3.00,  "output": 15.00},
    "anthropic/claude-opus-4":    {"input": 15.00, "output": 75.00},
    # Meta / open-source on OpenRouter
    "meta-llama/llama-3.2-3b-instruct": {"input": 0.06, "output": 0.06},
    "meta-llama/llama-3.1-8b-instruct": {"input": 0.10, "output": 0.10},
    # Ollama – free (local)
    "ollama":              {"input": 0.00,  "output": 0.00},
    # Anthropic direct
    "anthropic":           {"input": 0.00,  "output": 0.00},  # placeholder; set by model
}

DEFAULT_PRICING = {"input": 0.15, "output": 0.60}   # gpt-4o-mini fallback


def get_pricing(model_id: str) -> dict[str, float]:
    # Exact match first
    if model_id in PRICING_PER_MILLION:
        return PRICING_PER_MILLION[model_id]
    # Longest-key substring match (avoids "gpt-4o" swallowing "gpt-4o-mini")
    best_key, best_prices = "", DEFAULT_PRICING
    for key, prices in PRICING_PER_MILLION.items():
        if key in model_id and len(key) > len(best_key):
            best_key, best_prices = key, prices
    return best_prices


def tokens_to_usd(
    input_tokens: int,
    output_tokens: int,
    pricing: dict[str, float],
) -> float:
    return (
        input_tokens  / 1_000_000 * pricing["input"]
        + output_tokens / 1_000_000 * pricing["output"]
    )


# ---------------------------------------------------------------------------
# Query workload
# ---------------------------------------------------------------------------

# ── Seed queries: used to pre-populate the cache before Run B ────────────────
# These represent the "common questions" an agent has already answered before.
SEED_QUERIES: list[dict] = [
    {"text": "Where is my order?",                    "category": "order_tracking"},
    {"text": "How do I return an item?",              "category": "return_policy"},
    {"text": "I forgot my password",                  "category": "account"},
    {"text": "I want to cancel my subscription",      "category": "cancellation"},
    {"text": "Do you offer free shipping?",           "category": "shipping"},
    {"text": "What payment methods do you accept?",   "category": "payment"},
    {"text": "My item arrived damaged",               "category": "damaged_goods"},
    {"text": "How do I apply a discount code?",       "category": "promotions"},
]

# ── Test queries: the live query workload replayed in both Run A and Run B ────
# ~13 paraphrases (→ cache HIT in Run B) + ~8 novel questions (→ cache MISS)
# Expected hit rate: ~60-65%
DEFAULT_QUERIES: list[dict] = [
    # Paraphrases of seed queries — should HIT the cache in Run B
    {"text": "Track my package",                          "category": "order_tracking"},
    {"text": "What is the status of my delivery?",        "category": "order_tracking"},
    {"text": "Can I check where my parcel is?",           "category": "order_tracking"},
    {"text": "When will my order arrive?",                "category": "order_tracking"},
    {"text": "I want to send back my purchase",           "category": "return_policy"},
    {"text": "What is your refund policy?",               "category": "return_policy"},
    {"text": "How do I reset my login credentials?",      "category": "account"},
    {"text": "I cannot log in to my account",             "category": "account"},
    {"text": "Please cancel my order",                    "category": "cancellation"},
    {"text": "How do I stop my recurring billing?",       "category": "cancellation"},
    {"text": "Is there free delivery on orders?",         "category": "shipping"},
    {"text": "Can I pay with PayPal?",                    "category": "payment"},
    {"text": "The product I received is broken",          "category": "damaged_goods"},

    # Genuinely new questions — should MISS the cache in Run B
    {"text": "Do you have a loyalty rewards program?",    "category": "promotions"},
    {"text": "How do I update my delivery address?",      "category": "account"},
    {"text": "Can I change my order after placing it?",   "category": "order_tracking"},
    {"text": "Do you ship to international addresses?",   "category": "shipping"},
    {"text": "How long does standard delivery take?",     "category": "shipping"},
    {"text": "Is my payment information stored securely?","category": "payment"},
    {"text": "How do I contact customer support?",        "category": "general"},
    {"text": "Do you offer gift wrapping?",               "category": "general"},
]


def load_queries(path: Optional[str]) -> list[dict]:
    if path and Path(path).exists():
        with open(path) as f:
            data = json.load(f)
        logger.info("Loaded %d queries from %s", len(data), path)
        return data
    logger.info("Using built-in default query workload (%d queries).", len(DEFAULT_QUERIES))
    return DEFAULT_QUERIES


# ---------------------------------------------------------------------------
# Per-query result
# ---------------------------------------------------------------------------

@dataclass
class QueryResult:
    query_id:      int
    query_text:    str
    category:      str
    run:           str          # "no_cache" | "with_cache"
    cache_hit:     bool
    latency_ms:    float
    input_tokens:  int
    output_tokens: int
    total_tokens:  int
    cost_usd:      float
    answer:        str = field(default="", repr=False)


# ---------------------------------------------------------------------------
# BenchmarkRunner
# ---------------------------------------------------------------------------

class BenchmarkRunner:
    """
    Orchestrates the two-run benchmark.

    Parameters mirror SemanticSupportCache so the user can configure
    everything from one place.
    """

    def __init__(
        self,
        *,
        provider:         str,
        model:            Optional[str]  = None,
        base_url:         Optional[str]  = None,
        api_key:          Optional[str]  = None,
        threshold:        float          = 0.92,
        system_prompt:    str            = "",
        input_price_per_m:  Optional[float] = None,
        output_price_per_m: Optional[float] = None,
        results_dir:      str            = "results",
    ) -> None:
        self.provider      = provider
        self.model         = model
        self.threshold     = threshold
        self.results_dir   = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # ── Determine pricing ────────────────────────────────────────────
        if provider in ("ollama",):
            _auto = {"input": 0.0, "output": 0.0}
        else:
            _auto = get_pricing(model or "")

        self.pricing = {
            "input":  input_price_per_m  if input_price_per_m  is not None else _auto["input"],
            "output": output_price_per_m if output_price_per_m is not None else _auto["output"],
        }
        logger.info(
            "Pricing: $%.4f/M input tokens, $%.4f/M output tokens",
            self.pricing["input"], self.pricing["output"],
        )

        # ── Shared cache constructor kwargs ──────────────────────────────
        self._cache_kwargs = dict(
            llm_provider=provider,
            llm_model=model,
            llm_base_url=base_url,
            llm_api_key=api_key,
            threshold=threshold,
            system_prompt=system_prompt,
        )

    # ------------------------------------------------------------------
    # Run helpers
    # ------------------------------------------------------------------

    def _run_queries(
        self,
        cache: SemanticSupportCache,
        queries: list[dict],
        use_cache: bool,
        run_label: str,
    ) -> list[QueryResult]:
        results: list[QueryResult] = []
        for idx, q in enumerate(queries, start=1):
            text     = q["text"]
            category = q.get("category", "general")

            logger.info("[%s] Q%02d/%02d: %s", run_label, idx, len(queries), text[:70])

            outcome = cache.query(text, category=category, use_cache=use_cache)

            cost = tokens_to_usd(
                outcome["input_tokens"],
                outcome["output_tokens"],
                self.pricing,
            )

            results.append(QueryResult(
                query_id      = idx,
                query_text    = text,
                category      = category,
                run           = run_label,
                cache_hit     = outcome["cache_hit"],
                latency_ms    = outcome["latency_ms"],
                input_tokens  = outcome["input_tokens"],
                output_tokens = outcome["output_tokens"],
                total_tokens  = outcome["total_tokens"],
                cost_usd      = cost,
                answer        = outcome["answer"],
            ))

            status = "HIT ✓" if outcome["cache_hit"] else f"MISS ({outcome['total_tokens']} tok)"
            logger.info(
                "         → %s | %.0f ms | $%.6f",
                status, outcome["latency_ms"], cost,
            )

        return results

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, queries: list[dict]) -> dict:
        """
        Execute Run A (no cache) then Run B (with cache).

        Strategy:
          Step 1 — Call the LLM for all SEED_QUERIES to get real answers.
                   These seed answers populate the warm cache for Run B.
          Run A  — All test queries go directly to the LLM (no cache).
                   This is the cost baseline.
          Run B  — Same test queries run against the warm cache.
                   Paraphrases of seed questions → cache HIT (0 tokens).
                   Genuinely new questions       → cache MISS (LLM called).
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

        # ── Step 1: Call LLM for seed queries & build warm cache ──────
        logger.info("=" * 60)
        logger.info("SEEDING — Calling LLM for %d seed queries…", len(SEED_QUERIES))
        logger.info("=" * 60)
        cache_seed = SemanticSupportCache(qdrant_url=":memory:", **self._cache_kwargs)
        for sq in SEED_QUERIES:
            result = cache_seed._call_llm(sq["text"])
            cache_seed.update_cache(sq["text"], result["text"], category=sq["category"])
            logger.info("  Seeded: %s", sq["text"][:60])
        logger.info("Warm cache ready with %d entries.", cache_seed.count())

        # ── Run A: no cache (pure LLM baseline on test queries) ───────
        logger.info("=" * 60)
        logger.info("RUN A — No semantic cache (all queries go to LLM)")
        logger.info("=" * 60)
        cache_a = SemanticSupportCache(qdrant_url=":memory:", **self._cache_kwargs)
        results_a = self._run_queries(cache_a, queries, use_cache=False, run_label="no_cache")

        # ── Run B: same test queries against the warm cache ───────────
        logger.info("=" * 60)
        logger.info("RUN B — With semantic cache (threshold=%.2f)", self.threshold)
        logger.info("=" * 60)
        results_b = self._run_queries(cache_seed, queries, use_cache=True, run_label="with_cache")

        # ── Compute summary ───────────────────────────────────────────
        summary = self._compute_summary(results_a, results_b)
        summary["benchmark_timestamp"] = timestamp
        summary["provider"]  = self.provider
        summary["model"]     = self.model or "default"
        summary["threshold"] = self.threshold
        summary["pricing"]   = self.pricing

        # ── Persist results ───────────────────────────────────────────
        self._save_query_results(results_a + results_b, timestamp)
        self._save_summary(summary, timestamp)

        return summary

    # ------------------------------------------------------------------
    # Summary computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_summary(
        no_cache_results:   list[QueryResult],
        with_cache_results: list[QueryResult],
    ) -> dict:
        total_queries = len(no_cache_results)

        # --- No-cache stats ---
        nc_tokens_in  = sum(r.input_tokens  for r in no_cache_results)
        nc_tokens_out = sum(r.output_tokens for r in no_cache_results)
        nc_tokens_tot = sum(r.total_tokens  for r in no_cache_results)
        nc_cost       = sum(r.cost_usd      for r in no_cache_results)
        nc_latency    = [r.latency_ms for r in no_cache_results]
        nc_api_calls  = total_queries       # all go to LLM

        # --- With-cache stats ---
        hits = [r for r in with_cache_results if r.cache_hit]
        misses = [r for r in with_cache_results if not r.cache_hit]

        wc_tokens_in  = sum(r.input_tokens  for r in with_cache_results)
        wc_tokens_out = sum(r.output_tokens for r in with_cache_results)
        wc_tokens_tot = sum(r.total_tokens  for r in with_cache_results)
        wc_cost       = sum(r.cost_usd      for r in with_cache_results)
        wc_latency    = [r.latency_ms for r in with_cache_results]
        wc_api_calls  = len(misses)

        hit_rate        = len(hits) / total_queries if total_queries else 0
        llm_calls_saved = nc_api_calls - wc_api_calls
        tokens_saved    = nc_tokens_tot - wc_tokens_tot
        cost_saved      = nc_cost - wc_cost
        pct_reduction   = (cost_saved / nc_cost * 100) if nc_cost > 0 else 0.0

        avg = lambda lst: sum(lst) / len(lst) if lst else 0
        p95 = lambda lst: sorted(lst)[int(0.95 * len(lst))] if lst else 0

        return {
            "total_queries":              total_queries,
            # Cache metrics
            "cache_hits":                 len(hits),
            "cache_misses":               len(misses),
            "cache_hit_rate_pct":         round(hit_rate * 100, 2),
            "llm_calls_avoided":          llm_calls_saved,
            # Token metrics
            "tokens_no_cache":            nc_tokens_tot,
            "tokens_with_cache":          wc_tokens_tot,
            "tokens_saved":               tokens_saved,
            "tokens_saved_pct":           round(tokens_saved / nc_tokens_tot * 100, 2) if nc_tokens_tot else 0,
            "input_tokens_no_cache":      nc_tokens_in,
            "input_tokens_with_cache":    wc_tokens_in,
            "output_tokens_no_cache":     nc_tokens_out,
            "output_tokens_with_cache":   wc_tokens_out,
            # Cost metrics
            "cost_usd_no_cache":          round(nc_cost, 8),
            "cost_usd_with_cache":        round(wc_cost, 8),
            "cost_saved_usd":             round(cost_saved, 8),
            "cost_reduction_pct":         round(pct_reduction, 2),
            # Latency metrics
            "avg_latency_ms_no_cache":    round(avg(nc_latency), 2),
            "avg_latency_ms_with_cache":  round(avg(wc_latency), 2),
            "p95_latency_ms_no_cache":    round(p95(nc_latency), 2),
            "p95_latency_ms_with_cache":  round(p95(wc_latency), 2),
            "avg_cache_hit_latency_ms":   round(avg([r.latency_ms for r in hits]),   2) if hits   else 0,
            "avg_cache_miss_latency_ms":  round(avg([r.latency_ms for r in misses]), 2) if misses else 0,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_query_results(
        self,
        results: list[QueryResult],
        timestamp: str,
    ) -> None:
        csv_path  = self.results_dir / f"query_results_{timestamp}.csv"
        json_path = self.results_dir / f"query_results_{timestamp}.json"

        # CSV
        fieldnames = [f for f in QueryResult.__dataclass_fields__ if f != "answer"]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                row = asdict(r)
                row.pop("answer", None)
                writer.writerow(row)

        # JSON (includes answers)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in results], f, indent=2, ensure_ascii=False)

        logger.info("Query results → %s  /  %s", csv_path, json_path)

    def _save_summary(self, summary: dict, timestamp: str) -> None:
        csv_path  = self.results_dir / f"benchmark_summary_{timestamp}.csv"
        json_path = self.results_dir / f"benchmark_summary_{timestamp}.json"

        # JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # CSV (flat key-value pairs for easy import into spreadsheets)
        flat = _flatten_dict(summary)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            for k, v in flat.items():
                writer.writerow([k, v])

        logger.info("Benchmark summary → %s  /  %s", csv_path, json_path)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    items: dict = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(_flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def print_summary(summary: dict) -> None:
    """Pretty-print the benchmark summary table to stdout."""
    SEP = "─" * 62
    print()
    print("╔" + "═" * 60 + "╗")
    print("║{:^60}║".format("  SEMANTIC CACHE BENCHMARK SUMMARY  "))
    print("╠" + "═" * 60 + "╣")

    def row(label: str, value: str) -> None:
        print(f"║  {label:<36}{value:>20}  ║")

    def divider() -> None:
        print("╟" + "─" * 60 + "╢")

    row("Provider",            summary.get("provider", "–"))
    row("Model",               summary.get("model", "–"))
    row("Similarity threshold", f"{summary.get('threshold', '–'):.2f}")
    divider()
    row("Total queries",        str(summary["total_queries"]))
    row("Cache hits",           str(summary["cache_hits"]))
    row("Cache misses",         str(summary["cache_misses"]))
    row("Cache hit rate",       f"{summary['cache_hit_rate_pct']:.1f}%")
    row("LLM calls avoided",    str(summary["llm_calls_avoided"]))
    divider()
    row("Tokens (no cache)",    f"{summary['tokens_no_cache']:,}")
    row("Tokens (with cache)",  f"{summary['tokens_with_cache']:,}")
    row("Tokens saved",         f"{summary['tokens_saved']:,}")
    row("Token reduction",      f"{summary['tokens_saved_pct']:.1f}%")
    divider()
    row("Cost (no cache)",      f"${summary['cost_usd_no_cache']:.6f}")
    row("Cost (with cache)",    f"${summary['cost_usd_with_cache']:.6f}")
    row("Cost saved",           f"${summary['cost_saved_usd']:.6f}")
    row("Cost reduction",       f"{summary['cost_reduction_pct']:.1f}%")
    divider()
    row("Avg latency no cache", f"{summary['avg_latency_ms_no_cache']:.0f} ms")
    row("Avg latency w/ cache", f"{summary['avg_latency_ms_with_cache']:.0f} ms")
    row("Avg hit latency",      f"{summary['avg_cache_hit_latency_ms']:.0f} ms")
    row("Avg miss latency",     f"{summary['avg_cache_miss_latency_ms']:.0f} ms")
    print("╚" + "═" * 60 + "╝")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark the monetary impact of Qdrant semantic caching.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Run locally with Ollama (free):
  python benchmark.py --provider ollama --model llama3.2

  # Run with OpenRouter (requires OPENROUTER_API_KEY):
  python benchmark.py --provider openrouter --model openai/gpt-4o-mini

  # Custom queries file:
  python benchmark.py --queries data/my_queries.json

  # Custom pricing override:
  python benchmark.py --input-price 2.50 --output-price 10.00
""",
    )
    p.add_argument("--provider",      default=os.getenv("LLM_PROVIDER", "ollama"),
                   choices=["ollama", "anthropic", "openrouter"],
                   help="LLM provider (default: ollama)")
    p.add_argument("--model",         default=None,
                   help="Model identifier (provider-specific default if omitted)")
    p.add_argument("--base-url",      default=None,
                   help="Override provider base URL")
    p.add_argument("--api-key",       default=None,
                   help="API key (reads OPENROUTER_API_KEY / OPENAI_API_KEY env vars)")
    p.add_argument("--threshold",     type=float, default=0.92,
                   help="Cosine similarity threshold for cache hits (default: 0.92)")
    p.add_argument("--queries",       default=None,
                   help="Path to JSON file with query workload (uses built-in list if omitted)")
    p.add_argument("--results-dir",   default="results",
                   help="Output directory for CSV/JSON results (default: results/)")
    p.add_argument("--input-price",   type=float, default=None,
                   help="Input token price per 1M tokens in USD (overrides model default)")
    p.add_argument("--output-price",  type=float, default=None,
                   help="Output token price per 1M tokens in USD (overrides model default)")
    p.add_argument("--system-prompt", default=(
        "You are a helpful customer support assistant. "
        "Answer concisely in Markdown format."
    ), help="System prompt sent to the LLM on every cache miss")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Enable DEBUG-level logging")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    queries = load_queries(args.queries)

    runner = BenchmarkRunner(
        provider          = args.provider,
        model             = args.model,
        base_url          = args.base_url,
        api_key           = args.api_key,
        threshold         = args.threshold,
        system_prompt     = args.system_prompt,
        input_price_per_m = args.input_price,
        output_price_per_m= args.output_price,
        results_dir       = args.results_dir,
    )

    summary = runner.run(queries)
    print_summary(summary)


if __name__ == "__main__":
    main()
