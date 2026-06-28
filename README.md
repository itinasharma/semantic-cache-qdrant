# Semantic Caching with Qdrant

**Stop paying your LLM to answer the same question twice.**

---

Imagine running an AI customer support assistant.

One customer asks:
> *"Where is my order?"*

A few seconds later, another asks:
> *"Track my package."*

Different words. Same intent. Yet most AI systems still send both requests to the LLM — paying twice, waiting twice, for nearly identical answers.

This project demonstrates how **semantic caching with Qdrant** detects those repeated intents, serves cached responses in milliseconds, and benchmarks the resulting savings in token usage, latency, and API costs.

---

## Results at a glance

| Metric | Result |
|--------|--------|
| 🎯 Cache hit rate | **57.1%** |
| 🪙 Token reduction | **55.9%** |
| 💰 Cost reduction | **55.7%** |
| ⚡ Avg cache hit latency | **15 ms** |
| 🐢 Avg LLM latency | **2,575 ms** |
| 🔬 Model | `claude-haiku-4-5` · threshold `0.75` |

> Over half of all queries were served from cache in under 15 ms at zero token cost.

---

## Why semantic caching?

Traditional caching is exact-match: the query string must be identical to return a hit. Semantic caching uses vector similarity instead — so queries with the same *intent* but different *wording* still hit the cache.

```
User query
    │
    ▼
Embed query          ← local, ~2 ms, zero cost
    │
    ▼
Search Qdrant        ← cosine similarity ≥ threshold?
    │
    ├─ HIT  ──►  Return cached answer    0 tokens · < 30 ms
    │
    └─ MISS ──►  Call LLM API
                      │
                      ▼
                 Store in Qdrant  ←  future queries can now hit this
                      │
                      ▼
                 Return answer
```

---

## Why Qdrant?

| Database | Why not? |
|----------|----------|
| SQLite / Redis | Not designed for nearest-neighbor vector search |
| Chroma | Great for prototypes, but limited production features |
| Pinecone | Hosted-only, no named vectors, less control |
| **Qdrant** | ✅ Named vectors, metadata filtering, ANN search, runs locally or in cloud, production-ready |

Qdrant's **named vectors** feature is what makes the multi-vector benchmark possible — storing `intent`, `keywords`, and `question` vectors in a single point, then searching each independently.

---

## Benchmark goals

This project answers four concrete engineering questions:

- ✅ How many LLM calls are avoided?
- ✅ How many tokens are saved?
- ✅ How much money is saved?
- ✅ Does multi-vector retrieval improve semantic caching?

---

## Architecture

```
semantic-cache-project/
├── semantic_cache.py          # Core SemanticSupportCache class
├── benchmark.py               # Two-run cost benchmark (Run A vs Run B)
├── multi_vector_benchmark.py  # Single vs multi-vector named-vector comparison
├── demo.py                    # Interactive walkthrough
├── config.py                  # Centralised config loader (.env → cfg singleton)
├── charts/
│   ├── charts.py              # Benchmark charts (5 PNGs)
│   └── mv_comparison_charts.py # Comparison charts (5 PNGs)
├── data/
│   └── queries.json           # 40-query customer-support workload
├── results/                   # Auto-created: CSVs, JSONs, PNGs
├── tests/
│   ├── test_semantic_cache.py # Cache logic, invalidation, thresholds
│   ├── test_multi_vector.py   # Named vectors, keyword extraction, scoring
│   └── test_pricing.py        # Verified pricing table, cost calculation
├── requirements.txt
├── .env.example
├── Makefile
├── CHANGELOG.md
└── README.md
```

---

## Quick start

### 1. Install

```bash
python -m pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
```

Minimal `.env` for Anthropic:
```
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-haiku-4-5
SIMILARITY_THRESHOLD=0.75
```

Verify config loaded correctly:
```bash
python config.py
```

### 3. Run the demo

```bash
python demo.py --provider anthropic --threshold 0.75
```

### 4. Run the benchmark

```bash
python benchmark.py --provider anthropic --model claude-haiku-4-5 --threshold 0.75
```

### 5. Run the multi-vector comparison

```bash
python multi_vector_benchmark.py --provider anthropic --model claude-haiku-4-5 --threshold 0.75
```

### 6. Generate charts

```bash
python charts/charts.py
python charts/mv_comparison_charts.py
```

---

## Benchmark results

### Single-vector benchmark (claude-haiku-4-5, threshold=0.75)

```
╔════════════════════════════════════════════════════════════╗
║           SEMANTIC CACHE BENCHMARK SUMMARY                 ║
╠════════════════════════════════════════════════════════════╣
║  Provider                          anthropic               ║
║  Model                         claude-haiku-4-5            ║
║  Similarity threshold              0.75                    ║
╟────────────────────────────────────────────────────────────╢
║  Pricing source                          exact match       ║
║  Input price                    $1.00 / 1M tokens          ║
║  Output price                   $5.00 / 1M tokens          ║
╟────────────────────────────────────────────────────────────╢
║  Total queries                     21                      ║
║  Cache hits                        12                      ║
║  Cache misses                       9                      ║
║  Cache hit rate                    57.1%                   ║
║  LLM calls avoided                 12                      ║
╟────────────────────────────────────────────────────────────╢
║  Tokens (no cache)              3,532                      ║
║  Tokens (with cache)            1,557                      ║
║  Tokens saved                   1,975 (55.9%)              ║
╟────────────────────────────────────────────────────────────╢
║  Cost (no cache)              $0.015048                    ║
║  Cost (with cache)            $0.006673                    ║
║  Cost saved                   $0.008375 (55.7%)            ║
╟────────────────────────────────────────────────────────────╢
║  Avg latency no cache           2,416 ms                   ║
║  Avg latency w/ cache           1,112 ms                   ║
║  Avg hit latency                   15 ms                   ║
║  Avg miss latency               2,575 ms                   ║
╚════════════════════════════════════════════════════════════╝
```

### Multi-vector comparison

```
SINGLE vs MULTI-VECTOR CACHE COMPARISON
─────────────────────────────────────────────────────────────────
Metric                       Single Vector        Multi Vector
─────────────────────────────────────────────────────────────────
Total queries                       21                  21
Cache hits                          12                  12
Hit rate                          57.1%               57.1%
LLM calls avoided                   12                  12

Tokens saved                     1,975               1,796
Token reduction                  55.9%               50.9%

Cost saved                    $0.008375           $0.007480
Cost reduction                   55.7%               49.7%

Avg latency w/ cache            1,112 ms            1,221 ms
Avg hit latency                    15 ms               42 ms

Vectors stored                       8                   8
Indexing time                      904 ms              302 ms
─────────────────────────────────────────────────────────────────
Engineering note:
  On this dataset (short, focused queries) single-vector was
  sufficient. Multi-vector indexing was 67% faster but query
  latency was 10% slower. Re-run on your own query samples
  before choosing an approach.
─────────────────────────────────────────────────────────────────
```

---

## How the benchmark works

Two runs are executed against the same 21-query workload:

| Phase | Description |
|-------|-------------|
| **Seed** | 8 LLM calls populate the warm cache with real answers |
| **Run A — No cache** | All 21 test queries go to the LLM (cost baseline) |
| **Run B — With cache** | Same 21 queries hit the cache first; ~12 hit, ~9 miss |

The 21 test queries consist of:
- ~12 **paraphrases** of seed questions (expected cache hits)
- ~9 **novel questions** the cache has never seen (expected cache misses)

---

## Multi-vector retrieval

The multi-vector mode uses **Qdrant named vectors** — storing three vectors per cached point instead of one:

| Vector name | What it captures |
|-------------|-----------------|
| `intent` | Full query embedding — what the user wants overall |
| `keywords` | Content-bearing keywords (stop words removed) |
| `question` | Query reframed as a well-formed question |

At query time all three are searched independently and scores are combined:

```
final_score = 0.6 × best_score + 0.4 × avg_score
```

This reduces false positives by requiring both a strong best match **and** consistent agreement across multiple vectors.

---

## Tuning the similarity threshold

| Threshold | Behaviour |
|-----------|-----------|
| `≤ 0.65` | Too permissive — risk of wrong answers |
| `0.70–0.80` | ✅ Recommended for customer-support queries |
| `0.85–0.92` | Strict — fewer hits, very precise |
| `≥ 0.95` | Too strict — almost no cache hits |

> With `BAAI/bge-small-en-v1.5` (fastembed), `0.75` gives the best balance.

---

## Cache invalidation

### TTL-based

```python
# Remove entries older than 24 hours
cache.invalidate_by_ttl(max_age_seconds=86_400)
```

### Category-based

```python
# Policy changed? Purge only that category.
cache.invalidate_by_category("return_policy")
```

---

## Configuration reference

### `.env` variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `ollama` | `anthropic` \| `openrouter` \| `ollama` |
| `ANTHROPIC_API_KEY` | — | Your `sk-ant-...` key |
| `ANTHROPIC_MODEL` | `claude-haiku-4-5` | Anthropic model string |
| `OPENROUTER_API_KEY` | — | Your `sk-or-...` key |
| `OPENROUTER_MODEL` | `anthropic/claude-haiku-4-5` | OpenRouter model string |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Local Ollama URL |
| `OLLAMA_MODEL` | `llama3.2` | Ollama model name |
| `QDRANT_URL` | `:memory:` | Blank = in-memory, or `http://localhost:6333` |
| `QDRANT_API_KEY` | — | Qdrant Cloud key |
| `SIMILARITY_THRESHOLD` | `0.75` | Cosine similarity threshold (0–1) |
| `INPUT_PRICE_PER_MILLION` | — | Override input token price (USD/MTok) |
| `OUTPUT_PRICE_PER_MILLION` | — | Override output token price (USD/MTok) |
| `RESULTS_DIR` | `results` | Output directory for CSVs and PNGs |

### CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--provider` | `LLM_PROVIDER` env | `anthropic` \| `openrouter` \| `ollama` |
| `--model` | provider default | Model identifier |
| `--api-key` | env var | API key (overrides `.env`) |
| `--threshold` | `SIMILARITY_THRESHOLD` env | Cosine similarity threshold |
| `--queries` | built-in list | Path to JSON query workload |
| `--results-dir` | `results/` | Output directory |
| `--input-price` | model default | Input price per MTok in USD |
| `--output-price` | model default | Output price per MTok in USD |
| `--no-single-vector` | off | Skip single-vector run (mv-benchmark only) |
| `--no-multi-vector` | off | Skip multi-vector run (mv-benchmark only) |
| `--verbose` / `-v` | off | Enable DEBUG logging |

---

## Running tests

```bash
pytest tests/ -v
```

**112 tests, 4 skipped** (live embedding tests requiring HuggingFace access).

| Test file | Tests | What it covers |
|-----------|-------|----------------|
| `test_semantic_cache.py` | 36 | Cache hit/miss, invalidation, thresholds, providers |
| `test_multi_vector.py` | 29 | Named vectors, keyword extraction, weighted scoring |
| `test_pricing.py` | 47 | Verified pricing table, `get_pricing()`, cost maths |

---

## Built-in pricing table

> **Last verified: 2026-06-27.**
> Re-verify before any production billing use — prices change without notice.
> Sources: [openai.com/api/pricing](https://openai.com/api/pricing/) · [platform.claude.com/docs/en/about-claude/pricing](https://platform.claude.com/docs/en/about-claude/pricing)

### Anthropic

| Model | Input ($/MTok) | Output ($/MTok) | Notes |
|-------|---------------|----------------|-------|
| `claude-opus-4-8` | $5.00 | $25.00 | Flagship |
| `claude-sonnet-4-6` | $3.00 | $15.00 | Balanced |
| `claude-haiku-4-5` | $1.00 | $5.00 | ✅ Recommended for benchmarking |

### OpenAI

| Model | Input ($/MTok) | Output ($/MTok) | Notes |
|-------|---------------|----------------|-------|
| `gpt-5.5` | $5.00 | $30.00 | Flagship |
| `gpt-5.4` | $2.50 | $15.00 | Mid-tier |
| `gpt-5.4-mini` | $0.75 | $4.50 | Budget / default fallback |

Override any price with `--input-price` / `--output-price`.

---

## Tech stack

| Component | Library |
|-----------|---------|
| Vector database | [qdrant-client](https://github.com/qdrant/qdrant-client) ≥ 1.9 |
| Local embeddings | [fastembed](https://github.com/qdrant/fastembed) (`BAAI/bge-small-en-v1.5`, 384-dim) |
| HTTP client | [httpx](https://www.python-httpx.org/) |
| Charts | [matplotlib](https://matplotlib.org/) |
| Config | [python-dotenv](https://github.com/theskumar/python-dotenv) |
| Testing | [pytest](https://pytest.org/) |

---

## License

MIT
