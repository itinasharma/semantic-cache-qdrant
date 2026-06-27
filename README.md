# Semantic Caching with Qdrant — Tutorial Project

Companion code for the article  
**"How to Use Semantic Caching with Qdrant to Optimize Token Costs in Customer Support"**

---

## What this project does

In customer-support AI agents, up to **40 % of daily queries are semantic duplicates**  
(e.g. *"Where is my order?"* vs *"Track my package"*).  
Sending every variation to an LLM wastes tokens and inflates your API bill.

This project demonstrates how to intercept those duplicates with a **semantic cache** backed by [Qdrant](https://qdrant.tech/), serving stored answers in **< 30 ms** at **zero token cost**, and then benchmarks the exact monetary savings.

### Dual-path execution flow

```
User query
    │
    ▼
Embed query (fastembed, local, ~2 ms)
    │
    ▼
Search Qdrant (cosine similarity ≥ 0.92?)
    │
    ├─ HIT ──► Return cached Markdown answer
    │           Cost: 0 tokens | Latency: < 30 ms
    │
    └─ MISS ─► Call LLM API
                │
                ▼
               Store (query vector + answer) in Qdrant
                │
                ▼
               Return answer to user
```

---

## Project structure

```
semantic-cache-project/
├── semantic_cache.py      # Core SemanticSupportCache class
├── benchmark.py           # Two-run benchmark + cost analysis + CSV/JSON output
├── demo.py                # Interactive walkthrough mirroring the tutorial
├── charts.py              # Matplotlib chart generator for the article
├── data/
│   └── queries.json       # 40-query customer-support workload
├── results/               # Auto-created; benchmark CSVs and JSONs land here
├── tests/
│   └── test_semantic_cache.py
├── requirements.txt
├── .env.example
├── Makefile
└── README.md
```

---

## Quick start

### 1. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 2. Configure your environment

```bash
cp .env.example .env
# Edit .env — choose a provider and fill in credentials
```

### 3a. Run the demo (Ollama — free, local)

Make sure [Ollama](https://ollama.com) is running and the model is pulled:

```bash
ollama pull llama3.2
python demo.py --provider ollama
```

### 3b. Run the demo (OpenRouter — paid)

```bash
export OPENROUTER_API_KEY=sk-or-...
python demo.py --provider openrouter --model openai/gpt-4o-mini
```

---

## Benchmark

The benchmark runs **two identical query passes**:

| Run | Description |
|-----|-------------|
| **A — No cache** | Every query calls the LLM. |
| **B — With cache** | Cache is checked first; LLM only called on a miss. |

```bash
# Local (Ollama — token counts are real but cost = $0)
python benchmark.py --provider ollama

# Paid (OpenRouter)
python benchmark.py \
  --provider openrouter \
  --model openai/gpt-4o-mini \
  --api-key $OPENROUTER_API_KEY

# Custom query file
python benchmark.py --queries data/queries.json

# Override pricing
python benchmark.py \
  --provider openrouter \
  --input-price 2.50 \
  --output-price 10.00
```

### Sample output

```
╔════════════════════════════════════════════════════════════╗
║           SEMANTIC CACHE BENCHMARK SUMMARY                 ║
╠════════════════════════════════════════════════════════════╣
║  Provider                          ollama                  ║
║  Model                             llama3.2                ║
║  Similarity threshold              0.92                    ║
╟────────────────────────────────────────────────────────────╢
║  Total queries                     40                      ║
║  Cache hits                        29                      ║
║  Cache misses                      11                      ║
║  Cache hit rate                    72.5%                   ║
║  LLM calls avoided                 29                      ║
╟────────────────────────────────────────────────────────────╢
║  Tokens (no cache)                 12,400                  ║
║  Tokens (with cache)               3,410                   ║
║  Tokens saved                      8,990                   ║
║  Token reduction                   72.5%                   ║
╟────────────────────────────────────────────────────────────╢
║  Cost (no cache)              $0.002480                    ║
║  Cost (with cache)            $0.000682                    ║
║  Cost saved                   $0.001798                    ║
║  Cost reduction                    72.5%                   ║
╟────────────────────────────────────────────────────────────╢
║  Avg latency no cache              450 ms                  ║
║  Avg latency w/ cache              148 ms                  ║
║  Avg hit latency                    18 ms                  ║
║  Avg miss latency                  490 ms                  ║
╚════════════════════════════════════════════════════════════╝
```

Results are saved to `results/` as:

- `query_results_<timestamp>.csv` / `.json`  — per-query breakdown
- `benchmark_summary_<timestamp>.csv` / `.json` — aggregate metrics

### Generate charts

```bash
python charts.py
```

Produces five PNGs in the `results/` directory:

| File | Description |
|------|-------------|
| `bar_token_comparison.png` | Token usage with vs without cache |
| `bar_cost_comparison.png` | API cost comparison |
| `pie_cache_hits.png` | Hit / miss distribution |
| `bar_latency_comparison.png` | Latency breakdown |
| `summary_dashboard.png` | 2×2 combined overview (article hero image) |

---

## Configuration reference

### CLI flags (`benchmark.py`)

| Flag | Default | Description |
|------|---------|-------------|
| `--provider` | `ollama` | `ollama` or `openrouter` |
| `--model` | provider default | Model identifier |
| `--base-url` | provider default | Override API base URL |
| `--api-key` | env var | API key for paid providers |
| `--threshold` | `0.92` | Cosine similarity threshold (0–1) |
| `--queries` | built-in list | Path to JSON query workload |
| `--results-dir` | `results/` | Output directory |
| `--input-price` | model default | Input token price per 1M USD |
| `--output-price` | model default | Output token price per 1M USD |
| `--system-prompt` | see code | System prompt for the LLM |
| `--verbose` | off | Enable DEBUG logging |

### Environment variables

See `.env.example` for a full annotated list. Key variables:

```
LLM_PROVIDER          ollama | openrouter
OLLAMA_BASE_URL       http://localhost:11434
OLLAMA_MODEL          llama3.2
OPENROUTER_API_KEY    sk-or-...
OPENROUTER_MODEL      openai/gpt-4o-mini
QDRANT_URL            (blank = in-memory)
SIMILARITY_THRESHOLD  0.92
```

---

## Tuning the similarity threshold

| Threshold | Behaviour |
|-----------|-----------|
| `≤ 0.80` | Too permissive — serving order answers to cancellation requests (false positives) |
| `0.90–0.95` | **Recommended sweet spot** for customer-support queries |
| `≥ 0.98` | Too strict — almost no cache hits; benefits nullified |

---

## Cache invalidation strategies

### Strategy A — Time-to-Live (TTL)

```python
# Remove entries older than 24 hours
cache.invalidate_by_ttl(max_age_seconds=86_400)
```

### Strategy B — Category purge

```python
# Return policy changed? Invalidate only that category.
cache.invalidate_by_category("return_policy")
```

---

## Running tests

```bash
pytest tests/ -v
```

Tests run **entirely offline** using an in-memory Qdrant instance and mocked LLM responses. No API keys or Ollama server required.

---

## Built-in pricing table

The following models have known per-token prices pre-loaded
(all values in USD per 1 M tokens, as of the article's publication date):

| Model | Input | Output |
|-------|-------|--------|
| `openai/gpt-4o` | $2.50 | $10.00 |
| `openai/gpt-4o-mini` | $0.15 | $0.60 |
| `anthropic/claude-3-haiku` | $0.25 | $1.25 |
| `anthropic/claude-3-sonnet` | $3.00 | $15.00 |
| `meta-llama/llama-3.1-8b-instruct` | $0.10 | $0.10 |
| Ollama (local) | $0.00 | $0.00 |

Override any price with `--input-price` / `--output-price`.

---

## Tech stack

| Component | Library |
|-----------|---------|
| Vector database | [qdrant-client](https://github.com/qdrant/qdrant-client) |
| Local embeddings | [fastembed](https://github.com/qdrant/fastembed) (`BAAI/bge-small-en-v1.5`, 384-dim) |
| HTTP client | [httpx](https://www.python-httpx.org/) |
| Charts | [matplotlib](https://matplotlib.org/) |
| Testing | [pytest](https://pytest.org/) |

---

## License

MIT
