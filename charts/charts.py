"""
charts.py
---------
Generate publication-ready charts from benchmark results.

Reads the most-recent (or a specified) benchmark_summary_*.json file
and produces:

  1. bar_token_comparison.png   – tokens used with vs without cache
  2. bar_cost_comparison.png    – API cost with vs without cache
  3. pie_cache_hits.png         – cache hit vs miss distribution
  4. bar_latency_comparison.png – average latency comparison
  5. summary_dashboard.png      – 2×2 combined overview figure

Usage
-----
    python charts.py                          # auto-picks latest results
    python charts.py --summary results/benchmark_summary_20240101T120000Z.json
    python charts.py --out charts/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _import_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        return plt, mpatches
    except ImportError as e:
        print(f"ERROR: matplotlib is required for chart generation.\n  pip install matplotlib\n\n{e}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

C_NO_CACHE   = "#E74C3C"   # red
C_WITH_CACHE = "#2ECC71"   # green
C_HIT        = "#3498DB"   # blue
C_MISS       = "#E67E22"   # orange
C_LATENCY_A  = "#9B59B6"   # purple
C_LATENCY_B  = "#1ABC9C"   # teal


def _bar(ax, labels, values, colours, ylabel, title, fmt="{:.0f}"):
    bars = ax.bar(labels, values, color=colours, width=0.4, edgecolor="white", linewidth=1.5)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.02,
            fmt.format(val),
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )
    ax.set_ylim(0, max(values) * 1.18)
    return bars


# ---------------------------------------------------------------------------
# Individual chart functions
# ---------------------------------------------------------------------------

def chart_tokens(summary: dict, out_dir: Path, plt) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    _bar(
        ax,
        labels  = ["Without Cache", "With Cache"],
        values  = [summary["tokens_no_cache"], summary["tokens_with_cache"]],
        colours = [C_NO_CACHE, C_WITH_CACHE],
        ylabel  = "Total Tokens",
        title   = f"Token Usage Comparison\n(saved {summary['tokens_saved_pct']:.1f}% of tokens)",
        fmt     = "{:,.0f}",
    )
    path = out_dir / "bar_token_comparison.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path}")
    return path


def chart_cost(summary: dict, out_dir: Path, plt) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    _bar(
        ax,
        labels  = ["Without Cache", "With Cache"],
        values  = [summary["cost_usd_no_cache"], summary["cost_usd_with_cache"]],
        colours = [C_NO_CACHE, C_WITH_CACHE],
        ylabel  = "Estimated API Cost (USD)",
        title   = f"API Cost Comparison\n(saved {summary['cost_reduction_pct']:.1f}% → ${summary['cost_saved_usd']:.6f})",
        fmt     = "${:.6f}",
    )
    path = out_dir / "bar_cost_comparison.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path}")
    return path


def chart_pie(summary: dict, out_dir: Path, plt) -> Path:
    fig, ax = plt.subplots(figsize=(5, 5))
    sizes  = [summary["cache_hits"], summary["cache_misses"]]
    labels = [
        f"Cache Hits\n({summary['cache_hits']} queries)",
        f"Cache Misses\n({summary['cache_misses']} queries)",
    ]
    colours = [C_HIT, C_MISS]
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels     = labels,
        colors     = colours,
        autopct    = "%1.1f%%",
        startangle = 90,
        wedgeprops = {"edgecolor": "white", "linewidth": 2},
    )
    for at in autotexts:
        at.set_fontsize(12)
        at.set_fontweight("bold")
    ax.set_title(
        f"Cache Hit Distribution\n({summary['total_queries']} total queries)",
        fontsize=13, fontweight="bold", pad=15,
    )
    path = out_dir / "pie_cache_hits.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path}")
    return path


def chart_latency(summary: dict, out_dir: Path, plt) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["No Cache\n(all LLM)", "With Cache\n(avg)", "Cache Hit\nonly", "Cache Miss\nonly"]
    values = [
        summary["avg_latency_ms_no_cache"],
        summary["avg_latency_ms_with_cache"],
        summary["avg_cache_hit_latency_ms"],
        summary["avg_cache_miss_latency_ms"],
    ]
    colours = [C_NO_CACHE, C_WITH_CACHE, C_HIT, C_MISS]
    _bar(
        ax,
        labels  = labels,
        values  = values,
        colours = colours,
        ylabel  = "Average Latency (ms)",
        title   = "Latency Comparison",
        fmt     = "{:.0f} ms",
    )
    path = out_dir / "bar_latency_comparison.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path}")
    return path


def chart_dashboard(summary: dict, out_dir: Path, plt) -> Path:
    """2×2 combined overview — the article's main chart."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        "Semantic Cache Benchmark — Overview Dashboard\n"
        f"Model: {summary.get('model', '–')}  |  "
        f"Threshold: {summary.get('threshold', '–')}  |  "
        f"Queries: {summary['total_queries']}",
        fontsize=14, fontweight="bold", y=1.01,
    )

    # Top-left: tokens
    _bar(
        axes[0][0],
        labels  = ["No Cache", "With Cache"],
        values  = [summary["tokens_no_cache"], summary["tokens_with_cache"]],
        colours = [C_NO_CACHE, C_WITH_CACHE],
        ylabel  = "Tokens",
        title   = f"Token Usage  (−{summary['tokens_saved_pct']:.1f}%)",
        fmt     = "{:,.0f}",
    )

    # Top-right: cost
    _bar(
        axes[0][1],
        labels  = ["No Cache", "With Cache"],
        values  = [summary["cost_usd_no_cache"], summary["cost_usd_with_cache"]],
        colours = [C_NO_CACHE, C_WITH_CACHE],
        ylabel  = "Cost (USD)",
        title   = f"API Cost  (−{summary['cost_reduction_pct']:.1f}%)",
        fmt     = "${:.6f}",
    )

    # Bottom-left: hit/miss pie
    sizes   = [summary["cache_hits"], summary["cache_misses"]]
    pie_labels = [
        f"Hits ({summary['cache_hit_rate_pct']:.1f}%)",
        f"Misses ({100 - summary['cache_hit_rate_pct']:.1f}%)",
    ]
    axes[1][0].pie(
        sizes, labels=pie_labels,
        colors=[C_HIT, C_MISS],
        autopct="%1.1f%%", startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    axes[1][0].set_title("Cache Hit Rate", fontsize=13, fontweight="bold", pad=10)

    # Bottom-right: latency
    _bar(
        axes[1][1],
        labels  = ["No Cache", "Avg\nw/ Cache", "Hit\nonly", "Miss\nonly"],
        values  = [
            summary["avg_latency_ms_no_cache"],
            summary["avg_latency_ms_with_cache"],
            summary["avg_cache_hit_latency_ms"],
            summary["avg_cache_miss_latency_ms"],
        ],
        colours = [C_NO_CACHE, C_WITH_CACHE, C_HIT, C_MISS],
        ylabel  = "Avg Latency (ms)",
        title   = "Latency Comparison",
        fmt     = "{:.0f}ms",
    )

    fig.tight_layout(pad=2.5)
    path = out_dir / "summary_dashboard.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path}")
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def find_latest_summary(results_dir: Path) -> Path:
    candidates = sorted(results_dir.glob("benchmark_summary_*.json"), reverse=True)
    if not candidates:
        print(f"ERROR: No benchmark_summary_*.json files found in {results_dir}")
        sys.exit(1)
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate charts from benchmark results")
    parser.add_argument("--summary",     default=None,
                        help="Path to benchmark_summary_*.json (auto-detected if omitted)")
    parser.add_argument("--results-dir", default="results",
                        help="Directory to scan for the latest results (default: results/)")
    parser.add_argument("--out",         default=None,
                        help="Output directory for PNG files (defaults to same as results dir)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    summary_path = Path(args.summary) if args.summary else find_latest_summary(results_dir)
    out_dir = Path(args.out) if args.out else summary_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading summary: {summary_path}")
    with open(summary_path) as f:
        summary = json.load(f)

    plt, _ = _import_matplotlib()

    print("\nGenerating charts…")
    chart_tokens(summary, out_dir, plt)
    chart_cost(summary, out_dir, plt)
    chart_pie(summary, out_dir, plt)
    chart_latency(summary, out_dir, plt)
    chart_dashboard(summary, out_dir, plt)
    print(f"\nAll charts saved to {out_dir}/")


if __name__ == "__main__":
    main()
