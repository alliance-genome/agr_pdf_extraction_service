#!/usr/bin/env python3
"""Benchmark: Two-Tier Model Selection (mini/5.2) vs Original All-5.2.

Runs the current two-tier config (gpt-5-mini for zone resolution, gpt-5.2 for
large zones + header hierarchy) against papers that have cached gpt-5.2 baselines.

Usage:
  # Single paper:
  python benchmark_tiered.py <file_hash> [--output-dir /path/to/results]

  # Multiple papers:
  python benchmark_tiered.py <hash1> <hash2> ... [--output-dir /path/to/results]

  # All available papers:
  python benchmark_tiered.py --all [--output-dir /path/to/results]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Ensure app modules are importable
sys.path.insert(0, os.path.dirname(__file__))

from config import Config
from app.services.llm_service import LLM, compute_cost
from app.services.consensus_service import merge_with_consensus


def load_extractions(file_hash: str) -> tuple[str, str, str]:
    """Load cached GROBID, Docling, Marker outputs for a file hash."""
    cache_dir = Path(Config.CACHE_FOLDER)
    version = Config.EXTRACTION_CONFIG_VERSION

    texts = {}
    for method in ("grobid", "docling", "marker"):
        path = cache_dir / f"v{version}_{file_hash}_{method}.md"
        if not path.exists():
            raise FileNotFoundError(f"Missing cache: {path}")
        texts[method] = path.read_text(encoding="utf-8")

    return texts["grobid"], texts["docling"], texts["marker"]


def load_baseline(file_hash: str) -> str | None:
    """Load the existing gpt-5.2 merged output as quality baseline."""
    cache_dir = Path(Config.CACHE_FOLDER)
    version = Config.EXTRACTION_CONFIG_VERSION
    # Try combo key format first (more specific)
    path = cache_dir / f"v{version}_{file_hash}_docling_grobid_marker_merged.md"
    if path.exists():
        return path.read_text(encoding="utf-8")
    # Fallback to simple merged key
    path = cache_dir / f"v{version}_{file_hash}_merged.md"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return None


def find_all_benchmarkable_hashes() -> list[str]:
    """Find all file hashes that have cached extractions AND a baseline."""
    cache_dir = Path(Config.CACHE_FOLDER)
    version = Config.EXTRACTION_CONFIG_VERSION

    # Find all hashes with all 3 extraction methods
    hashes_with_extractions = set()
    for method in ("grobid", "docling", "marker"):
        method_hashes = set()
        for p in cache_dir.glob(f"v{version}_*_{method}.md"):
            # Extract hash from filename: v4_<hash>_<method>.md
            name = p.name
            prefix = f"v{version}_"
            suffix = f"_{method}.md"
            h = name[len(prefix):-len(suffix)]
            method_hashes.add(h)
        if not hashes_with_extractions:
            hashes_with_extractions = method_hashes
        else:
            hashes_with_extractions &= method_hashes

    # Filter to those with baselines
    result = []
    for h in sorted(hashes_with_extractions):
        if load_baseline(h) is not None:
            result.append(h)
    return result


def compute_similarity(text_a: str, text_b: str) -> float:
    """Simple character-level similarity between two texts."""
    from difflib import SequenceMatcher
    return SequenceMatcher(None, text_a, text_b).ratio()


def run_benchmark(file_hash: str) -> dict:
    """Run the two-tier benchmark on a single paper, return metrics."""
    print(f"\n{'='*60}")
    print(f"  Paper: {file_hash}")
    print(f"{'='*60}")

    grobid, docling, marker = load_extractions(file_hash)
    baseline = load_baseline(file_hash)
    print(f"  GROBID:  {len(grobid):,} chars")
    print(f"  Docling: {len(docling):,} chars")
    print(f"  Marker:  {len(marker):,} chars")
    print(f"  Baseline: {len(baseline):,} chars" if baseline else "  WARNING: No baseline")

    # Create LLM instance (uses current Config defaults: mini for zones, 5.2 for large/hierarchy)
    llm = LLM(
        api_key=Config.OPENAI_API_KEY,
        model=Config.LLM_MODEL_ZONE_RESOLUTION,
        reasoning_effort=Config.LLM_REASONING_EFFORT,
        conflict_batch_size=Config.LLM_CONFLICT_BATCH_SIZE,
        conflict_max_workers=Config.LLM_CONFLICT_MAX_WORKERS,
        conflict_retry_rounds=Config.LLM_CONFLICT_RETRY_ROUNDS,
    )

    # Run consensus pipeline
    t0 = time.monotonic()
    try:
        result_md, metrics, audit = merge_with_consensus(grobid, docling, marker, llm)
        elapsed = time.monotonic() - t0
        success = result_md is not None
    except Exception as e:
        elapsed = time.monotonic() - t0
        result_md = None
        metrics = {"error": str(e)}
        audit = []
        success = False

    # Collect token usage + cost
    usage_summary = llm.usage.summary()
    total_cost, usage_json = compute_cost(usage_summary, Config.LLM_PRICING)

    # Compare to baseline
    baseline_similarity = None
    if baseline and result_md:
        baseline_similarity = round(compute_similarity(result_md, baseline), 4)

    result = {
        "file_hash": file_hash,
        "success": success,
        "elapsed_sec": round(elapsed, 2),
        "total_cost_usd": round(total_cost, 6),
        "result_length": len(result_md) if result_md else 0,
        "baseline_length": len(baseline) if baseline else 0,
        "baseline_similarity": baseline_similarity,
        "usage_json": usage_json,
        "metrics": {k: v for k, v in (metrics or {}).items()
                    if k in ("conflict", "agree_exact", "agree_near", "gap",
                             "failed", "conflict_ratio", "total_blocks",
                             "quality_grade", "quality_score", "degraded_segments",
                             "error")},
        "audit_count": len(audit) if audit else 0,
    }

    # Print summary
    print(f"  Result:     {'SUCCESS' if success else 'FAILED'}")
    print(f"  Time:       {result['elapsed_sec']}s")
    print(f"  Cost:       ${result['total_cost_usd']:.6f}")
    print(f"  Output:     {result['result_length']:,} chars")
    if baseline_similarity is not None:
        print(f"  vs Baseline: {baseline_similarity:.1%} similar")

    # Print usage breakdown
    cost_bk = usage_json.get("breakdown", {})
    print(f"  Token usage:")
    for call_type, data in cost_bk.items():
        model = data.get("model", "?")
        cost_val = data.get("cost_usd", 0.0)
        prompt = data.get("prompt_tokens", 0)
        completion = data.get("completion_tokens", 0)
        print(f"    {call_type} ({model}): {prompt:,} in / {completion:,} out / ${cost_val:.6f}")

    return result, result_md


def main():
    parser = argparse.ArgumentParser(description="Two-tier model benchmark")
    parser.add_argument("file_hashes", nargs="*", help="MD5 hashes of papers to benchmark")
    parser.add_argument("--all", action="store_true", help="Benchmark all available papers")
    parser.add_argument("--output-dir", default=".", help="Directory for results")
    args = parser.parse_args()

    if args.all:
        hashes = find_all_benchmarkable_hashes()
        print(f"Found {len(hashes)} papers with cached extractions + baselines")
    elif args.file_hashes:
        hashes = args.file_hashes
    else:
        parser.error("Provide file hashes or use --all")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nConfig: zone_resolution={Config.LLM_MODEL_ZONE_RESOLUTION}, "
          f"escalation_threshold={Config.ZONE_ESCALATION_THRESHOLD}, "
          f"escalation_model={Config.ZONE_ESCALATION_MODEL}, "
          f"hierarchy={Config.HIERARCHY_LLM_MODEL}")
    print(f"Papers to benchmark: {len(hashes)}")

    all_results = []
    for file_hash in hashes:
        try:
            result, result_md = run_benchmark(file_hash)
            all_results.append(result)

            # Save individual output
            if result_md:
                out_path = output_dir / f"{file_hash}_tiered_output.md"
                out_path.write_text(result_md, encoding="utf-8")
        except FileNotFoundError as e:
            print(f"\n  SKIP {file_hash}: {e}")
            continue
        except Exception as e:
            print(f"\n  ERROR {file_hash}: {e}")
            all_results.append({
                "file_hash": file_hash,
                "success": False,
                "error": str(e),
            })

    # Save all results
    results_path = output_dir / "benchmark_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # Print comparison table
    if all_results:
        print(f"\n{'='*80}")
        print("  BENCHMARK SUMMARY: Two-Tier (mini/5.2) vs All-5.2 Baseline")
        print(f"{'='*80}")
        print(f"{'Hash':<36} {'OK':<5} {'Time(s)':<9} {'Cost($)':<11} {'Similarity':<12} {'Blocks':<8} {'Conflicts':<10}")
        print(f"{'-'*36} {'-'*5} {'-'*9} {'-'*11} {'-'*12} {'-'*8} {'-'*10}")

        total_cost = 0
        total_time = 0
        similarities = []

        for r in all_results:
            if not r.get("success"):
                print(f"{r['file_hash']:<36} FAIL")
                continue

            sim_str = f"{r['baseline_similarity']:.1%}" if r.get("baseline_similarity") else "N/A"
            blocks = r.get("metrics", {}).get("total_blocks", "?")
            conflicts = r.get("metrics", {}).get("conflict", "?")

            print(f"{r['file_hash']:<36} {'Y':<5} {r['elapsed_sec']:<9} "
                  f"${r['total_cost_usd']:<10.4f} {sim_str:<12} {str(blocks):<8} {str(conflicts):<10}")

            total_cost += r["total_cost_usd"]
            total_time += r["elapsed_sec"]
            if r.get("baseline_similarity") is not None:
                similarities.append(r["baseline_similarity"])

        # Aggregate stats
        n = len([r for r in all_results if r.get("success")])
        if n > 0:
            print(f"\n  Papers benchmarked: {n}")
            print(f"  Total cost:         ${total_cost:.4f}")
            print(f"  Avg cost/paper:     ${total_cost / n:.4f}")
            print(f"  Total time:         {total_time:.1f}s")
            print(f"  Avg time/paper:     {total_time / n:.1f}s")
        if similarities:
            avg_sim = sum(similarities) / len(similarities)
            min_sim = min(similarities)
            max_sim = max(similarities)
            print(f"  Avg similarity:     {avg_sim:.1%}")
            print(f"  Min similarity:     {min_sim:.1%}")
            print(f"  Max similarity:     {max_sim:.1%}")


if __name__ == "__main__":
    main()
