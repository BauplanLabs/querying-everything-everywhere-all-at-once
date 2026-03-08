"""
Benchmark: ad hoc (per-branch) vs native multiverse engine.

Downloads parquet data from S3 if not present locally, then runs queries at
varying branch counts to measure latency and sub-phase timing.

Engine strategies
-----------------
Ad hoc:   Independent DataFusion SessionContext per branch, executed in a
          ThreadPoolExecutor(max_workers=2).  Each context registers its own
          copy of every table via register_parquet().

Native:   Depends on the query type:
          - Aggregate / JOIN: a single Rust DataFusion SessionContext with a
            MultiverseTableProvider (UnionExec over per-branch ListingTables)
            plus shared ListingTables.  Everything runs in one Rust process --
            no FFI boundary -- so the optimizer has full visibility for
            predicate pushdown, projection pruning, and shared-table reuse.
            Exposed to Python via the query_native() PyO3 function.
          - Boolean: parallel per-branch evaluation with early termination
            (short-circuit).  Once two branches disagree the supervaluationary
            verdict is determined, so remaining branches are skipped.

Produces JSONL output for paper charts and a printed summary table.

Usage:
    uv run python src/benchmarks/bench.py
    uv run python src/benchmarks/bench.py --max-branches 16 --runs 3
    uv run python src/benchmarks/bench.py --query q3_join --engine native
    uv run python src/benchmarks/bench.py --paper   # generate PDF charts
"""

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import datafusion
import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

S3_BUCKET = "alpha-hello-bauplan"
S3_PREFIX = "benchmarks/multiverse/v2"
LOCAL_DATA_DIR = Path(__file__).parent / "data"

# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------
# Ad hoc SQL runs per-branch (no __branch_id needed).
# Native SQL runs once across all branches via MultiverseTableProvider,
# which injects __branch_id; the query must GROUP BY it.
# Boolean queries use the ad hoc SQL for short-circuit evaluation.
#
# Q1: Simple single-table COUNT with a filter.  Tests raw scan speed.
#     Ad hoc wins because 50 independent small plans beat one large
#     UnionExec plan -- the unified plan pays coordination overhead
#     (RepartitionExec, hash partitioning for GROUP BY) that exceeds
#     the actual work.
#
# Q3: JOIN between branch-specific predictions and a shared 3M-row
#     dimension table with expensive window functions.  Native wins
#     because it computes the CTE once and reuses the hash-join build
#     side, while ad hoc recomputes it per branch.
#
# Q4: Boolean threshold.  Native uses short-circuit: once two branches
#     disagree, the supervaluationary verdict (mixed) is determined and
#     remaining branches are skipped.  Near-constant latency.
# ---------------------------------------------------------------------------

QUERIES = {
    "q1_count": {
        "label": "COUNT predicted buyers (single table)",
        "type": "number",
        "adhoc": (
            "SELECT COUNT(*) AS n_buyers "
            "FROM user_predictions WHERE predicted_label = 1"
        ),
        "native": (
            "SELECT __branch_id, COUNT(*) AS n_buyers "
            "FROM user_predictions WHERE predicted_label = 1 "
            "GROUP BY __branch_id"
        ),
    },
    "q3_join": {
        "label": "COUNT buyers in large segments (JOIN + window functions)",
        "type": "number",
        "adhoc": (
            "WITH user_segments AS ("
            "  SELECT user_id, customer_segment,"
            "    COUNT(*) OVER (PARTITION BY customer_segment) AS seg_size,"
            "    ROW_NUMBER() OVER (PARTITION BY customer_segment ORDER BY user_id) AS seg_rank"
            "  FROM ecommerce_users"
            ") "
            "SELECT COUNT(*) AS n_buyers "
            "FROM user_segments u "
            "JOIN user_predictions p ON u.user_id = p.user_id "
            "WHERE p.predicted_label = 1 AND u.seg_size > 50000 AND u.seg_rank <= 500000"
        ),
        "native": (
            "WITH user_segments AS ("
            "  SELECT user_id, customer_segment,"
            "    COUNT(*) OVER (PARTITION BY customer_segment) AS seg_size,"
            "    ROW_NUMBER() OVER (PARTITION BY customer_segment ORDER BY user_id) AS seg_rank"
            "  FROM ecommerce_users"
            ") "
            "SELECT __branch_id, COUNT(*) AS n_buyers "
            "FROM user_segments u "
            "JOIN user_predictions p ON u.user_id = p.user_id "
            "WHERE p.predicted_label = 1 AND u.seg_size > 50000 AND u.seg_rank <= 500000 "
            "GROUP BY __branch_id"
        ),
    },
    "q4_bool": {
        "label": "Boolean: is conversion rate above 2%?",
        "type": "boolean",
        "adhoc": (
            "SELECT CAST(SUM(CASE WHEN predicted_label = 1 THEN 1 ELSE 0 END) AS DOUBLE) "
            "/ CAST(COUNT(*) AS DOUBLE) > 0.02 AS above_threshold "
            "FROM user_predictions"
        ),
        "native": (
            "SELECT __branch_id, "
            "CAST(SUM(CASE WHEN predicted_label = 1 THEN 1 ELSE 0 END) AS DOUBLE) "
            "/ CAST(COUNT(*) AS DOUBLE) > 0.02 AS above_threshold "
            "FROM user_predictions GROUP BY __branch_id"
        ),
    },
}

BRANCH_COUNTS = [1, 2, 4, 8, 16, 32, 50]
DEFAULT_RUNS = 5
WARMUP_RUNS = 1
SHORTCIRCUIT_WORKERS = 8


# ---------------------------------------------------------------------------
# Data management: download from S3 if local files missing
# ---------------------------------------------------------------------------


def ensure_data():
    """Check local data exists; if not, download from S3 using the manifest."""
    manifest_path = LOCAL_DATA_DIR / "manifest.json"

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        all_present = all(
            (
                LOCAL_DATA_DIR / "branches" / e["variant"] / "user_predictions.parquet"
            ).exists()
            for e in manifest["branches"]
        )
        if (
            all_present
            and (LOCAL_DATA_DIR / "shared" / "ecommerce_users.parquet").exists()
        ):
            print(f"Local data OK: {len(manifest['branches'])} branches")
            return manifest

    print("Local data not found. Downloading from S3...")
    s3 = pafs.S3FileSystem()

    manifest_s3 = f"{S3_BUCKET}/{S3_PREFIX}/manifest.json"
    with s3.open_input_stream(manifest_s3) as f:
        manifest = json.loads(f.read().decode())

    LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))

    shared_dir = LOCAL_DATA_DIR / "shared"
    shared_dir.mkdir(parents=True, exist_ok=True)
    shared_s3 = f"{S3_BUCKET}/{S3_PREFIX}/shared/ecommerce_users.parquet"
    table = pq.read_table(shared_s3, filesystem=s3)
    pq.write_table(table, shared_dir / "ecommerce_users.parquet")
    print(f"  Downloaded ecommerce_users ({table.num_rows} rows)")

    for entry in manifest["branches"]:
        vname = entry["variant"]
        branch_dir = LOCAL_DATA_DIR / "branches" / vname
        branch_dir.mkdir(parents=True, exist_ok=True)
        branch_s3 = f"{S3_BUCKET}/{S3_PREFIX}/branches/{vname}/user_predictions.parquet"
        table = pq.read_table(branch_s3, filesystem=s3)
        pq.write_table(table, branch_dir / "user_predictions.parquet")
        print(f"  Downloaded {vname} ({table.num_rows} rows)")

    print(f"Downloaded {len(manifest['branches'])} branches to {LOCAL_DATA_DIR}")
    return manifest


def load_branch_names(manifest):
    """Return sorted list of variant names from manifest."""
    return sorted(e["variant"] for e in manifest["branches"])


# ---------------------------------------------------------------------------
# Engine: ad hoc (per-branch DataFusion contexts)
# ---------------------------------------------------------------------------
# Each branch gets its own SessionContext with register_parquet().
# Branches run in a ThreadPoolExecutor(max_workers=2) -- limited parallelism
# to represent a realistic resource-constrained deployment.  Each DataFusion
# context internally parallelizes across all CPU cores for its single-branch
# query.


def run_adhoc(sql: str, branch_names: list[str], shared_path: Path, branches_dir: Path):
    """Per-branch DataFusion contexts with limited parallelism."""
    from multiverse import run_adhoc_parallel

    def _build_ctx(branch):
        branch_parquet = branches_dir / branch / "user_predictions.parquet"
        ctx = datafusion.SessionContext()
        ctx.register_parquet("user_predictions", str(branch_parquet))
        ctx.register_parquet("ecommerce_users", str(shared_path))
        return ctx

    all_batches, exec_ms = run_adhoc_parallel(
        branch_names, _build_ctx, sql, max_workers=2
    )

    t0 = time.perf_counter_ns()
    combined = pa.Table.from_batches(all_batches) if all_batches else None
    concat_ns = time.perf_counter_ns() - t0

    return {
        "plan_ms": 0,
        "exec_ms": exec_ms,
        "concat_ms": round(concat_ns / 1_000_000),
        "rows": combined.num_rows if combined else 0,
    }


# ---------------------------------------------------------------------------
# Engine: native unified plan (pure Rust, no FFI)
# ---------------------------------------------------------------------------
# Uses query_native() from the multiverse_provider crate.  This function
# creates a Rust DataFusion SessionContext, registers a MultiverseTableProvider
# (one ListingTable per branch wrapped in a UnionExec with __branch_id), and
# registers shared tables as plain ListingTables -- all within a single Rust
# process.  No FFI boundary means the optimizer has full visibility:
#   - Predicate pushdown into parquet (row group pruning)
#   - Projection pushdown (only needed columns read)
#   - Shared-table reuse (CTE / hash-join build side computed once)


def run_native_multiverse(
    sql: str, branch_names: list[str], shared_path: Path, branches_dir: Path
):
    """Pure Rust DataFusion -- no FFI boundary."""
    from multiverse_provider import query_native

    t0 = time.perf_counter_ns()

    branch_paths = [
        (branch, str(branches_dir / branch / "user_predictions.parquet"))
        for branch in branch_names
    ]
    shared_tables = [("ecommerce_users", str(shared_path))]

    batches = query_native(sql, branch_paths, shared_tables)
    exec_ns = time.perf_counter_ns() - t0

    t0 = time.perf_counter_ns()
    if batches:
        combined = pa.Table.from_batches(batches)
    else:
        combined = None
    concat_ns = time.perf_counter_ns() - t0

    return {
        "plan_ms": 0,
        "exec_ms": round(exec_ns / 1_000_000),
        "concat_ms": round(concat_ns / 1_000_000),
        "rows": combined.num_rows if combined else 0,
    }


# ---------------------------------------------------------------------------
# Engine: native short-circuit (boolean early termination)
# ---------------------------------------------------------------------------
# For boolean queries, the supervaluationary verdict can be determined without
# scanning all branches: once two branches disagree (one true, one false), the
# answer is a truth glut regardless of the remaining branches.  This is an
# optimization with no classical analogue -- it exploits the non-classical
# semantics to skip work.


def run_native_shortcircuit(
    sql: str, branch_names: list[str], shared_path: Path, branches_dir: Path
):
    """Parallel per-branch boolean evaluation with early termination."""
    cancel = threading.Event()
    seen_true = threading.Event()
    seen_false = threading.Event()
    branches_evaluated = []
    lock = threading.Lock()

    t0 = time.perf_counter_ns()

    def eval_branch(branch):
        if cancel.is_set():
            return None

        branch_parquet = branches_dir / branch / "user_predictions.parquet"
        ctx = datafusion.SessionContext()
        ctx.register_parquet("user_predictions", str(branch_parquet))
        ctx.register_parquet("ecommerce_users", str(shared_path))

        if cancel.is_set():
            return None

        batches = ctx.sql(sql).collect()

        if cancel.is_set():
            return None

        result = pa.Table.from_batches(batches)
        val = bool(result.column(0)[0].as_py())

        with lock:
            branches_evaluated.append(branch)

        if val:
            seen_true.set()
        else:
            seen_false.set()

        if seen_true.is_set() and seen_false.is_set():
            cancel.set()

        return val

    n_workers = min(SHORTCIRCUIT_WORKERS, len(branch_names))
    pool = ThreadPoolExecutor(max_workers=n_workers)
    futures = [pool.submit(eval_branch, b) for b in branch_names]

    for f in as_completed(futures):
        exc = f.exception()
        if exc is not None:
            cancel.set()
            pool.shutdown(wait=True, cancel_futures=True)
            raise RuntimeError(f"shortcircuit worker failed: {exc}") from exc
        if cancel.is_set():
            break

    pool.shutdown(wait=True, cancel_futures=True)
    exec_ns = time.perf_counter_ns() - t0

    return {
        "plan_ms": 0,
        "exec_ms": round(exec_ns / 1_000_000),
        "concat_ms": 0,
        "rows": len(branches_evaluated),
    }


# ---------------------------------------------------------------------------
# Benchmark loop
# ---------------------------------------------------------------------------


def run_benchmark(
    query_id: str,
    engine: str,
    branch_names: list[str],
    shared_path: Path,
    branches_dir: Path,
    n_runs: int,
):
    """Run a single (query, engine, B) config n_runs times. Returns list of results."""
    q = QUERIES[query_id]

    if engine == "adhoc":
        sql = q["adhoc"]
        runner = run_adhoc
    elif q["type"] == "boolean":
        # Native engine uses short-circuit for boolean queries
        sql = q["adhoc"]  # per-branch SQL, no __branch_id
        runner = run_native_shortcircuit
    else:
        # Native engine uses MultiverseTable for non-boolean queries
        sql = q["native"]
        runner = run_native_multiverse

    results = []
    total_runs = WARMUP_RUNS + n_runs

    for i in range(total_runs):
        is_warmup = i < WARMUP_RUNS
        t_start = time.perf_counter_ns()
        metrics = runner(sql, branch_names, shared_path, branches_dir)
        wall_ns = time.perf_counter_ns() - t_start

        if not is_warmup:
            results.append(
                {
                    "query": query_id,
                    "engine": engine,
                    "branches": len(branch_names),
                    "run": i - WARMUP_RUNS,
                    "wall_ms": round(wall_ns / 1_000_000),
                    **metrics,
                }
            )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark ad hoc vs native multiverse engine"
    )
    parser.add_argument(
        "--query",
        choices=list(QUERIES.keys()),
        default=None,
        help="Run only this query (default: all)",
    )
    parser.add_argument(
        "--engine",
        choices=["adhoc", "native"],
        default=None,
        help="Run only this engine (default: both)",
    )
    parser.add_argument(
        "--max-branches",
        type=int,
        default=50,
        help="Max branch count to test (default: 50)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_RUNS,
        help=f"Runs per config, excluding warmup (default: {DEFAULT_RUNS})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="JSONL output file (default: stdout summary only)",
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        help="Generate publication charts in src/benchmarks/figures/",
    )
    args = parser.parse_args()

    manifest = ensure_data()
    all_branches = load_branch_names(manifest)
    shared_path = LOCAL_DATA_DIR / "shared" / "ecommerce_users.parquet"
    branches_dir = LOCAL_DATA_DIR / "branches"

    queries = [args.query] if args.query else list(QUERIES.keys())
    engines = [args.engine] if args.engine else ["adhoc", "native"]
    branch_counts = [
        b for b in BRANCH_COUNTS if b <= args.max_branches and b <= len(all_branches)
    ]

    if not branch_counts:
        print(f"Not enough branches. Have {len(all_branches)}, need at least 1.")
        sys.exit(1)

    print("\nBenchmark configuration:")
    print(f"  Queries:  {queries}")
    print(f"  Engines:  {engines}")
    print(f"  Branches: {branch_counts} (of {len(all_branches)} available)")
    print(f"  Runs:     {args.runs} + {WARMUP_RUNS} warmup")
    print()

    all_results = []
    jsonl_file = open(args.output, "w") if args.output else None

    for query_id in queries:
        for engine in engines:
            for b_count in branch_counts:
                selected = all_branches[:b_count]
                label = f"{query_id} | {engine:8s} | B={b_count:3d}"
                print(f"  Running {label} ...", end="", flush=True)

                results = run_benchmark(
                    query_id, engine, selected, shared_path, branches_dir, args.runs
                )
                all_results.extend(results)

                if jsonl_file:
                    for r in results:
                        jsonl_file.write(json.dumps(r) + "\n")

                walls = [r["wall_ms"] for r in results]
                median_ms = sorted(walls)[len(walls) // 2]
                print(
                    f"  median={median_ms:>7d}ms  min={min(walls):>7d}ms  max={max(walls):>7d}ms"
                )

    if jsonl_file:
        jsonl_file.close()
        print(f"\nResults written to {args.output}")

    # Summary table
    print("\n" + "=" * 90)
    print("SUMMARY (median wall_ms)")
    print("=" * 90)
    print(
        f"{'Query':<12} {'Engine':<8} "
        + "".join(f"{'B=' + str(b):>10}" for b in branch_counts)
    )
    print("-" * 90)

    for query_id in queries:
        for engine in engines:
            row = f"{query_id:<12} {engine:<8} "
            for b_count in branch_counts:
                matching = [
                    r["wall_ms"]
                    for r in all_results
                    if r["query"] == query_id
                    and r["engine"] == engine
                    and r["branches"] == b_count
                ]
                if matching:
                    median = sorted(matching)[len(matching) // 2]
                    row += f"{median:>9d}ms"
                else:
                    row += f"{'--':>10}"
            print(row)

    # Speedup table
    print("\n" + "=" * 90)
    print("SPEEDUP (ad hoc / native median)")
    print("=" * 90)
    print(f"{'Query':<12} " + "".join(f"{'B=' + str(b):>10}" for b in branch_counts))
    print("-" * 90)

    for query_id in queries:
        row = f"{query_id:<12} "
        for b_count in branch_counts:
            adhoc_ms = [
                r["wall_ms"]
                for r in all_results
                if r["query"] == query_id
                and r["engine"] == "adhoc"
                and r["branches"] == b_count
            ]
            native_ms = [
                r["wall_ms"]
                for r in all_results
                if r["query"] == query_id
                and r["engine"] == "native"
                and r["branches"] == b_count
            ]
            if adhoc_ms and native_ms:
                n_median = sorted(adhoc_ms)[len(adhoc_ms) // 2]
                v_median = sorted(native_ms)[len(native_ms) // 2]
                speedup = n_median / v_median if v_median > 0 else float("inf")
                row += f"{speedup:>9.1f}x"
            else:
                row += f"{'--':>10}"
        print(row)

    if args.paper:
        print("\nGenerating paper charts...")
        fig_dir = Path(__file__).parent / "figures"
        generate_paper_charts(all_results, branch_counts, fig_dir)

    # Sub-phase breakdown for the largest branch count
    max_b = branch_counts[-1]
    print(f"\n{'=' * 90}")
    print(f"SUB-PHASE BREAKDOWN (median ms, B={max_b})")
    print(f"{'=' * 90}")
    print(
        f"{'Query':<12} {'Engine':<8} {'Plan':>10} {'Exec':>10} {'Concat':>10} {'Total':>10}"
    )
    print("-" * 90)

    for query_id in queries:
        for engine in engines:
            matching = [
                r
                for r in all_results
                if r["query"] == query_id
                and r["engine"] == engine
                and r["branches"] == max_b
            ]
            if matching:
                matching.sort(key=lambda r: r["wall_ms"])
                mid = matching[len(matching) // 2]
                print(
                    f"{query_id:<12} {engine:<8} "
                    f"{mid['plan_ms']:>9d}  {mid['exec_ms']:>9d}  "
                    f"{mid['concat_ms']:>9d}  {mid['wall_ms']:>9d}"
                )


def generate_paper_charts(all_results, branch_counts, output_dir: Path):
    """Generate publication-quality charts from benchmark results."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    def median(vals):
        s = sorted(vals)
        return s[len(s) // 2]

    def get_medians(query_id, engine):
        medians = []
        for b in branch_counts:
            vals = [
                r["wall_ms"] for r in all_results
                if r["query"] == query_id and r["engine"] == engine
                and r["branches"] == b
            ]
            if not vals:
                return None
            medians.append(median(vals))
        return medians

    query_labels = {
        "q1_count": "Q1: COUNT (single table)",
        "q3_join": "Q3: COUNT with JOIN + windows",
        "q4_bool": "Q4: Boolean threshold",
    }

    # --- Chart 1: Latency scaling (all queries, both engines) ---
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=False)

    for ax, qid in zip(axes, ["q1_count", "q3_join", "q4_bool"]):
        adhoc_med = get_medians(qid, "adhoc")
        native_med = get_medians(qid, "native")

        if adhoc_med:
            ax.plot(branch_counts, adhoc_med, "o-", color="#d62728", label="Ad hoc", linewidth=2, markersize=5)
        if native_med:
            ax.plot(branch_counts, native_med, "s-", color="#1f77b4", label="Native", linewidth=2, markersize=5)

        ax.set_title(query_labels[qid], fontsize=10)
        ax.set_xlabel("Number of branches")
        ax.set_xticks(branch_counts)
        ax.grid(True, alpha=0.3)
        if ax == axes[0]:
            ax.set_ylabel("Median latency (ms)")
            ax.legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    fig.savefig(output_dir / "bench_latency.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(output_dir / "bench_latency.png", bbox_inches="tight", dpi=300)
    print(f"  Saved latency chart to {output_dir / 'bench_latency.pdf'}")
    plt.close(fig)

    # --- Chart 2: Speedup (ad hoc / native) ---
    fig, ax = plt.subplots(figsize=(5, 3.5))
    colors = {"q1_count": "#2ca02c", "q3_join": "#1f77b4", "q4_bool": "#ff7f0e"}
    markers = {"q1_count": "o", "q3_join": "s", "q4_bool": "^"}

    for qid in ["q1_count", "q3_join", "q4_bool"]:
        adhoc_med = get_medians(qid, "adhoc")
        native_med = get_medians(qid, "native")
        if adhoc_med and native_med:
            speedups = [n / v if v > 0 else 0 for n, v in zip(adhoc_med, native_med)]
            ax.plot(
                branch_counts, speedups, f"{markers[qid]}-",
                color=colors[qid], label=query_labels[qid],
                linewidth=2, markersize=5,
            )

    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.set_xlabel("Number of branches")
    ax.set_ylabel("Speedup (ad hoc / native)")
    ax.set_xticks(branch_counts)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "bench_speedup.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(output_dir / "bench_speedup.png", bbox_inches="tight", dpi=300)
    print(f"  Saved speedup chart to {output_dir / 'bench_speedup.pdf'}")
    plt.close(fig)


if __name__ == "__main__":
    main()
