"""
FastAPI backend for Query the Multiverse.

Proxies lakehouse calls, runs text-to-SQL, executes multiverse queries,
and serves the static frontend.
"""

import asyncio
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager
from pathlib import Path

import bauplan
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

# Ensure app modules are importable
sys.path.insert(0, str(Path(__file__).parent))

from lakehouse import (  # noqa: E402
    TRANSACTIONAL_BRANCH_MARKER,
    expand_run_commits,
    fetch_namespace_schemas,
    find_branches_forking_from,
    get_all_branches,
    get_branch_commits,
    resolve_branch_metadata,
)
from multiverse import NAMESPACE  # noqa: E402
from naive_multiverse import NaiveMultiverse  # noqa: E402
from native_multiverse import NativeMultiverse  # noqa: E402
from query_shape import UnsupportedQueryError, classify  # noqa: E402
from text_to_sql import nl_to_sql  # noqa: E402
from helpers import commit_label, short_branch_label, QueryRequest  # noqa: E402

logger = logging.getLogger(__name__)

DEMO_MAIN_SUFFIX = "multiverse_main"
DEMO_BRANCH_MARKER = "multiverse_v_"


# Global state set during lifespan
client = None
username = None
demo_main = None
head_commit = None


STARTUP_POLL_INTERVAL = 3
STARTUP_MAX_WAIT = 120


async def wait_for_branch(
    bpln_client,
    uname: str,
    branch_name: str,
    max_wait: float = STARTUP_MAX_WAIT,
    poll_interval: float = STARTUP_POLL_INTERVAL,
) -> bool:
    """Poll until branch_name exists or max_wait seconds elapse.

    Returns True if found, False if timed out.
    """
    deadline = time.monotonic() + max_wait
    while True:
        all_branches = get_all_branches(bpln_client, uname, exclude_main=False)
        names = {b["name"] for b in all_branches}
        if branch_name in names:
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        elapsed = int(max_wait - remaining)
        logger.info("Waiting for branch %s... (%ds)", branch_name, elapsed)
        await asyncio.sleep(min(poll_interval, remaining))


@asynccontextmanager
async def lifespan(app: FastAPI):
    global client, username, demo_main, head_commit
    client = bauplan.Client()
    username = client.info().user.username
    logger.info("Logged in as %s", username)

    demo_main = f"{username}.{DEMO_MAIN_SUFFIX}"

    found = await wait_for_branch(client, username, demo_main)
    if not found:
        logger.error(
            "Branch %s not found after %ds. Run run_demo.sh first.",
            demo_main,
            STARTUP_MAX_WAIT,
        )
        sys.exit(1)

    commits = get_branch_commits(client, demo_main, limit=1, lookback_hours=24 * 365)
    if not commits:
        logger.error("No commits on %s", demo_main)
        sys.exit(1)
    head_commit = commits[0]
    logger.info("Trunk HEAD: %s", head_commit["short_hash"])
    yield


app = FastAPI(lifespan=lifespan)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/status")
async def status():
    return {
        "username": username,
        "demo_main": demo_main,
        "head": {
            "hash": head_commit["hash"],
            "short_hash": head_commit["short_hash"],
            "message": head_commit["message"][:80],
        },
    }


@app.get("/api/branches")
async def branches():
    t0 = time.perf_counter()
    all_br = get_all_branches(client, username)

    # Find demo branches: either they have unique commits (forking) or they
    # were just created from trunk and have no commits yet (hash == trunk HEAD).
    demo_br = [b for b in all_br if DEMO_BRANCH_MARKER in b["name"]]
    forking = find_branches_forking_from(client, head_commit["hash"], demo_br)
    forking_names = {b["name"] for b in forking}

    # Include branches that sit exactly on trunk HEAD (no unique commits yet —
    # pipeline is still running on a tx branch).
    pending = []
    for b in demo_br:
        if b["name"] not in forking_names and b["hash"] == head_commit["hash"]:
            forking.append(b)
            pending.append(b["name"])
    if pending:
        logger.info("Pending branches (no commits yet): %s", pending)

    forking.sort(key=lambda b: b["name"])

    result = []
    if not forking:
        return {"trunk": head_commit["short_hash"], "branches": result}

    # Also fetch tx branches so we can show in-progress pipeline commits
    all_br_with_tx = get_all_branches(client, username, include_tx=True)
    tx_branches = {
        b["name"]: b for b in all_br_with_tx
        if TRANSACTIONAL_BRANCH_MARKER in b["name"]
    }
    if tx_branches:
        logger.info("Active tx branches: %s", list(tx_branches.keys()))

    commits_by_branch = _fetch_branch_commits(forking, head_commit["hash"])

    # For each DAG branch, also fetch commits from any active tx branch
    tx_commits_by_dag = _fetch_tx_commits(forking, tx_branches)
    if tx_commits_by_dag:
        logger.info("TX commits found for: %s", {
            k: len(v) for k, v in tx_commits_by_dag.items()
        })

    for b in forking:
        name = b["name"]
        label = short_branch_label(name)
        # Prefer expanded DAG branch commits; fall back to tx branch commits
        # if the DAG branch has no commits yet (pipeline still running)
        raw_commits = commits_by_branch.get(name, [])
        if not raw_commits:
            raw_commits = tx_commits_by_dag.get(name, [])
        commits_out = _format_commits(raw_commits)
        result.append(
            {
                "name": name,
                "label": label,
                "commits": commits_out,
            }
        )

    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info("/api/branches: %d branches, %.0fms", len(result), elapsed_ms)
    return {"trunk": head_commit["short_hash"], "branches": result}


def _format_commits(raw_commits):
    """Deduplicate by table and format commit dicts for the API response."""
    commits_out = []
    seen_tables = set()
    for c in raw_commits:
        tbl = commit_label(c["message"])
        if tbl and tbl in seen_tables:
            continue
        if tbl:
            seen_tables.add(tbl)
        date_str = ""
        if c.get("date"):
            try:
                date_str = c["date"].strftime("%H:%M:%S")
            except Exception:
                date_str = str(c["date"])
        commits_out.append(
            {
                "hash": c["short_hash"],
                "label": tbl or c["short_hash"],
                "message": c["message"],
                "date": date_str,
            }
        )
    return commits_out


MAX_COMMIT_PAGES = 4
COMMITS_PER_PAGE = 50


def _fetch_branch_commits(forking_branches, trunk_head_hash):
    def _fetch(name):
        # Paginate to find the fork hash, up to MAX_COMMIT_PAGES pages.
        unique = []
        found_fork = False
        for page in range(MAX_COMMIT_PAGES):
            raw = (
                get_branch_commits(
                    client, name,
                    limit=COMMITS_PER_PAGE * (page + 1),
                    lookback_hours=24 * 365,
                    exclude_internal_authors=False,
                )
                or []
            )
            # Skip commits we already processed in earlier pages.
            start = len(unique)
            for c in raw[start:]:
                if c["hash"] == trunk_head_hash:
                    found_fork = True
                    break
                unique.append(c)
            if found_fork or len(raw) < COMMITS_PER_PAGE * (page + 1):
                break
        if not found_fork:
            unique = []
        # unique is newest-first from the API; reverse to oldest-first so
        # expand_run_commits produces a consistent oldest-first result
        # (bronze -> silver -> gold).
        unique.reverse()
        expanded = expand_run_commits(client, name, unique, trunk_head_hash)
        return name, expanded

    cache = {}
    with ThreadPoolExecutor(max_workers=min(10, len(forking_branches) or 1)) as pool:
        futures = {pool.submit(_fetch, b["name"]): b["name"] for b in forking_branches}
        for future in as_completed(futures):
            name, commits = future.result()
            cache[name] = commits
    return cache


def _fetch_tx_commits(forking_branches, tx_branches):
    """For each DAG branch, find active tx branches and fetch their commits.

    Tx branches are named ``<dag_branch>-bpln-tx-run-<job_id>``.  Their
    commits show individual table materializations in progress, giving the
    UI real-time progression instead of waiting for the merge.
    """
    # Map each DAG branch to its tx branches
    dag_to_tx = {}
    for dag_b in forking_branches:
        dag_name = dag_b["name"]
        prefix = dag_name + TRANSACTIONAL_BRANCH_MARKER
        matching = [
            name for name in tx_branches if name.startswith(prefix)
        ]
        if matching:
            # Pick the most recent tx branch (last alphabetically = latest job)
            dag_to_tx[dag_name] = sorted(matching)[-1]

    if not dag_to_tx:
        return {}

    def _fetch_tx(dag_name, tx_name):
        raw = (
            get_branch_commits(
                client, tx_name,
                limit=20,
                lookback_hours=24 * 365,
                exclude_internal_authors=False,
            )
            or []
        )
        # Only keep model materialization commits from our namespace
        model_commits = [
            c for c in raw
            if "ICEBERG_TABLE" in c["message"] and NAMESPACE in c["message"]
        ]
        model_commits.reverse()  # oldest first
        return dag_name, model_commits

    result = {}
    with ThreadPoolExecutor(max_workers=min(10, len(dag_to_tx) or 1)) as pool:
        futures = {
            pool.submit(_fetch_tx, dag, tx): dag
            for dag, tx in dag_to_tx.items()
        }
        for future in as_completed(futures):
            dag_name, commits = future.result()
            if commits:
                result[dag_name] = commits
    return result


@app.post("/api/query")
async def query(req: QueryRequest):
    all_br = get_all_branches(client, username)
    forking = find_branches_forking_from(client, head_commit["hash"], all_br)
    forking = [b for b in forking if DEMO_BRANCH_MARKER in b["name"]]
    if not forking:
        return {"error": "No branches available to query"}

    branch_names = [b["name"] for b in forking]

    try:
        schemas = fetch_namespace_schemas(client, branch_names[0], NAMESPACE)
    except Exception as e:
        return {"error": f"Failed to fetch schemas: {e}"}

    try:
        sql, cache_hit = nl_to_sql(
            req.question, schemas, use_cache=req.use_cache, engine=req.engine
        )
    except Exception as e:
        return {"error": f"Failed to generate SQL: {e}"}

    try:
        shape = classify(sql, schemas=schemas, strict=(req.engine == "adhoc"), engine=req.engine)
        result_type = shape.result_type.value
    except UnsupportedQueryError as e:
        return {"error": e.reason, "hint": e.hint, "sql": sql}

    try:
        metadata = resolve_branch_metadata(client, branch_names, NAMESPACE)
        if not metadata:
            return {
                "error": "No branches have finished running their pipelines yet.",
                "sql": sql,
            }
        ready_branches = list(metadata.keys())
    except Exception as e:
        return {"error": f"Query execution failed: {e}", "sql": sql}

    try:
        if req.engine == "native":
            engine = NativeMultiverse(metadata)
        else:
            engine = NaiveMultiverse(metadata)
        combined, errors, meta = engine.query_multiverse(
            sql, ready_branches, shape=shape
        )
    except Exception as e:
        return {"error": f"Query execution failed: {e}", "sql": sql}

    response = {
        "sql": sql,
        "engine": req.engine,
        "result_type": result_type,
        "errors": errors,
        "cache_hit": cache_hit,
    }

    if engine.stats.result_bytes > 0:
        response["stats"] = {
            "result_bytes": engine.stats.result_bytes,
            "branches_queried": engine.stats.branches_queried,
        }

    if meta is not None:
        response["meta"] = {
            "kind": meta.kind,
            "summary": meta.summary,
            "per_branch": _serialize_per_branch(meta.per_branch),
            "details": _serialize_details(meta.details),
        }

    return response


def _serialize_per_branch(per_branch):
    out = {}
    for k, v in per_branch.items():
        label = short_branch_label(k)
        if isinstance(v, set):
            out[label] = sorted(v)[:100]
        else:
            out[label] = v
    return out


def _serialize_details(details):
    out = {}
    for k, v in details.items():
        if isinstance(v, set):
            out[k] = sorted(v)[:100]
        elif isinstance(v, dict):
            out[k] = {
                short_branch_label(dk): (
                    sorted(dv)[:100] if isinstance(dv, set) else dv
                )
                for dk, dv in v.items()
            }
        else:
            out[k] = v
    return out
