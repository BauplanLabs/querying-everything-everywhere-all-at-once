"""
Bauplan lakehouse navigation — commit history, branches, tags, schemas.

Pure SDK logic, no Streamlit dependency.
"""

import logging
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol, runtime_checkable

import pyarrow as pa
from bauplan.exceptions import BauplanError, TableNotFoundError
from pyiceberg.io.pyarrow import schema_to_pyarrow
from pyiceberg.table import StaticTable
from pyiceberg.types import IcebergType

from helpers import BranchMetadata

logger = logging.getLogger(__name__)

DEFAULT_LOOKBACK_HOURS: int = 24

EXCLUDED_BRANCH_PREFIX: str = "bauplan-e2e-check-bauplan-prod"
EXCLUDED_AUTHOR_SUBSTRING: str = "bauplan"
TRANSACTIONAL_BRANCH_MARKER: str = "-bpln-tx-run-"

# Type alias for the commit dicts returned by get_*_commits
CommitDict = dict[str, Any]
BranchDict = dict[str, str]


@runtime_checkable
class BauplanClient(Protocol):
    """Structural type for the bauplan client methods used by this module."""

    def get_commits(self, *, ref: str, **kwargs: Any) -> list: ...
    def get_branches(self, *, user: str) -> list: ...
    def get_tables(self, ref: str, *, filter_by_namespace: str | None = None, **kwargs: Any) -> list: ...
    def get_table(self, name: str, *, ref: str, namespace: str) -> Any: ...
    def query(self, sql: str, *, ref: str, namespace: str) -> pa.Table: ...


def _is_excluded_branch(name: str) -> bool:
    return name.startswith(EXCLUDED_BRANCH_PREFIX) or TRANSACTIONAL_BRANCH_MARKER in name


def _is_excluded_author(author: str) -> bool:
    return EXCLUDED_AUTHOR_SUBSTRING in author.lower()


def _recent_cutoff(hours: int = DEFAULT_LOOKBACK_HOURS) -> datetime:
    """Return a timezone-aware datetime `hours` ago from now."""
    return datetime.now(timezone.utc) - timedelta(hours=hours)


def _commit_to_dict(c: object) -> CommitDict:
    """Convert a bauplan Commit object to a plain dict."""
    return {
        "hash": c.ref.hash,  # type: ignore[attr-defined]
        "short_hash": c.ref.hash[:10],  # type: ignore[attr-defined]
        "message": c.message,  # type: ignore[attr-defined]
        "author": c.authors[0].name if c.authors else "unknown",  # type: ignore[attr-defined]
        "date": c.authored_date,  # type: ignore[attr-defined]
        "parent_hash": c.parent_ref.hash if c.parent_ref else None,  # type: ignore[attr-defined]
        "parent_hashes": list(c.parent_hashes) if c.parent_hashes else [],  # type: ignore[attr-defined]
    }


def _fetch_commits(
    client: BauplanClient, ref: str, limit: int, lookback_hours: int
) -> list[CommitDict]:
    """Fetch commits within the lookback window, filtering out internal authors."""
    commits = client.get_commits(
        ref=ref,
        limit=limit,
        filter_by_authored_date_start_at=_recent_cutoff(lookback_hours),
    )
    return [
        d
        for c in commits
        if not _is_excluded_author((d := _commit_to_dict(c))["author"])
    ]



def get_branch_commits(
    client: BauplanClient,
    branch_name: str,
    limit: int = 10,
    lookback_hours: int = DEFAULT_LOOKBACK_HOURS,
    exclude_internal_authors: bool = True,
) -> list[CommitDict] | None:
    """Return recent commits on a branch as dicts.

    Returns ``None`` on API errors so callers can distinguish "no recent
    commits" (empty list) from "branch fetch failed" (``None``).

    Set ``exclude_internal_authors=False`` to include commits authored by
    bauplan internals (e.g. pipeline materialization commits).
    """
    try:
        commits = client.get_commits(
            ref=branch_name,
            limit=limit,
            filter_by_authored_date_start_at=_recent_cutoff(lookback_hours),
        )
        result = []
        for c in commits:
            d = _commit_to_dict(c)
            if exclude_internal_authors and _is_excluded_author(d["author"]):
                continue
            result.append(d)
        return result
    except BauplanError as e:
        logger.warning("Failed to fetch commits for %s: %s", branch_name, e)
        return None


def get_all_branches(
    client: BauplanClient, username: str, exclude_main: bool = True,
    include_tx: bool = False,
) -> list[BranchDict]:
    """Return branches for ``username`` as list of {name, hash} dicts.

    Set ``include_tx=True`` to include transactional pipeline branches
    (normally filtered out).
    """
    branches = client.get_branches(user=username)
    result: list[BranchDict] = []
    for b in branches:
        if b.name.startswith(EXCLUDED_BRANCH_PREFIX):
            continue
        if not include_tx and TRANSACTIONAL_BRANCH_MARKER in b.name:
            continue
        result.append({"name": b.name, "hash": b.hash})
    if exclude_main:
        result = [b for b in result if b["name"] != "main"]
    return result


def expand_run_commits(
    client: BauplanClient,
    branch_name: str,
    commits: list[CommitDict],
    trunk_head_hash: str | None = None,
) -> list[CommitDict]:
    """Expand DAG-run merge commits into individual model materialization commits.

    Each successful bauplan DAG run is transactional: models are materialized
    on an ephemeral branch, then merged back.  The merge commit
    ("Run job_id=...") has two parents — the second parent leads to the
    individual model commits ("Update ICEBERG_TABLE ...").

    This replaces each merge commit with those granular model commits so the
    branch graph shows *what* each pipeline run actually did.

    Only commits that happened *after* the trunk fork point are considered.
    Inherited main history is ignored entirely.
    """
    # Build hash → ~N position from the *unfiltered* first-parent history.
    # The filtered `commits` list may skip entries, so indices don't match ~N.
    try:
        full_history = client.get_commits(ref=branch_name, limit=100)
    except BauplanError:
        return commits  # can't expand without the full history
    hash_to_idx: dict[str, int] = {c.ref.hash: i for i, c in enumerate(full_history)}

    # If we know the trunk HEAD, filter commits to only those after the fork.
    if trunk_head_hash:
        commits = [c for c in commits if c["hash"] != trunk_head_hash]

    expanded: list[CommitDict] = []
    for commit in commits:
        ph = commit.get("parent_hashes", [])
        if len(ph) == 2 and commit["message"].startswith("Run job_id="):
            merge_parent = ph[0]
            idx = hash_to_idx.get(commit["hash"])
            if idx is None:
                expanded.append(commit)
                continue
            # branch^2 for HEAD, branch~1^2 for HEAD~1, etc.
            tilde = f"~{idx}" if idx > 0 else ""
            ref = f"{branch_name}{tilde}^2"
            try:
                raw = client.get_commits(ref=ref, limit=50)
                model_commits: list[CommitDict] = []
                for rc in raw:
                    d = _commit_to_dict(rc)
                    if d["hash"] == merge_parent:
                        break
                    model_commits.append(d)
                if model_commits:
                    model_commits.reverse()  # oldest first (DAG execution order)
                    expanded.extend(model_commits)
                    continue
            except BauplanError:
                logger.warning(
                    "Failed to expand run commit %s via %s", commit["short_hash"], ref
                )
            expanded.append(commit)  # fallback: keep original merge commit
        else:
            expanded.append(commit)
    return expanded


def find_branches_forking_from(
    client: BauplanClient, commit_hash: str, branches: list[BranchDict]
) -> list[BranchDict]:
    """Find which branches were created from ``commit_hash``.

    A branch forks from a commit when its *earliest unique* commit (the
    one that is on the branch but not on main) has ``commit_hash`` as its
    parent.  We detect this by querying main for commits with the same
    parent and then, for each branch, checking whether any of its
    commits with that parent are *not* on main.
    """
    # Commits on main whose parent is commit_hash — these are NOT forks.
    main_children: set[str] = {
        c.ref.hash
        for c in client.get_commits(
            ref="main",
            filter_by_parent_hash=commit_hash,
            limit=50,
        )
    }

    def _check_branch(b: BranchDict) -> BranchDict | None:
        try:
            hits = client.get_commits(
                ref=b["name"],
                filter_by_parent_hash=commit_hash,
                limit=5,
            )
            branch_only = [h for h in hits if h.ref.hash not in main_children]
            return b if branch_only else None
        except BauplanError as e:
            logger.warning("Failed to check branch %s: %s", b["name"], e)
            return None

    forking: list[BranchDict] = []
    total = len(branches)
    with ThreadPoolExecutor(max_workers=min(10, total or 1)) as pool:
        futures: dict[Future[BranchDict | None], BranchDict] = {
            pool.submit(_check_branch, b): b for b in branches
        }
        for i, future in enumerate(as_completed(futures), 1):
            b = futures[future]
            logger.debug("Checked branch %d/%d: %s", i, total, b["name"])
            result = future.result()
            if result:
                logger.debug("  -> forks from %s", commit_hash[:10])
                forking.append(result)
    return forking


# ======== Iceberg I/O utilities ========


def get_metadata_location(
    client: BauplanClient, table_name: str, branch: str, namespace: str
) -> str:
    """Return the S3 metadata location for a table on a given branch."""
    iceberg_table = client.get_table(table_name, ref=branch, namespace=namespace)
    return iceberg_table.metadata_location



SHARED_NAMESPACE = "bauplan"
SHARED_TABLES = ["ecommerce_users", "ecommerce_sessions"]


def resolve_branch_metadata(
    client: BauplanClient,
    branches: list[str],
    namespace: str,
) -> BranchMetadata:
    """Resolve Iceberg metadata locations for every table across branches.

    Discovers table names from the first branch, then fetches the metadata
    location (S3 URI) for each branch × table.  Also includes shared tables
    from the bauplan namespace (same across all branches).

    Returns:
        metadata_by_branch mapping branch -> {table_name -> metadata_location}.
    """
    if not branches:
        return {}

    table_names = [
        t.name for t in client.get_tables(branches[0], filter_by_namespace=namespace)
    ]

    # Resolve shared tables once (they're identical across branches)
    shared_meta: dict[str, str] = {}
    for table_name in SHARED_TABLES:
        try:
            shared_meta[table_name] = get_metadata_location(
                client, table_name, branches[0], SHARED_NAMESPACE
            )
        except Exception:
            logger.warning("Shared table %s.%s not found", SHARED_NAMESPACE, table_name)

    metadata_by_branch: BranchMetadata = {}
    for branch in branches:
        branch_meta: dict[str, str] = {}
        try:
            for table_name in table_names:
                branch_meta[table_name] = get_metadata_location(
                    client, table_name, branch, namespace
                )
        except TableNotFoundError:
            logger.warning(
                "Branch %s missing table %s, skipping", branch, table_name
            )
            continue
        # Add shared tables (same metadata for every branch)
        branch_meta.update(shared_meta)
        metadata_by_branch[branch] = branch_meta

    return metadata_by_branch


# ======== Schema utilities ========


def bauplan_field_to_arrow(type_str: str) -> pa.DataType:
    """Map a bauplan/Iceberg field type string to a PyArrow DataType.

    Delegates to PyIceberg's type system for the canonical Iceberg-to-Arrow mapping.
    """
    iceberg_type = IcebergType.model_validate(type_str.strip().lower())
    return schema_to_pyarrow(iceberg_type, include_field_ids=False)


def bauplan_table_to_arrow_schema(table_with_metadata: object) -> pa.Schema:
    """Convert a bauplan TableWithMetadata to a pa.Schema.

    Args:
        table_with_metadata: bauplan table object with a .fields attribute,
            where each field has .name and .type properties.

    Returns:
        pa.Schema with the corresponding Arrow types.
    """
    fields: list[pa.Field] = []
    for field in table_with_metadata.fields:  # type: ignore[attr-defined]
        arrow_type = bauplan_field_to_arrow(field.type)
        fields.append(pa.field(field.name, arrow_type))
    return pa.schema(fields)


def fetch_namespace_schemas(
    client: BauplanClient, ref: str, namespace: str
) -> dict[str, pa.Schema]:
    """Fetch Arrow schemas for all tables in a bauplan namespace.

    Lists tables, then fetches full metadata (with fields) in parallel.
    Also includes shared tables from the bauplan namespace.
    """
    tables = client.get_tables(ref, filter_by_namespace=namespace)
    table_names: list[str] = [t.name for t in tables]

    # Include shared tables from the bauplan namespace
    fetch_tasks: list[tuple[str, str]] = [(n, namespace) for n in table_names]
    shared_names: set[str] = set()
    for shared_name in SHARED_TABLES:
        if shared_name not in table_names:
            fetch_tasks.append((shared_name, SHARED_NAMESPACE))
            shared_names.add(shared_name)

    def _fetch_one(name: str, ns: str) -> tuple[str, pa.Schema]:
        meta = client.get_table(name, ref=ref, namespace=ns)
        return name, bauplan_table_to_arrow_schema(meta)

    schemas: dict[str, pa.Schema] = {}
    with ThreadPoolExecutor(max_workers=max(1, min(10, len(fetch_tasks)))) as pool:
        futures: dict[Future[tuple[str, pa.Schema]], str] = {
            pool.submit(_fetch_one, n, ns): n for n, ns in fetch_tasks
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                _, schema = future.result()
                schemas[name] = schema
            except Exception:
                if name in shared_names:
                    logger.warning("Shared table %s unavailable, skipping", name)
                else:
                    raise
    return dict(sorted(schemas.items()))
