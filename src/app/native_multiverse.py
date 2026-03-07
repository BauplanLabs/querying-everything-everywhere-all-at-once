"""
Native DataFusion multiverse engine — Rust TableProvider via PyCapsule FFI.

Registers a single logical table backed by a custom Rust TableProvider that
unions all branches internally.  DataFusion sees ONE table; the multiverse
expansion (adding ``__branch_id`` and unioning branches) happens inside the
provider's ``scan()`` implementation.  Arbitrary SQL — CTEs, subqueries,
joins, window functions — just works because no SQL rewriting is needed.

The inner providers are MemTables populated by eagerly reading S3 Parquet
files at construction time.  (The datafusion-ffi bridge creates a fresh
SessionContext at scan time that lacks registered object stores, so lazy
ListingTable reads don't work through FFI.)
"""

import logging
import os
import time
from urllib.parse import urlparse

import datafusion
import pyarrow as pa
from datafusion.object_store import AmazonS3
from multiverse_provider import S3MultiverseTable

from multiverse import (
    BranchMetadata,
    MultiverseEngine,
    QueryResult,
    get_parquet_files,
    parse_operator_bytes,
)

logger = logging.getLogger(__name__)

# Only these tables get the multiverse treatment (one per branch).
# Everything else is a shared table registered once from any branch.
MULTIVERSE_TABLES = {"user_predictions"}



def _ensure_s3_store(
    ctx: datafusion.SessionContext,
    s3_url: str,
    registered: set[str],
) -> None:
    """Register an S3 object store for the bucket in s3_url if not already done."""
    parsed = urlparse(s3_url)
    bucket = parsed.netloc
    if bucket in registered:
        return
    region = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
    store = AmazonS3(bucket_name=bucket, region=region)
    ctx.register_object_store("s3://", store, host=bucket)
    registered.add(bucket)


class NativeMultiverse(MultiverseEngine):
    """Execute SQL across branches using a native Rust TableProvider.

    S3 Parquet files are read eagerly at construction time into MemTables.
    """

    def __init__(self, metadata_by_branch: BranchMetadata) -> None:
        super().__init__()
        self._metadata_by_branch = metadata_by_branch

    def _build_context(self, branches: list[str]) -> datafusion.SessionContext | None:
        """Build a DataFusion context with native multiverse tables registered.

        Only tables in MULTIVERSE_TABLES are wrapped in a MultiverseTable
        (one partition per branch). Shared tables (e.g. ecommerce_users) are
        registered once from the first branch. Intermediate pipeline tables
        (e.g. bronze_events, user_features) are skipped entirely.
        """
        if not self._metadata_by_branch:
            return None

        first_branch_meta = next(iter(self._metadata_by_branch.values()))
        table_names = list(first_branch_meta.keys())
        ctx = datafusion.SessionContext()

        # Register S3 object stores so DataFusion can read parquet directly
        registered_buckets: set[str] = set()

        for table_name in table_names:
            if table_name in MULTIVERSE_TABLES:
                # Wrap in MultiverseTable — one partition per branch
                branch_files: list[tuple[str, list[str]]] = []
                for branch in branches:
                    metadata_location = self._metadata_by_branch[branch][table_name]
                    parquet_files = get_parquet_files(metadata_location)
                    if parquet_files:
                        branch_files.append((branch, parquet_files))
                mv_table = S3MultiverseTable(branch_files)
                ctx.register_table(table_name, mv_table)
            else:
                # Shared table — register parquet directly from S3
                metadata_location = first_branch_meta[table_name]
                parquet_files = get_parquet_files(metadata_location)
                if parquet_files:
                    _ensure_s3_store(ctx, parquet_files[0], registered_buckets)
                    ctx.register_parquet(table_name, parquet_files[0])

        return ctx

    def _query(self, sql: str, branches: list[str]) -> QueryResult:
        t0 = time.perf_counter_ns()
        ctx = self._build_context(branches)
        plan_ns = time.perf_counter_ns() - t0

        if ctx is None:
            return None, {}

        # The SQL already includes __branch_id (added by the text-to-SQL
        # prompt for the native engine).  Run as a single query across
        # all branches — the MultiverseTable handles the union internally.
        t0 = time.perf_counter_ns()
        result_batches = ctx.sql(sql).collect()
        exec_ns = time.perf_counter_ns() - t0

        if not result_batches:
            return None, {}

        t0 = time.perf_counter_ns()
        combined = pa.Table.from_batches(result_batches)
        concat_ns = time.perf_counter_ns() - t0

        if combined.num_rows == 0:
            return None, {}

        self.stats.plan_ms = round(plan_ns / 1_000_000)
        self.stats.exec_ms = round(exec_ns / 1_000_000)
        self.stats.concat_ms = round(concat_ns / 1_000_000)

        return combined, {}

    def _explain_analyze(
        self, sql: str, branches: list[str]
    ) -> tuple[int, dict[str, int]]:
        """Run EXPLAIN ANALYZE on the native engine.

        The native engine uses a single context for all branches, so
        per_branch breakdown is not available — total only.
        """
        ctx = self._build_context(branches)
        if ctx is None:
            return 0, {}

        total = 0
        try:
            analyze_df = ctx.sql(f"EXPLAIN ANALYZE {sql}")
            for batch in analyze_df.collect():
                for plan_text in batch.to_pydict().get("plan", []):
                    total += parse_operator_bytes(plan_text)
        except Exception:
            logger.debug("EXPLAIN ANALYZE failed for native engine")

        return total, {}
