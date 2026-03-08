"""
Ad hoc multiverse engine — DataFusion per-branch execution.

Registers Iceberg tables in DataFusion as lazy PyArrow datasets backed by
Parquet files on S3.  For each branch a fresh DataFusion context is created
with the tables registered under their *original* names — so the user SQL
runs unmodified.  Results are tagged with ``__branch_id`` and concatenated.
"""

import logging
import time

import datafusion
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.fs as pafs

from multiverse import (
    BranchMetadata,
    MultiverseEngine,
    QueryResult,
    get_parquet_files,
    parse_operator_bytes,
    run_adhoc_parallel,
)

logger = logging.getLogger(__name__)


class NaiveMultiverse(MultiverseEngine):
    """Ad hoc engine: execute SQL across branches using per-branch DataFusion contexts.

    For each branch a fresh SessionContext is created with the Iceberg tables
    registered under their *original* names.  The user SQL is executed as-is
    (no rewriting), results are tagged with ``__branch_id``, and concatenated
    via PyArrow.
    """

    def __init__(self, metadata_by_branch: BranchMetadata) -> None:
        super().__init__()
        self._metadata_by_branch = metadata_by_branch
        self._s3_fs: pafs.S3FileSystem | None = None

    def _get_s3_fs(self) -> pafs.S3FileSystem:
        if self._s3_fs is None:
            self._s3_fs = pafs.S3FileSystem()
        return self._s3_fs

    def _build_branch_context(self, branch: str) -> datafusion.SessionContext:
        """Build a DataFusion context with all tables for a single branch."""
        tables = self._metadata_by_branch[branch]
        s3_fs = self._get_s3_fs()
        ctx = datafusion.SessionContext()
        for table_name, metadata_location in tables.items():
            file_paths = get_parquet_files(metadata_location)
            stripped = [p.replace("s3://", "", 1) for p in file_paths]
            dataset = ds.dataset(stripped, format="parquet", filesystem=s3_fs)
            ctx.register_dataset(table_name, dataset)
        return ctx

    def _query(self, sql: str, branches: list[str]) -> QueryResult:
        """Execute SQL across all branches in parallel."""
        if not self._metadata_by_branch:
            return None, {}

        all_batches, exec_ms = run_adhoc_parallel(
            branches, self._build_branch_context, sql
        )

        if not all_batches:
            return None, {}

        t0 = time.perf_counter_ns()
        combined = pa.Table.from_batches(all_batches)
        concat_ns = time.perf_counter_ns() - t0

        if combined.num_rows == 0:
            return None, {}

        self.stats.exec_ms = exec_ms
        self.stats.concat_ms = round(concat_ns / 1_000_000)

        return combined, {}

    def _explain_analyze(
        self, sql: str, branches: list[str]
    ) -> tuple[int, dict[str, int]]:
        """Re-execute each branch with EXPLAIN ANALYZE to collect operator bytes."""
        total = 0
        per_branch: dict[str, int] = {}

        for branch in branches:
            ctx = self._build_branch_context(branch)
            branch_sql = f"SELECT *, '{branch}' AS __branch_id FROM ({sql})"
            try:
                analyze_df = ctx.sql(f"EXPLAIN ANALYZE {branch_sql}")
                branch_bytes = 0
                for batch in analyze_df.collect():
                    for plan_text in batch.to_pydict().get("plan", []):
                        branch_bytes += parse_operator_bytes(plan_text)
                per_branch[branch] = branch_bytes
                total += branch_bytes
            except Exception:
                logger.debug("EXPLAIN ANALYZE failed for %s", branch)
                per_branch[branch] = 0

        return total, per_branch
