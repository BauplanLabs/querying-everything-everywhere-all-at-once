"""Integration tests — query real S3 Parquet files with both engines.

Reads Parquet files directly from S3 using paths built from
S3_TEST_READ_BUCKET and S3_PREFIX (loaded from the project .env).

The test data is uploaded by:
  uv run python src/playground/make_multiverse_parquet.py

Data layout (MULTIVERSE_PARQUET_N_ROWS per branch, default 1M, deterministic):
  branch_1: 1M rows, 1 predicted_label=1 (at index 7)
  branch_2: 1M rows, 0 predicted_label=1
  branch_3: 1M rows, 1 predicted_label=1 (at index 13)

Requires:
  S3_TEST_READ_BUCKET  — bucket name (e.g. "alpha-hello-bauplan")
  S3_PREFIX            — key prefix  (e.g. "multiverse_test/user_predictions")

Both engines authenticate via AWS credentials from the environment
(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION).
"""

import os
from pathlib import Path

import datafusion
import pyarrow as pa
import pyarrow.dataset as pads
import pyarrow.fs as pafs
import pytest
from dotenv import load_dotenv

from multiverse_provider import S3MultiverseTable

# Load .env from project root
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

BUCKET = os.environ.get("S3_TEST_READ_BUCKET", "")
PREFIX = os.environ.get("S3_PREFIX", "").strip("/")

BRANCHES = ["branch_1", "branch_2", "branch_3"]
N_ROWS = int(os.environ.get("MULTIVERSE_PARQUET_N_ROWS", "1_000_000"))


def _s3_path(branch: str) -> str:
    """Full s3:// URI for a branch's parquet file."""
    return f"s3://{BUCKET}/{PREFIX}/{branch}/user_predictions.parquet"


def _bare_path(branch: str) -> str:
    """S3 path without the s3:// prefix (for PyArrow S3FileSystem)."""
    return f"{BUCKET}/{PREFIX}/{branch}/user_predictions.parquet"


skip_no_s3 = pytest.mark.skipif(
    not BUCKET or not PREFIX,
    reason="S3_TEST_READ_BUCKET and S3_PREFIX not set in .env",
)


# -- Helpers to build engine contexts --


def _build_naive_s3_fs():
    """Build a shared S3 filesystem for naive engine tests."""
    region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    return pafs.S3FileSystem(region=region)


def _naive_branch_ctx(branch: str, s3_fs=None):
    """Build a DataFusion context for a single branch (tables under original names)."""
    if s3_fs is None:
        s3_fs = _build_naive_s3_fs()
    ctx = datafusion.SessionContext()
    dataset = pads.dataset(
        _bare_path(branch), format="parquet", filesystem=s3_fs
    )
    ctx.register_dataset("user_predictions", dataset)
    return ctx


def _build_native_ctx():
    """Build a DataFusion context with a S3MultiverseTable."""
    branch_files = [(branch, [_s3_path(branch)]) for branch in BRANCHES]
    mv_table = S3MultiverseTable(branch_files)
    ctx = datafusion.SessionContext()
    ctx.register_table("user_predictions", mv_table)
    return ctx


def _naive_query(sql: str, branches: list[str] | None = None) -> pa.Table:
    """Run a query per branch and concatenate results (mirrors NaiveMultiverse)."""
    if branches is None:
        branches = BRANCHES
    s3_fs = _build_naive_s3_fs()
    all_batches: list[pa.RecordBatch] = []
    for branch in branches:
        ctx = _naive_branch_ctx(branch, s3_fs)
        branch_sql = f"SELECT *, '{branch}' AS __branch_id FROM ({sql})"
        all_batches.extend(ctx.sql(branch_sql).collect())
    return pa.Table.from_batches(all_batches)


def _native_query(sql: str, ctx) -> pa.Table:
    """Run a query through the native engine and return a result."""
    return pa.Table.from_batches(ctx.sql(sql).collect())


# -- Cross-engine agreement tests --


@skip_no_s3
class TestEngineAgreement:
    """Both engines must return identical per-branch results."""

    @pytest.fixture()
    def native(self):
        return _build_native_ctx()

    def test_count_per_branch(self, native):
        sql = "SELECT COUNT(*) AS n FROM user_predictions"
        naive_result = _naive_query(sql)
        native_result = _native_query(
            "SELECT __branch_id, COUNT(*) AS n FROM user_predictions "
            "GROUP BY __branch_id ORDER BY __branch_id",
            native,
        )

        naive_counts = dict(
            zip(
                naive_result.column("__branch_id").to_pylist(),
                naive_result.column("n").to_pylist(),
            )
        )
        native_counts = dict(
            zip(
                native_result.column("__branch_id").to_pylist(),
                native_result.column("n").to_pylist(),
            )
        )
        assert naive_counts == native_counts
        assert naive_counts == {
            "branch_1": N_ROWS,
            "branch_2": N_ROWS,
            "branch_3": N_ROWS,
        }

    def test_sum_predicted_label_per_branch(self, native):
        sql = "SELECT SUM(predicted_label) AS total FROM user_predictions"
        naive_result = _naive_query(sql)
        native_result = _native_query(
            "SELECT __branch_id, SUM(predicted_label) AS total FROM user_predictions "
            "GROUP BY __branch_id ORDER BY __branch_id",
            native,
        )

        naive_sums = dict(
            zip(
                naive_result.column("__branch_id").to_pylist(),
                naive_result.column("total").to_pylist(),
            )
        )
        native_sums = dict(
            zip(
                native_result.column("__branch_id").to_pylist(),
                native_result.column("total").to_pylist(),
            )
        )
        assert naive_sums == native_sums
        assert naive_sums == {"branch_1": 1, "branch_2": 0, "branch_3": 1}

    def test_filter_predicted_label(self, native):
        sql = "SELECT * FROM user_predictions WHERE predicted_label = 1"
        naive_result = _naive_query(sql)
        native_result = _native_query(sql, native)

        # Both return 2 rows total (branch_1: 1, branch_3: 1)
        assert naive_result.num_rows == 2
        assert native_result.num_rows == 2

        naive_branches = set(naive_result.column("__branch_id").to_pylist())
        native_branches = set(native_result.column("__branch_id").to_pylist())
        assert naive_branches == native_branches == {"branch_1", "branch_3"}


# -- Naive-only tests --


@skip_no_s3
class TestNaiveS3:
    """Test NaiveMultiverse with S3 datasets directly."""

    def test_total_rows(self):
        sql = "SELECT COUNT(*) AS n FROM user_predictions"
        result = _naive_query(sql)
        total = sum(result.column("n").to_pylist())
        assert total == N_ROWS * 3

    def test_two_branches_only(self):
        result = _naive_query(
            "SELECT COUNT(*) AS n FROM user_predictions",
            branches=["branch_1", "branch_3"],
        )
        assert result.num_rows == 2
        branch_ids = set(result.column("__branch_id").to_pylist())
        assert branch_ids == {"branch_1", "branch_3"}


# -- Native-only tests --


@skip_no_s3
class TestNativeRustS3:
    """Test S3MultiverseTable against real S3 Parquet files."""

    @pytest.fixture()
    def ctx(self):
        return _build_native_ctx()

    def test_cte(self, ctx):
        """CTEs work because the native provider needs no SQL rewriting."""
        result = _native_query(
            "WITH sums AS ("
            "  SELECT __branch_id, SUM(predicted_label) AS total"
            "  FROM user_predictions GROUP BY __branch_id"
            ") SELECT * FROM sums WHERE total > 0 ORDER BY __branch_id",
            ctx,
        )
        d = result.to_pydict()
        assert d["__branch_id"] == ["branch_1", "branch_3"]
        assert d["total"] == [1, 1]

    def test_two_branches_only(self):
        """Register only 2 branches and verify results."""
        branch_files = [
            ("branch_1", [_s3_path("branch_1")]),
            ("branch_2", [_s3_path("branch_2")]),
        ]
        mv_table = S3MultiverseTable(branch_files)
        ctx = datafusion.SessionContext()
        ctx.register_table("user_predictions", mv_table)
        result = pa.Table.from_batches(
            ctx.sql("SELECT COUNT(*) AS n FROM user_predictions").collect()
        )
        assert result.num_rows == 1
        assert result.column("n").to_pylist() == [N_ROWS * 2]

    def test_schema_columns(self, ctx):
        result = _native_query(
            "SELECT * FROM user_predictions LIMIT 1", ctx
        )
        assert set(result.column_names) == {
            "user_id",
            "conversion_prob",
            "predicted_label",
            "__branch_id",
        }
