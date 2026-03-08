"""Tests for NativeMultiverse — Rust TableProvider engine.

Tests the MultiverseTable Rust extension directly (construction, schema,
queries via DataFusion) and the Python engine class end-to-end.
"""

from unittest.mock import patch

import datafusion
import pyarrow as pa
import pytest
from multiverse_provider import MultiverseTable

from app.native_multiverse import NativeMultiverse
from app.query_shape import QueryShape, ResultType


# -- Helpers --


def _count_table(n: int) -> pa.Table:
    return pa.table({"n": pa.array([n], type=pa.int64())})


def _predictions_table(user_ids, probs, labels) -> pa.Table:
    return pa.table(
        {
            "user_id": pa.array(user_ids, type=pa.int64()),
            "conversion_prob": pa.array(probs, type=pa.float64()),
            "predicted_label": pa.array(labels, type=pa.int32()),
        }
    )


def _make_mv_table(branch_data: dict[str, pa.Table]) -> MultiverseTable:
    """Create a MultiverseTable from {branch_id: arrow_table}."""
    return MultiverseTable(
        [(branch, table.to_batches()) for branch, table in branch_data.items()]
    )


# -- TestMultiverseTable --


class TestMultiverseTable:
    def test_construction(self):
        mv = _make_mv_table({"a": _count_table(1), "b": _count_table(2)})
        assert mv is not None

    def test_schema_has_branch_id(self):
        mv = _make_mv_table({"a": _count_table(1)})
        schema = mv.schema()
        assert schema.field("__branch_id").type == pa.utf8()
        assert schema.field("n").type == pa.int64()

    def test_schema_mismatch_raises(self):
        t1 = pa.table({"x": pa.array([1], type=pa.int64())})
        t2 = pa.table({"y": pa.array([2], type=pa.float64())})
        with pytest.raises(ValueError, match="schema mismatch"):
            MultiverseTable([
                ("a", t1.to_batches()),
                ("b", t2.to_batches()),
            ])

    def test_empty_branches_raises(self):
        with pytest.raises(ValueError, match="at least one branch"):
            MultiverseTable([])

    def test_single_branch(self):
        mv = _make_mv_table({"only": _count_table(42)})
        ctx = datafusion.SessionContext()
        ctx.register_table("t", mv)
        result = pa.Table.from_batches(ctx.sql("SELECT * FROM t").collect())
        assert result.num_rows == 1
        assert result.column("__branch_id").to_pylist() == ["only"]
        assert result.column("n").to_pylist() == [42]


# -- TestDirectQuery --


class TestDirectQuery:
    def test_select_star(self):
        mv = _make_mv_table({
            "a": _count_table(10),
            "b": _count_table(20),
        })
        ctx = datafusion.SessionContext()
        ctx.register_table("t", mv)
        result = pa.Table.from_batches(ctx.sql("SELECT * FROM t").collect())

        assert result.num_rows == 2
        assert "__branch_id" in result.column_names
        assert sorted(result.column("n").to_pylist()) == [10, 20]

    def test_count_group_by_branch(self):
        mv = _make_mv_table({
            "a": _predictions_table([1, 2, 3], [0.9, 0.1, 0.5], [1, 0, 1]),
            "b": _predictions_table([4, 5], [0.8, 0.3], [1, 0]),
        })
        ctx = datafusion.SessionContext()
        ctx.register_table("t", mv)
        result = pa.Table.from_batches(
            ctx.sql(
                "SELECT __branch_id, COUNT(*) AS n FROM t GROUP BY __branch_id ORDER BY __branch_id"
            ).collect()
        )

        assert result.num_rows == 2
        assert result.column("__branch_id").to_pylist() == ["a", "b"]
        assert result.column("n").to_pylist() == [3, 2]

    def test_where_filter(self):
        mv = _make_mv_table({
            "a": _predictions_table([1, 2], [0.9, 0.1], [1, 0]),
            "b": _predictions_table([3], [0.8], [1]),
        })
        ctx = datafusion.SessionContext()
        ctx.register_table("t", mv)
        result = pa.Table.from_batches(
            ctx.sql("SELECT * FROM t WHERE predicted_label = 1").collect()
        )

        assert result.num_rows == 2
        assert all(v == 1 for v in result.column("predicted_label").to_pylist())

    def test_limit(self):
        mv = _make_mv_table({
            "a": _predictions_table([1, 2, 3], [0.9, 0.1, 0.5], [1, 0, 1]),
            "b": _predictions_table([4, 5], [0.8, 0.3], [1, 0]),
        })
        ctx = datafusion.SessionContext()
        ctx.register_table("t", mv)
        result = pa.Table.from_batches(
            ctx.sql("SELECT * FROM t LIMIT 2").collect()
        )
        assert result.num_rows == 2

    def test_cte(self):
        """CTE query — this would break SQL rewriting but works with TableProvider."""
        mv = _make_mv_table({
            "a": _predictions_table([1, 2], [0.9, 0.1], [1, 0]),
            "b": _predictions_table([3, 4], [0.8, 0.3], [1, 0]),
        })
        ctx = datafusion.SessionContext()
        ctx.register_table("t", mv)
        result = pa.Table.from_batches(
            ctx.sql("""
                WITH ranked AS (
                    SELECT *, ROW_NUMBER() OVER (
                        PARTITION BY __branch_id ORDER BY conversion_prob DESC
                    ) AS rn
                    FROM t
                )
                SELECT __branch_id, user_id, conversion_prob FROM ranked WHERE rn = 1
            """).collect()
        )

        assert result.num_rows == 2
        branch_ids = sorted(result.column("__branch_id").to_pylist())
        assert branch_ids == ["a", "b"]

    def test_self_join(self):
        """Self-join — another pattern that breaks SQL rewriting."""
        mv = _make_mv_table({
            "a": _count_table(10),
            "b": _count_table(20),
        })
        ctx = datafusion.SessionContext()
        ctx.register_table("t", mv)
        result = pa.Table.from_batches(
            ctx.sql("""
                SELECT t1.__branch_id, t1.n AS n1, t2.n AS n2
                FROM t AS t1
                JOIN t AS t2 ON t1.__branch_id = t2.__branch_id
            """).collect()
        )

        assert result.num_rows == 2

    def test_subquery(self):
        """Subquery — another pattern that breaks SQL rewriting."""
        mv = _make_mv_table({
            "a": _predictions_table([1, 2], [0.9, 0.1], [1, 0]),
            "b": _predictions_table([3], [0.5], [1]),
        })
        ctx = datafusion.SessionContext()
        ctx.register_table("t", mv)
        result = pa.Table.from_batches(
            ctx.sql("""
                SELECT * FROM t
                WHERE conversion_prob > (SELECT AVG(conversion_prob) FROM t)
            """).collect()
        )

        assert result.num_rows > 0
        assert all(p > 0.5 for p in result.column("conversion_prob").to_pylist())


# -- TestQueryMultiverse (end-to-end via engine, mocked Iceberg lookups) --


def _make_engine_mocks(data: dict[str, dict[str, pa.Table]]):
    """Build mocks for get_parquet_files and S3MultiverseTable.

    Since S3MultiverseTable needs real S3, we intercept the constructor and
    return a MultiverseTable (MemTable-backed) built from the test data instead.
    """
    def mock_get_parquet_files(metadata_location: str) -> list[str]:
        parts = metadata_location.replace("s3://fake/", "").split("/")
        branch, table = parts[0], parts[1]
        return [f"s3://fake/{branch}/{table}/part-0.parquet"]

    def mock_lazy_table(branch_files):
        """Replace S3MultiverseTable with MultiverseTable using test data."""
        branch_batches = []
        for branch_id, _files in branch_files:
            tables_for_branch = data[branch_id]
            table_name = next(iter(tables_for_branch))
            arrow_table = tables_for_branch[table_name]
            branch_batches.append((branch_id, arrow_table.to_batches()))
        return MultiverseTable(branch_batches)

    return mock_get_parquet_files, mock_lazy_table


class TestQueryMultiverse:
    def test_end_to_end_count(self):
        data = {
            "A": {"user_predictions": _count_table(100)},
            "B": {"user_predictions": _count_table(200)},
            "C": {"user_predictions": _count_table(300)},
        }
        metadata = {b: {t: f"s3://fake/{b}/{t}" for t in tables}
                    for b, tables in data.items()}
        engine = NativeMultiverse(metadata)

        shape = QueryShape(
            result_type=ResultType.NUMBER,
            value_column="n",
            set_column=None,
            expects_single_row=True,
            max_rows_hint=None,
        )

        mock_files, mock_lazy = _make_engine_mocks(data)
        with patch("app.native_multiverse.get_parquet_files", side_effect=mock_files), \
             patch("app.native_multiverse.S3MultiverseTable", side_effect=mock_lazy):
            combined, errors, meta = engine.query_multiverse(
                "SELECT * FROM user_predictions", ["A", "B", "C"], shape=shape
            )

        assert combined is not None
        assert combined.num_rows == 3
        assert sorted(combined.column("n").to_pylist()) == [100, 200, 300]
        assert meta is not None
        assert meta.kind == "number"
        assert meta.details["min"] == 100
        assert meta.details["max"] == 300

    def test_end_to_end_aggregate(self):
        data = {
            "A": {"user_predictions": _predictions_table([1, 2], [0.9, 0.1], [1, 0])},
            "B": {"user_predictions": _predictions_table([3], [0.5], [1])},
        }
        metadata = {b: {t: f"s3://fake/{b}/{t}" for t in tables}
                    for b, tables in data.items()}
        engine = NativeMultiverse(metadata)

        mock_files, mock_lazy = _make_engine_mocks(data)
        with patch("app.native_multiverse.get_parquet_files", side_effect=mock_files), \
             patch("app.native_multiverse.S3MultiverseTable", side_effect=mock_lazy):
            combined, errors, summary = engine.query_multiverse(
                "SELECT __branch_id, COUNT(*) AS n FROM user_predictions GROUP BY __branch_id",
                ["A", "B"],
            )

        assert combined is not None
        # A has 2 rows, B has 1 row
        assert sorted(combined.column("n").to_pylist()) == [1, 2]

    def test_cte_via_engine(self):
        """CTE through the engine — proves no SQL rewriting interference."""
        data = {
            "X": {"user_predictions": _predictions_table([1, 2], [0.9, 0.1], [1, 0])},
            "Y": {"user_predictions": _predictions_table([3, 4], [0.7, 0.2], [1, 0])},
        }
        metadata = {b: {t: f"s3://fake/{b}/{t}" for t in tables}
                    for b, tables in data.items()}
        engine = NativeMultiverse(metadata)

        sql = """
            WITH ranked AS (
                SELECT *, ROW_NUMBER() OVER (
                    PARTITION BY __branch_id ORDER BY conversion_prob DESC
                ) AS rn
                FROM user_predictions
            )
            SELECT __branch_id, user_id FROM ranked WHERE rn = 1
        """

        mock_files, mock_lazy = _make_engine_mocks(data)
        with patch("app.native_multiverse.get_parquet_files", side_effect=mock_files), \
             patch("app.native_multiverse.S3MultiverseTable", side_effect=mock_lazy):
            combined, errors, summary = engine.query_multiverse(sql, ["X", "Y"])

        assert combined is not None
        assert combined.num_rows == 2

    def test_single_branch(self):
        data = {"only": {"user_predictions": _count_table(7)}}
        metadata = {"only": {"user_predictions": "s3://fake/only/user_predictions"}}
        engine = NativeMultiverse(metadata)

        mock_files, mock_lazy = _make_engine_mocks(data)
        with patch("app.native_multiverse.get_parquet_files", side_effect=mock_files), \
             patch("app.native_multiverse.S3MultiverseTable", side_effect=mock_lazy):
            combined, errors, summary = engine.query_multiverse(
                "SELECT * FROM user_predictions", ["only"]
            )

        assert combined is not None
        assert combined.num_rows == 1
        assert combined.column("n").to_pylist() == [7]


# -- TestErrorHandling --


class TestErrorHandling:
    def test_missing_branch_crashes(self):
        """Missing branch key → KeyError, caught by app."""
        metadata = {"A": {"user_predictions": "s3://fake/A/user_predictions"}}
        engine = NativeMultiverse(metadata)

        with patch("app.native_multiverse.get_parquet_files", return_value=["dummy.parquet"]):
            with pytest.raises(KeyError):
                engine.query_multiverse("SELECT * FROM user_predictions", ["A", "B"])

    def test_get_parquet_files_failure_crashes(self):
        """get_parquet_files failure → exception, caught by app."""
        metadata = {"A": {"user_predictions": "s3://fake/A/user_predictions"}}
        engine = NativeMultiverse(metadata)

        def boom(loc):
            raise RuntimeError("S3 error")

        with patch("app.native_multiverse.get_parquet_files", side_effect=boom):
            with pytest.raises(RuntimeError, match="S3 error"):
                engine.query_multiverse("SELECT * FROM user_predictions", ["A"])

    def test_empty_metadata_returns_none(self):
        """Empty metadata → returns None, no crash."""
        engine = NativeMultiverse({})
        combined, errors, summary = engine.query_multiverse("SELECT * FROM t", ["A"])
        assert combined is None
