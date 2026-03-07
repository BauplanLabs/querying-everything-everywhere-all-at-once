"""Tests for NaiveMultiverse — per-branch DataFusion execution and summarization.

The execution tests write temporary Parquet files and register them as PyArrow
datasets in DataFusion — same mechanism the real engine uses, but local disk
instead of S3.
"""

from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest

from app.naive_multiverse import NaiveMultiverse
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


def _write_parquet(table: pa.Table, path: Path) -> str:
    """Write an Arrow table to a Parquet file and return the path string."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(path))
    return str(path)


# -- Mock for _register_branch (replaces S3 with local Parquet) --


def _mock_register_branch_factory(tmp_path, data_by_branch):
    """Create a mock _register_branch that writes local Parquet and registers datasets."""

    def _mock_register_branch(self, ctx, tables, s3_fs=None):
        # `tables` is {table_name: metadata_location} — we ignore the metadata
        # location and look up the Arrow table from data_by_branch instead.
        # We need to figure out which branch this is by matching the metadata
        # locations back to the branch.
        for branch, branch_tables in data_by_branch.items():
            # Check if these metadata locations match this branch
            branch_meta = self._metadata_by_branch.get(branch, {})
            if set(tables.values()) == set(branch_meta.values()):
                for table_name in tables:
                    arrow_table = branch_tables[table_name]
                    pq_path = _write_parquet(
                        arrow_table, tmp_path / branch / f"{table_name}.parquet"
                    )
                    dataset = ds.dataset(pq_path, format="parquet")
                    ctx.register_dataset(table_name, dataset)
                return
        raise ValueError("Could not match tables to a branch")

    return _mock_register_branch


# -- TestQueryMultiverse (end-to-end with mocked _register_branch) --


class TestQueryMultiverse:
    def test_end_to_end_count(self, tmp_path):
        data = {
            "A": {"t": _count_table(100)},
            "B": {"t": _count_table(200)},
            "C": {"t": _count_table(300)},
        }
        metadata = {b: {"t": f"s3://fake/{b}"} for b in data}
        engine = NaiveMultiverse(metadata)
        shape = QueryShape(
            result_type=ResultType.NUMBER,
            value_column="n",
            set_column=None,
            expects_single_row=True,
            max_rows_hint=None,
        )

        mock_fn = _mock_register_branch_factory(tmp_path, data)
        with patch.object(NaiveMultiverse, "_register_branch", mock_fn):
            combined, errors, meta = engine.query_multiverse(
                "SELECT * FROM t", ["A", "B", "C"], shape=shape
            )

        assert errors == {}
        assert combined is not None
        assert combined.num_rows == 3
        assert sorted(combined.column("n").to_pylist()) == [100, 200, 300]
        assert meta is not None
        assert meta.kind == "number"
        assert meta.details["min"] == 100
        assert meta.details["max"] == 300

    def test_end_to_end_aggregate(self, tmp_path):
        data = {
            "A": {"t": _predictions_table([1, 2], [0.9, 0.1], [1, 0])},
            "B": {"t": _predictions_table([3], [0.5], [1])},
        }
        metadata = {b: {"t": f"s3://fake/{b}"} for b in data}
        engine = NaiveMultiverse(metadata)

        mock_fn = _mock_register_branch_factory(tmp_path, data)
        with patch.object(NaiveMultiverse, "_register_branch", mock_fn):
            combined, errors, summary = engine.query_multiverse(
                "SELECT COUNT(*) AS n FROM t", ["A", "B"]
            )

        assert errors == {}
        assert combined is not None
        assert sorted(combined.column("n").to_pylist()) == [1, 2]

    def test_partial_branch_failure(self, tmp_path):
        """Requesting a branch not in metadata → KeyError."""
        data = {"A": {"t": _count_table(42)}}
        metadata = {"A": {"t": "s3://fake/A"}}
        engine = NaiveMultiverse(metadata)

        mock_fn = _mock_register_branch_factory(tmp_path, data)
        with patch.object(NaiveMultiverse, "_register_branch", mock_fn):
            with pytest.raises(KeyError):
                engine.query_multiverse("SELECT * FROM t", ["A", "B"])


# -- TestErrorHandling --


class TestErrorHandling:
    def test_empty_metadata_returns_none(self):
        engine = NaiveMultiverse({})
        combined, errors, summary = engine.query_multiverse("SELECT * FROM t", ["A", "B"])
        assert combined is None

    def test_single_branch_success(self, tmp_path):
        data = {"only": {"t": _count_table(7)}}
        metadata = {"only": {"t": "s3://fake/only"}}
        engine = NaiveMultiverse(metadata)

        mock_fn = _mock_register_branch_factory(tmp_path, data)
        with patch.object(NaiveMultiverse, "_register_branch", mock_fn):
            combined, errors, summary = engine.query_multiverse(
                "SELECT * FROM t", ["only"]
            )

        assert errors == {}
        assert combined is not None
        assert combined.num_rows == 1
        assert combined.column("n").to_pylist() == [7]

    def test_empty_table_branch(self, tmp_path):
        empty = pa.table({"n": pa.array([], type=pa.int64())})
        data = {"empty": {"t": empty}, "full": {"t": _count_table(10)}}
        metadata = {b: {"t": f"s3://fake/{b}"} for b in data}
        engine = NaiveMultiverse(metadata)

        mock_fn = _mock_register_branch_factory(tmp_path, data)
        with patch.object(NaiveMultiverse, "_register_branch", mock_fn):
            combined, errors, summary = engine.query_multiverse(
                "SELECT * FROM t", ["empty", "full"]
            )

        assert combined is not None
        assert combined.column("__branch_id").to_pylist() == ["full"]


# -- TestSummarize --


class TestSummarize:
    @pytest.fixture()
    def engine(self):
        return NaiveMultiverse({})

    @pytest.fixture()
    def number_shape(self):
        return QueryShape(
            result_type=ResultType.NUMBER,
            value_column="n",
            set_column=None,
            expects_single_row=True,
            max_rows_hint=None,
        )

    def test_single_int_column(self, engine, number_shape):
        combined = pa.table(
            {
                "__branch_id": ["a", "b", "c"],
                "n": pa.array([10, 20, 30], type=pa.int64()),
            }
        )
        m = engine._summarize(combined, number_shape)
        assert m is not None
        assert m.kind == "number"
        assert m.details["min"] == 10
        assert m.details["max"] == 30
        assert m.details["mean"] == pytest.approx(20.0)

    def test_single_float_column(self, engine):
        combined = pa.table(
            {
                "__branch_id": ["a", "b"],
                "pct": pa.array([0.12, 0.18], type=pa.float64()),
            }
        )
        shape = QueryShape(
            result_type=ResultType.NUMBER,
            value_column="pct",
            set_column=None,
            expects_single_row=True,
            max_rows_hint=None,
        )
        m = engine._summarize(combined, shape)
        assert m is not None
        assert m.details["min"] == pytest.approx(0.12)
        assert m.details["max"] == pytest.approx(0.18)
        assert m.details["mean"] == pytest.approx(0.15)

    def test_none_shape_returns_none(self, engine):
        combined = pa.table(
            {
                "__branch_id": ["a"],
                "n": [10],
                "pct": [0.5],
            }
        )
        assert engine._summarize(combined, None) is None

    def test_none_table_returns_none(self, engine, number_shape):
        assert engine._summarize(None, number_shape) is None

    def test_both_none_returns_none(self, engine):
        assert engine._summarize(None, None) is None

    def test_set_shape(self, engine):
        combined = pa.table(
            {
                "__branch_id": ["a", "a", "b"],
                "user_id": pa.array([1, 2, 2], type=pa.int64()),
            }
        )
        shape = QueryShape(
            result_type=ResultType.SET,
            value_column=None,
            set_column="user_id",
            expects_single_row=False,
            max_rows_hint=None,
        )
        m = engine._summarize(combined, shape)
        assert m is not None
        assert m.kind == "set"
        assert m.details["intersection"] == {2}

    def test_identical_values_across_branches(self, engine, number_shape):
        combined = pa.table(
            {
                "__branch_id": ["a", "b", "c"],
                "n": pa.array([42, 42, 42], type=pa.int64()),
            }
        )
        m = engine._summarize(combined, number_shape)
        assert m.details["min"] == 42
        assert m.details["max"] == 42
        assert m.details["agreement"] is True
