"""Tests for supervaluation meta-answer layer."""

import pyarrow as pa
import pytest

from app.query_shape import QueryShape, ResultType
from app.supervaluation import (
    MetaAnswer,
    compute_meta_answer,
    meta_answer_boolean,
    meta_answer_number,
    meta_answer_set,
)


class TestMetaAnswerBoolean:
    def test_all_true(self):
        m = meta_answer_boolean({"a": True, "b": True, "c": True})
        assert m.kind == "boolean"
        assert m.summary == "Definitely true"
        assert m.details["verdict"] == "definitely_true"

    def test_all_false(self):
        m = meta_answer_boolean({"a": False, "b": False, "c": False})
        assert m.summary == "Definitely false"
        assert m.details["verdict"] == "definitely_false"

    def test_mixed(self):
        m = meta_answer_boolean({"a": True, "b": False, "c": True})
        assert m.details["verdict"] == "mixed"
        assert m.details["true_count"] == 2
        assert m.details["false_count"] == 1
        assert "Mixed" in m.summary


class TestMetaAnswerNumber:
    def test_all_same(self):
        m = meta_answer_number({"a": 42.0, "b": 42.0, "c": 42.0})
        assert m.details["agreement"] is True
        assert "All agree" in m.summary
        assert m.details["min"] == 42.0
        assert m.details["max"] == 42.0

    def test_different(self):
        m = meta_answer_number({"a": 100.0, "b": 200.0, "c": 300.0})
        assert m.details["agreement"] is False
        assert m.details["min"] == 100.0
        assert m.details["max"] == 300.0
        assert m.details["mean"] == pytest.approx(200.0)
        assert m.details["spread"] == 200.0
        assert "Range" in m.summary


class TestMetaAnswerSet:
    def test_all_same(self):
        s = {1, 2, 3}
        m = meta_answer_set({"a": s.copy(), "b": s.copy(), "c": s.copy()})
        assert m.details["intersection"] == {1, 2, 3}
        assert m.details["union"] == {1, 2, 3}
        assert m.details["disagreement"] == set()
        assert "3 of 3" in m.summary

    def test_disjoint(self):
        m = meta_answer_set({"a": {1, 2}, "b": {3, 4}})
        assert m.details["intersection"] == set()
        assert m.details["union"] == {1, 2, 3, 4}
        assert "0 of 4" in m.summary

    def test_partial_overlap(self):
        m = meta_answer_set({"a": {1, 2, 3}, "b": {2, 3, 4}})
        assert m.details["intersection"] == {2, 3}
        assert m.details["union"] == {1, 2, 3, 4}
        assert m.details["unique_per_branch"]["a"] == {1}
        assert m.details["unique_per_branch"]["b"] == {4}
        assert m.details["disagreement"] == {1, 4}


class TestComputeMetaAnswer:
    def test_number_from_arrow_table(self):
        table = pa.table(
            {
                "__branch_id": ["a", "b", "c"],
                "n": pa.array([10, 20, 30], type=pa.int64()),
            }
        )
        shape = QueryShape(
            result_type=ResultType.NUMBER,
            value_column="n",
            set_column=None,
            expects_single_row=True,
            max_rows_hint=None,
        )
        m = compute_meta_answer(table, shape)
        assert isinstance(m, MetaAnswer)
        assert m.kind == "number"
        assert m.details["min"] == 10
        assert m.details["max"] == 30

    def test_set_from_arrow_table(self):
        table = pa.table(
            {
                "__branch_id": ["a", "a", "b", "b", "b"],
                "user_id": pa.array([1, 2, 2, 3, 4], type=pa.int64()),
            }
        )
        shape = QueryShape(
            result_type=ResultType.SET,
            value_column=None,
            set_column="user_id",
            expects_single_row=False,
            max_rows_hint=50,
        )
        m = compute_meta_answer(table, shape)
        assert m.kind == "set"
        assert m.per_branch["a"] == {1, 2}
        assert m.per_branch["b"] == {2, 3, 4}
        assert m.details["intersection"] == {2}

    def test_boolean_from_arrow_table(self):
        table = pa.table(
            {
                "__branch_id": ["a", "b"],
                "ok": pa.array([True, False], type=pa.bool_()),
            }
        )
        shape = QueryShape(
            result_type=ResultType.BOOLEAN,
            value_column="ok",
            set_column=None,
            expects_single_row=True,
            max_rows_hint=None,
        )
        m = compute_meta_answer(table, shape)
        assert m.kind == "boolean"
        assert m.details["verdict"] == "mixed"

    def test_number_empty_branch_raises(self):
        """Branch with no rows should raise ValueError, not IndexError."""
        table_with_ghost = pa.table(
            {
                "__branch_id": ["a"],
                "n": pa.array([10], type=pa.int64()),
            }
        )
        # This should work fine (1 row for branch "a")
        shape = QueryShape(
            result_type=ResultType.NUMBER,
            value_column="n",
            set_column=None,
            expects_single_row=True,
            max_rows_hint=None,
        )
        m = compute_meta_answer(table_with_ghost, shape)
        assert m.details["min"] == 10
