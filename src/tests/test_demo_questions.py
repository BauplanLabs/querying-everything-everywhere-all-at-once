"""Deterministic tests for the demo question pipeline.

For each of the 4 demo questions (Phase 2-5), we provide the expected SQL
that the LLM would produce and verify:
  1. The SQL passes DataFusion validation against the real table schemas.
  2. classify() returns the correct ResultType.
  3. compute_meta_answer() produces the right MetaAnswer kind from synthetic data.

The LLM call itself is non-deterministic, so we test everything downstream of it.
"""

import pyarrow as pa
import pytest

from app.query_shape import QueryShape, ResultType, UnsupportedQueryError, classify
from app.text_to_sql import validate_sql
from app.supervaluation import (
    compute_meta_answer,
    meta_answer_boolean,
    meta_answer_number,
    meta_answer_set,
)

# -- Schemas matching the real tables ---

USER_PREDICTIONS_SCHEMA = pa.schema(
    [
        pa.field("user_id", pa.string()),
        pa.field("conversion_prob", pa.float64()),
        pa.field("predicted_label", pa.int64()),
    ]
)

ECOMMERCE_USERS_SCHEMA = pa.schema(
    [
        pa.field("user_id", pa.string()),
        pa.field("category_of_interest", pa.string()),
        pa.field("customer_segment", pa.string()),
    ]
)

SCHEMAS = {
    "user_predictions": USER_PREDICTIONS_SCHEMA,
    "ecommerce_users": ECOMMERCE_USERS_SCHEMA,
}

# -- Expected SQL per demo phase --

PHASE_2_SQL = (
    "SELECT COUNT(*) AS n_buyers FROM user_predictions WHERE predicted_label = 1"
)

PHASE_3_SQL = (
    "SELECT CAST(SUM(CASE WHEN predicted_label = 1 THEN 1 ELSE 0 END) AS DOUBLE) "
    "/ COUNT(*) > 0.02 AS answer "
    "FROM user_predictions"
)

PHASE_4_SQL = (
    "SELECT COUNT(*) AS n_buyers "
    "FROM user_predictions p "
    "JOIN (SELECT DISTINCT user_id FROM ecommerce_users "
    "WHERE category_of_interest = 'electronics.smartphone') u "
    "ON p.user_id = u.user_id "
    "WHERE p.predicted_label = 1"
)

PHASE_5_SQL = (
    "SELECT user_id "
    "FROM user_predictions "
    "WHERE predicted_label = 1 "
    "ORDER BY conversion_prob DESC "
    "LIMIT 50"
)


# ---- Phase 2: NUMBER ----


class TestPhase2Number:
    """Phase 2: 'How many customers will buy tomorrow?'"""

    def test_sql_validates(self):
        validate_sql(PHASE_2_SQL, SCHEMAS)

    def test_classify_returns_number(self):
        shape = classify(PHASE_2_SQL, schemas=SCHEMAS)
        assert shape.result_type == ResultType.NUMBER
        assert shape.value_column == "n_buyers"
        assert shape.expects_single_row is True

    def test_meta_answer_agreement(self):
        per_branch = {"branch_a": 150, "branch_b": 150, "branch_c": 150}
        meta = meta_answer_number(per_branch)
        assert meta.kind == "number"
        assert meta.details["agreement"] is True
        assert meta.details["min"] == 150

    def test_meta_answer_disagreement(self):
        per_branch = {"branch_a": 120, "branch_b": 180, "branch_c": 95}
        meta = meta_answer_number(per_branch)
        assert meta.kind == "number"
        assert meta.details["agreement"] is False
        assert meta.details["min"] == 95
        assert meta.details["max"] == 180

    def test_compute_meta_answer_from_table(self):
        table = pa.table(
            {
                "n_buyers": [120, 180, 95],
                "__branch_id": ["branch_a", "branch_b", "branch_c"],
            }
        )
        shape = QueryShape(
            result_type=ResultType.NUMBER,
            value_column="n_buyers",
            set_column=None,
            expects_single_row=True,
            max_rows_hint=None,
        )
        meta = compute_meta_answer(table, shape)
        assert meta.kind == "number"
        assert meta.per_branch["branch_a"] == 120
        assert meta.details["min"] == 95


# ---- Phase 3: BOOLEAN ----


class TestPhase3Boolean:
    """Phase 3: 'Is tomorrow's conversion rate above 2%?'"""

    def test_sql_validates(self):
        validate_sql(PHASE_3_SQL, SCHEMAS)

    def test_classify_returns_boolean(self):
        shape = classify(PHASE_3_SQL, schemas=SCHEMAS)
        assert shape.result_type == ResultType.BOOLEAN
        assert shape.value_column == "answer"
        assert shape.expects_single_row is True

    def test_meta_answer_definitely_true(self):
        per_branch = {"a": True, "b": True, "c": True}
        meta = meta_answer_boolean(per_branch)
        assert meta.details["verdict"] == "definitely_true"

    def test_meta_answer_definitely_false(self):
        per_branch = {"a": False, "b": False}
        meta = meta_answer_boolean(per_branch)
        assert meta.details["verdict"] == "definitely_false"

    def test_meta_answer_mixed(self):
        per_branch = {"a": True, "b": False, "c": True}
        meta = meta_answer_boolean(per_branch)
        assert meta.details["verdict"] == "mixed"
        assert meta.details["true_count"] == 2
        assert meta.details["false_count"] == 1

    def test_compute_meta_answer_from_table(self):
        table = pa.table(
            {
                "answer": [True, False, True],
                "__branch_id": ["a", "b", "c"],
            }
        )
        shape = QueryShape(
            result_type=ResultType.BOOLEAN,
            value_column="answer",
            set_column=None,
            expects_single_row=True,
            max_rows_hint=None,
        )
        meta = compute_meta_answer(table, shape)
        assert meta.kind == "boolean"
        assert meta.details["verdict"] == "mixed"


# ---- Phase 4: NUMBER with JOIN ----


class TestPhase4Join:
    """Phase 4: 'How many smartphone shoppers will convert tomorrow?'"""

    def test_sql_validates(self):
        validate_sql(PHASE_4_SQL, SCHEMAS)

    def test_classify_returns_number(self):
        shape = classify(PHASE_4_SQL, schemas=SCHEMAS)
        assert shape.result_type == ResultType.NUMBER
        assert shape.value_column == "n_buyers"

    def test_join_with_shared_table(self):
        """The SQL references both user_predictions and ecommerce_users."""
        assert "ecommerce_users" in PHASE_4_SQL
        assert "user_predictions" in PHASE_4_SQL
        assert "JOIN" in PHASE_4_SQL


# ---- Phase 5: SET ----


class TestPhase5Set:
    """Phase 5: 'Which customers should we target tomorrow?'"""

    def test_sql_validates(self):
        validate_sql(PHASE_5_SQL, SCHEMAS)

    def test_classify_returns_set(self):
        shape = classify(PHASE_5_SQL, schemas=SCHEMAS)
        assert shape.result_type == ResultType.SET
        assert shape.set_column == "user_id"
        assert shape.max_rows_hint == 50

    def test_meta_answer_full_consensus(self):
        per_branch = {
            "a": {"u1", "u2", "u3"},
            "b": {"u1", "u2", "u3"},
        }
        meta = meta_answer_set(per_branch)
        assert meta.kind == "set"
        assert meta.details["intersection"] == {"u1", "u2", "u3"}
        assert meta.details["disagreement"] == set()

    def test_meta_answer_partial_overlap(self):
        per_branch = {
            "a": {"u1", "u2", "u3"},
            "b": {"u2", "u3", "u4"},
            "c": {"u3", "u5"},
        }
        meta = meta_answer_set(per_branch)
        assert meta.details["intersection"] == {"u3"}
        assert meta.details["union"] == {"u1", "u2", "u3", "u4", "u5"}
        assert meta.details["unique_per_branch"]["a"] == {"u1", "u2"}

    def test_compute_meta_answer_from_table(self):
        table = pa.table(
            {
                "user_id": ["u1", "u2", "u3", "u2", "u3", "u4"],
                "__branch_id": ["a", "a", "a", "b", "b", "b"],
            }
        )
        shape = QueryShape(
            result_type=ResultType.SET,
            value_column=None,
            set_column="user_id",
            expects_single_row=False,
            max_rows_hint=50,
        )
        meta = compute_meta_answer(table, shape)
        assert meta.kind == "set"
        assert meta.details["intersection"] == {"u2", "u3"}


# ---- Edge cases ----


class TestClassifyGuards:
    """Verify classify rejects queries the naive engine can't handle."""

    def test_rejects_cte(self):
        sql = "WITH cte AS (SELECT * FROM user_predictions) SELECT COUNT(*) FROM cte"
        with pytest.raises(UnsupportedQueryError, match="CTEs"):
            classify(sql, schemas=SCHEMAS)

    def test_rejects_window(self):
        sql = "SELECT ROW_NUMBER() OVER (ORDER BY conversion_prob) AS rn FROM user_predictions"
        with pytest.raises(UnsupportedQueryError, match="Window"):
            classify(sql, schemas=SCHEMAS)

    def test_rejects_multi_column(self):
        sql = "SELECT user_id, conversion_prob FROM user_predictions LIMIT 10"
        with pytest.raises(UnsupportedQueryError, match="Multi-column"):
            classify(sql, schemas=SCHEMAS)

    def test_allows_cte_in_non_strict(self):
        sql = "WITH cte AS (SELECT user_id FROM user_predictions) SELECT COUNT(*) AS n FROM cte"
        shape = classify(sql, schemas=SCHEMAS, strict=False)
        assert shape.result_type == ResultType.NUMBER


# ---- Branch filtering ----


class TestBranchFiltering:
    """Verify transactional branch exclusion and helper functions."""

    def test_transactional_branch_excluded(self):
        from app.lakehouse import _is_excluded_branch
        assert _is_excluded_branch(
            "jacopo.multiverse_v_v_60m_sql_gb-bpln-tx-run-20260307-abcd1234"
        )

    def test_normal_branch_not_excluded(self):
        from app.lakehouse import _is_excluded_branch
        assert not _is_excluded_branch("jacopo.multiverse_v_v_60m_sql_gb")

    def test_e2e_check_branch_excluded(self):
        from app.lakehouse import _is_excluded_branch
        assert _is_excluded_branch("bauplan-e2e-check-bauplan-prod-something")

    def test_regular_user_branch_not_excluded(self):
        from app.lakehouse import _is_excluded_branch
        assert not _is_excluded_branch("jacopo.multiverse_main")


# ---- Operator bytes parsing ----


class TestParseOperatorBytes:
    """Verify EXPLAIN ANALYZE output_bytes metric extraction."""

    def test_simple_bytes(self):
        from app.multiverse import parse_operator_bytes
        plan = "AggregateExec: metrics=[output_rows=1, output_bytes=8.0 B]"
        assert parse_operator_bytes(plan) == 8

    def test_kilobytes(self):
        from app.multiverse import parse_operator_bytes
        plan = "FilterExec: metrics=[output_bytes=32.0 KB]"
        assert parse_operator_bytes(plan) == 32 * 1024

    def test_mebibytes(self):
        from app.multiverse import parse_operator_bytes
        plan = "ScanExec: metrics=[output_bytes=2.5 MiB]"
        assert parse_operator_bytes(plan) == int(2.5 * 1024 ** 2)

    def test_multiple_operators_summed(self):
        from app.multiverse import parse_operator_bytes
        plan = (
            "AggregateExec: metrics=[output_bytes=96.0 B]\n"
            "  RepartitionExec: metrics=[output_bytes=1.0 KB]\n"
            "    DatasetExec: metrics=[]"
        )
        assert parse_operator_bytes(plan) == 96 + 1024

    def test_empty_metrics(self):
        from app.multiverse import parse_operator_bytes
        plan = "DatasetExec: number_of_fragments=1, metrics=[]"
        assert parse_operator_bytes(plan) == 0

    def test_real_explain_analyze_output(self):
        from app.multiverse import parse_operator_bytes
        plan = (
            "ProjectionExec: metrics=[output_rows=1, output_bytes=8.0 B]\n"
            "  AggregateExec: mode=Final, metrics=[output_rows=1, output_bytes=8.0 B]\n"
            "    CoalescePartitionsExec, metrics=[output_rows=12, output_bytes=96.0 B]\n"
            "      AggregateExec: mode=Partial, metrics=[output_rows=12, output_bytes=96.0 B]\n"
            "        RepartitionExec: metrics=[spilled_bytes=0.0 B]\n"
            "          DatasetExec: metrics=[]"
        )
        # 8 + 8 + 96 + 96 + 0 (spilled_bytes not matched by output_bytes regex)
        assert parse_operator_bytes(plan) == 208
