"""Tests for query shape classifier."""

import pyarrow as pa
import pytest

from app.query_shape import ResultType, UnsupportedQueryError, classify

SCHEMAS = {
    "user_predictions": pa.schema(
        [
            pa.field("user_id", pa.int64()),
            pa.field("conversion_prob", pa.float64()),
            pa.field("predicted_label", pa.int32()),
        ]
    ),
}


class TestClassifyNumber:
    def test_count(self):
        shape = classify("SELECT COUNT(*) FROM user_predictions", schemas=SCHEMAS)
        assert shape.result_type == ResultType.NUMBER
        assert shape.expects_single_row is True

    def test_sum(self):
        shape = classify("SELECT SUM(predicted_label) FROM user_predictions", schemas=SCHEMAS)
        assert shape.result_type == ResultType.NUMBER
        assert shape.value_column is not None

    def test_avg(self):
        shape = classify("SELECT AVG(conversion_prob) FROM user_predictions", schemas=SCHEMAS)
        assert shape.result_type == ResultType.NUMBER


class TestClassifyBoolean:
    def test_exists(self):
        shape = classify(
            "SELECT EXISTS(SELECT 1 FROM user_predictions WHERE predicted_label = 1) AS ok",
            schemas=SCHEMAS,
        )
        assert shape.result_type == ResultType.BOOLEAN
        assert shape.value_column == "ok"

    def test_comparison(self):
        shape = classify(
            "SELECT COUNT(*) > 100 AS result FROM user_predictions",
            schemas=SCHEMAS,
        )
        assert shape.result_type == ResultType.BOOLEAN
        assert shape.value_column == "result"


class TestClassifySet:
    def test_select_ids_with_limit(self):
        shape = classify(
            "SELECT user_id FROM user_predictions WHERE predicted_label = 1 LIMIT 50",
            schemas=SCHEMAS,
        )
        assert shape.result_type == ResultType.SET
        assert shape.set_column == "user_id"
        assert shape.max_rows_hint == 50
        assert shape.expects_single_row is False


class TestUnsupported:
    def test_multi_column(self):
        with pytest.raises(UnsupportedQueryError, match="Multi-column"):
            classify("SELECT user_id, conversion_prob FROM user_predictions", schemas=SCHEMAS)

    def test_grouped_multi_row(self):
        with pytest.raises(UnsupportedQueryError, match="GROUP BY"):
            classify(
                "SELECT predicted_label FROM user_predictions GROUP BY predicted_label",
                schemas=SCHEMAS,
            )

    def test_cte(self):
        with pytest.raises(UnsupportedQueryError, match="CTE"):
            classify(
                "WITH cte AS (SELECT * FROM user_predictions) SELECT COUNT(*) FROM cte",
                schemas=SCHEMAS,
            )

    def test_window_function(self):
        with pytest.raises(UnsupportedQueryError, match="Window"):
            classify(
                "SELECT ROW_NUMBER() OVER(ORDER BY conversion_prob) AS rn FROM user_predictions",
                schemas=SCHEMAS,
            )

    def test_order_by_without_limit(self):
        with pytest.raises(UnsupportedQueryError, match="ORDER BY without LIMIT"):
            classify(
                "SELECT user_id FROM user_predictions ORDER BY conversion_prob DESC",
                schemas=SCHEMAS,
            )


class TestNonStrictMode:
    """With strict=False, CTE/window/ORDER BY guards are skipped."""

    def test_cte_allowed(self):
        shape = classify(
            "WITH cte AS (SELECT COUNT(*) AS n FROM user_predictions) SELECT n FROM cte",
            schemas=SCHEMAS,
            strict=False,
        )
        assert shape.result_type == ResultType.NUMBER

    def test_window_allowed(self):
        shape = classify(
            "SELECT ROW_NUMBER() OVER(ORDER BY conversion_prob) AS rn FROM user_predictions",
            schemas=SCHEMAS,
            strict=False,
        )
        # ROW_NUMBER returns int64, no aggregate → SET
        assert shape.result_type == ResultType.SET

    def test_order_by_without_limit_allowed(self):
        shape = classify(
            "SELECT user_id FROM user_predictions ORDER BY conversion_prob DESC",
            schemas=SCHEMAS,
            strict=False,
        )
        assert shape.result_type == ResultType.SET

    def test_multi_column_still_rejected(self):
        """Multi-column is a shape issue, not an engine limitation."""
        with pytest.raises(UnsupportedQueryError, match="Multi-column"):
            classify(
                "SELECT user_id, conversion_prob FROM user_predictions",
                schemas=SCHEMAS,
                strict=False,
            )
