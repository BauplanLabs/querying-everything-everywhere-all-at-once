"""Unit tests for text_to_sql validation and prompt building.

Pure tests — no bauplan dependency, no mocking (except TestRetryLoop).
The TestBusinessQuestions class requires OPENAI_API_KEY and is skipped without it.
"""

import os
from unittest.mock import patch

import pyarrow as pa
import pytest

from app.text_to_sql import (
    build_system_prompt,
    cache_delete,
    cache_get,
    cache_put,
    nl_to_sql,
    validate_sql,
)

# -- Shared fixture: a realistic user_predictions schema --

SCHEMAS = {
    "user_predictions": pa.schema(
        [
            pa.field("user_id", pa.int64()),
            pa.field("conversion_prob", pa.float64()),
            pa.field("predicted_label", pa.int32()),
        ]
    )
}


class TestValidateSql:
    def test_valid_select(self):
        validate_sql("SELECT COUNT(*) FROM user_predictions", SCHEMAS)

    def test_valid_select_with_where(self):
        validate_sql(
            "SELECT user_id FROM user_predictions WHERE predicted_label = 1",
            SCHEMAS,
        )

    def test_invalid_table(self):
        with pytest.raises(ValueError, match="validation failed"):
            validate_sql("SELECT * FROM nonexistent", SCHEMAS)

    def test_invalid_column(self):
        with pytest.raises(ValueError, match="validation failed"):
            validate_sql("SELECT bogus FROM user_predictions", SCHEMAS)

    def test_forbidden_insert(self):
        with pytest.raises(ValueError, match="Only SELECT"):
            validate_sql("INSERT INTO user_predictions VALUES (1, 0.5, 1)", SCHEMAS)

    def test_forbidden_drop(self):
        with pytest.raises(ValueError, match="Only SELECT"):
            validate_sql("DROP TABLE user_predictions", SCHEMAS)

    def test_bad_syntax(self):
        with pytest.raises(ValueError):
            validate_sql("SELEC T blah blah", SCHEMAS)

    def test_empty_schemas(self):
        with pytest.raises(ValueError, match="No table schemas"):
            validate_sql("SELECT 1", {})


class TestBuildSystemPrompt:
    def test_includes_table_names(self):
        schemas = {
            "user_predictions": pa.schema([pa.field("user_id", pa.int64())]),
            "orders": pa.schema([pa.field("order_id", pa.int64())]),
        }
        prompt = build_system_prompt(schemas)
        assert "user_predictions" in prompt
        assert "orders" in prompt

    def test_includes_column_names(self):
        prompt = build_system_prompt(SCHEMAS)
        assert "user_id" in prompt
        assert "conversion_prob" in prompt
        assert "predicted_label" in prompt

    def test_includes_sql_types(self):
        prompt = build_system_prompt(SCHEMAS)
        assert "BIGINT" in prompt
        assert "DOUBLE" in prompt
        assert "INT" in prompt

    def test_includes_business_context(self):
        prompt = build_system_prompt(SCHEMAS)
        assert "e-commerce" in prompt
        assert "business decision-maker" in prompt


requires_llm = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)


def _translate_and_validate(question: str) -> str:
    """Translate a question to SQL via LLM and validate it against DataFusion.

    nl_to_sql now validates internally, so this just calls it and uppercases.
    """
    sql, _hit = nl_to_sql(question, SCHEMAS, use_cache=False)
    return sql.upper()


@requires_llm
class TestBusinessQuestions:
    """End-to-end: business question → LLM → SQL → DataFusion validation.

    Each test sends a realistic business question (the kind a decision-maker
    would ask without knowing about branches or agents) and checks that the
    generated SQL is valid and targets the right columns/logic.
    """

    def test_how_many_expected_to_buy(self):
        sql = _translate_and_validate("How many customers are expected to buy?")
        assert "PREDICTED_LABEL" in sql
        assert "USER_PREDICTIONS" in sql

    def test_conversion_rate_above_threshold(self):
        sql = _translate_and_validate("Is our expected conversion rate above 3%?")
        assert "PREDICTED_LABEL" in sql
        assert "USER_PREDICTIONS" in sql

    def test_which_users_most_likely_to_convert(self):
        sql = _translate_and_validate("Which users are most likely to convert?")
        assert "USER_ID" in sql
        assert "USER_PREDICTIONS" in sql

    def test_highest_conversion_probability(self):
        sql = _translate_and_validate("Who has the highest chance of buying?")
        assert "USER_PREDICTIONS" in sql

    def test_count_low_intent_users(self):
        sql = _translate_and_validate("How many users are unlikely to convert?")
        assert "PREDICTED_LABEL" in sql
        assert "USER_PREDICTIONS" in sql


def _fake_response(sql: str) -> dict:
    """Build a minimal LiteLLM-shaped response containing *sql*."""
    return {"choices": [{"message": {"content": sql}}]}


VALID_SQL = "SELECT COUNT(*) FROM user_predictions"
BAD_COL_SQL = "SELECT bogus FROM user_predictions"


@patch("app.text_to_sql.assert_api_key")
class TestRetryLoop:
    """Deterministic tests for the generate→validate→retry loop.

    Mocks ``completion`` so no LLM calls are made. Tests cover:
    - first-attempt success (no retry),
    - success after a failed first attempt,
    - exhaustion after max_retries failures,
    - invalid max_retries values.
    """

    @patch("app.text_to_sql.completion", return_value=_fake_response(VALID_SQL))
    def test_success_on_first_attempt(self, mock_comp, _key):
        sql, hit = nl_to_sql("how many?", SCHEMAS, max_retries=3, use_cache=False)
        assert sql == VALID_SQL
        assert hit is False
        assert mock_comp.call_count == 1

    @patch(
        "app.text_to_sql.completion",
        side_effect=[_fake_response(BAD_COL_SQL), _fake_response(VALID_SQL)],
    )
    def test_success_after_retry(self, mock_comp, _key):
        sql, _hit = nl_to_sql("how many?", SCHEMAS, max_retries=3, use_cache=False)
        assert sql == VALID_SQL
        assert mock_comp.call_count == 2
        # The second call should contain the feedback message
        second_call_messages = mock_comp.call_args_list[1].kwargs["messages"]
        feedback = second_call_messages[-1]["content"]
        assert "failed validation" in feedback

    @patch(
        "app.text_to_sql.completion",
        side_effect=[_fake_response(BAD_COL_SQL)] * 3,
    )
    def test_exhaustion_raises(self, mock_comp, _key):
        with pytest.raises(ValueError, match="Failed to produce valid SQL after 3"):
            nl_to_sql("how many?", SCHEMAS, max_retries=3, use_cache=False)
        assert mock_comp.call_count == 3

    @patch(
        "app.text_to_sql.completion",
        side_effect=[_fake_response(BAD_COL_SQL), _fake_response(VALID_SQL)],
    )
    def test_max_retries_one_no_second_chance(self, mock_comp, _key):
        with pytest.raises(ValueError, match="Failed to produce valid SQL after 1"):
            nl_to_sql("how many?", SCHEMAS, max_retries=1, use_cache=False)
        assert mock_comp.call_count == 1

    def test_max_retries_zero_raises(self, _key):
        with pytest.raises(ValueError, match="max_retries must be >= 1"):
            nl_to_sql("how many?", SCHEMAS, max_retries=0, use_cache=False)

    def test_max_retries_negative_raises(self, _key):
        with pytest.raises(ValueError, match="max_retries must be >= 1"):
            nl_to_sql("how many?", SCHEMAS, max_retries=-5, use_cache=False)


class TestCache:
    """Tests for the disk-based SQL cache."""

    QUESTION = "test cache question xyz"

    def setup_method(self):
        cache_delete(self.QUESTION)

    def teardown_method(self):
        cache_delete(self.QUESTION)

    def test_roundtrip(self):
        assert cache_get(self.QUESTION) is None
        cache_put(self.QUESTION, VALID_SQL)
        assert cache_get(self.QUESTION) == VALID_SQL

    def test_delete(self):
        cache_put(self.QUESTION, VALID_SQL)
        cache_delete(self.QUESTION)
        assert cache_get(self.QUESTION) is None

    def test_delete_missing_is_noop(self):
        cache_delete("nonexistent question 999")

    @patch("app.text_to_sql.assert_api_key")
    @patch("app.text_to_sql.completion", return_value=_fake_response(VALID_SQL))
    def test_cache_hit_skips_llm(self, mock_comp, _key):
        cache_put(self.QUESTION, VALID_SQL)
        sql, hit = nl_to_sql(self.QUESTION, SCHEMAS, use_cache=True)
        assert hit is True
        assert sql == VALID_SQL
        assert mock_comp.call_count == 0

    @patch("app.text_to_sql.assert_api_key")
    @patch("app.text_to_sql.completion", return_value=_fake_response(VALID_SQL))
    def test_cache_off_skips_read_and_write(self, mock_comp, _key):
        sql, hit = nl_to_sql(self.QUESTION, SCHEMAS, use_cache=False)
        assert hit is False
        assert mock_comp.call_count == 1
        # Should NOT have written to cache
        assert cache_get(self.QUESTION) is None

    @patch("app.text_to_sql.assert_api_key")
    @patch("app.text_to_sql.completion", return_value=_fake_response(VALID_SQL))
    def test_stale_cache_falls_through(self, mock_comp, _key):
        # Write SQL that references a column not in SCHEMAS
        cache_put(self.QUESTION, BAD_COL_SQL)
        sql, hit = nl_to_sql(self.QUESTION, SCHEMAS, use_cache=True)
        # Should have treated stale entry as miss, called LLM, and cached new result
        assert hit is False
        assert sql == VALID_SQL
        assert mock_comp.call_count == 1
        # Stale entry should be replaced with valid SQL
        assert cache_get(self.QUESTION) == VALID_SQL
