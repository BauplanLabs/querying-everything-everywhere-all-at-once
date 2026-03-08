import bauplan
from bauplan.standard_expectations import (
    expect_column_accepted_values,
    expect_column_mean_greater_or_equal_than,
    expect_column_mean_smaller_than,
    expect_column_no_nulls,
)


@bauplan.expectation()
@bauplan.python("3.11")
def test_bronze_no_null_keys(data=bauplan.Model("bronze_events")):
    """user_id and event_time_parsed must never be null after cleaning."""
    for col in ["user_id", "event_time_parsed"]:
        result = expect_column_no_nulls(data, col)
        assert result, f"{col} contains null values"
    return True


@bauplan.expectation()
@bauplan.python("3.11")
def test_bronze_valid_event_types(data=bauplan.Model("bronze_events")):
    """event_type must be one of the known e-commerce actions."""
    result = expect_column_accepted_values(
        data, "event_type", ["view", "cart", "purchase", "remove_from_cart"]
    )
    assert result, "event_type contains unexpected values"
    return result


@bauplan.expectation()
@bauplan.python("3.11")
def test_silver_no_null_user_id(
    data=bauplan.Model("user_features"),
):
    """user_id must not be null in feature table."""
    result = expect_column_no_nulls(data, "user_id")
    assert result, "user_id contains null values"
    return result


@bauplan.expectation()
@bauplan.python("3.11")
def test_silver_converted_binary(
    data=bauplan.Model("user_features"),
):
    """converted label must be 0 or 1."""
    result = expect_column_accepted_values(data, "converted", [0, 1])
    assert result, "converted contains values other than 0 or 1"
    return result


@bauplan.expectation()
@bauplan.python("3.11")
def test_gold_no_null_user_id(data=bauplan.Model("user_predictions")):
    """user_id must not be null in predictions."""
    result = expect_column_no_nulls(data, "user_id")
    assert result, "user_id contains null values"
    return result


@bauplan.expectation()
@bauplan.python("3.11")
def test_gold_probs_in_range(data=bauplan.Model("user_predictions")):
    """conversion_prob must be between 0 and 1."""
    result = expect_column_mean_smaller_than(data, "conversion_prob", 1.01)
    assert result, "conversion_prob mean exceeds 1"
    result = expect_column_mean_greater_or_equal_than(data, "conversion_prob", 0.0)
    assert result, "conversion_prob mean is negative"
    return True
