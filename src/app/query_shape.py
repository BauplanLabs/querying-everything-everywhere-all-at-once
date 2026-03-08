"""
Query shape classifier.

Inspects the output schema and SQL text of a user query to determine
its result type (NUMBER, BOOLEAN, SET) and guard against patterns that
the ad hoc engine cannot handle.
"""

import re
from dataclasses import dataclass
from enum import Enum

import pyarrow as pa

from helpers import build_datafusion_context


class ResultType(Enum):
    NUMBER = "number"
    BOOLEAN = "boolean"
    SET = "set"


@dataclass
class QueryShape:
    result_type: ResultType
    value_column: str | None  # for NUMBER/BOOLEAN — the single output column
    set_column: str | None  # for SET — the identifier column
    expects_single_row: bool
    max_rows_hint: int | None  # from LIMIT if present


class UnsupportedQueryError(Exception):
    def __init__(self, reason: str, hint: str) -> None:
        self.reason = reason
        self.hint = hint
        super().__init__(reason)


# ---- SQL feature detection ----

_AGGREGATE_PATTERN = re.compile(r"\b(COUNT|SUM|AVG|MIN|MAX)\s*\(", re.IGNORECASE)
_EXISTS_PATTERN = re.compile(r"\bEXISTS\s*\(", re.IGNORECASE)
_CTE_PATTERN = re.compile(r"\bWITH\s+", re.IGNORECASE)
_WINDOW_PATTERN = re.compile(r"\bOVER\s*\(", re.IGNORECASE)
_GROUP_BY_PATTERN = re.compile(r"\bGROUP\s+BY\b", re.IGNORECASE)
_ORDER_BY_PATTERN = re.compile(r"\bORDER\s+BY\b", re.IGNORECASE)
_LIMIT_PATTERN = re.compile(r"\bLIMIT\s+(\d+)", re.IGNORECASE)
_PARTITION_BY_BRANCH_PATTERN = re.compile(
    r"\bPARTITION\s+BY\s+__branch_id\b", re.IGNORECASE
)


@dataclass
class _SQLFeatures:
    """Extracted SQL features used for classification."""

    has_agg: bool
    has_exists: bool
    has_group_by: bool
    has_order_by: bool
    has_cte: bool
    has_window: bool
    limit_val: int | None


def _has_partition_by_branch(sql: str) -> bool:
    return bool(_PARTITION_BY_BRANCH_PATTERN.search(sql))


def _extract_sql_features(sql: str) -> _SQLFeatures:
    return _SQLFeatures(
        has_agg=bool(_AGGREGATE_PATTERN.search(sql)),
        has_exists=bool(_EXISTS_PATTERN.search(sql)),
        has_group_by=bool(_GROUP_BY_PATTERN.search(sql)),
        has_order_by=bool(_ORDER_BY_PATTERN.search(sql)),
        has_cte=bool(_CTE_PATTERN.search(sql)),
        has_window=bool(_WINDOW_PATTERN.search(sql)),
        limit_val=int(m.group(1)) if (m := _LIMIT_PATTERN.search(sql)) else None,
    )


# ---- Strict-mode validation ----


def _validate_strict(features: _SQLFeatures) -> None:
    if features.has_cte:
        raise UnsupportedQueryError(
            reason="CTEs (WITH clauses) are not supported by the ad hoc engine.",
            hint="Rewrite as a single SELECT, or switch to the Native engine.",
        )
    if features.has_window:
        raise UnsupportedQueryError(
            reason="Window functions (OVER) are not supported by the ad hoc engine.",
            hint="Try a simple aggregate, or switch to the Native engine.",
        )


# ---- Output schema planning ----


def _plan_output_schema(
    sql: str, schemas: dict[str, pa.Schema], engine: str = "adhoc"
) -> list[tuple[str, pa.DataType]]:
    planning_schemas = schemas
    if engine == "native":
        planning_schemas = dict(schemas)
        for table_name in ("user_predictions",):
            if table_name in planning_schemas:
                existing = planning_schemas[table_name]
                planning_schemas[table_name] = pa.schema(
                    list(existing) + [pa.field("__branch_id", pa.string())]
                )
    ctx = build_datafusion_context(planning_schemas)
    try:
        df = ctx.sql(sql)
    except Exception as e:
        raise UnsupportedQueryError(
            reason=f"SQL planning failed: {e}",
            hint="Check your SQL syntax and table/column names.",
        ) from e

    return [
        (field.name, field.type) for field in df.schema() if field.name != "__branch_id"
    ]


# ---- Result type inference ----


def _infer_result_type(
    col_name: str,
    col_type: pa.DataType,
    features: _SQLFeatures,
    strict: bool,
) -> QueryShape:
    if features.has_group_by and not features.has_agg:
        raise UnsupportedQueryError(
            reason="GROUP BY without a final aggregation produces multiple rows.",
            hint="Add a final aggregate: SELECT COUNT(*) FROM (...) or wrap in a subquery.",
        )

    if pa.types.is_boolean(col_type):
        return QueryShape(
            result_type=ResultType.BOOLEAN,
            value_column=col_name,
            set_column=None,
            expects_single_row=True,
            max_rows_hint=features.limit_val,
        )

    is_numeric = pa.types.is_integer(col_type) or pa.types.is_floating(col_type)
    if is_numeric and (features.has_agg or features.has_exists):
        return QueryShape(
            result_type=ResultType.NUMBER,
            value_column=col_name,
            set_column=None,
            expects_single_row=True,
            max_rows_hint=features.limit_val,
        )

    if not features.has_agg:
        if strict and features.has_order_by and features.limit_val is None:
            raise UnsupportedQueryError(
                reason="ORDER BY without LIMIT on a set query could return unbounded rows.",
                hint="Add a LIMIT clause, e.g. ORDER BY conversion_prob DESC LIMIT 50",
            )
        return QueryShape(
            result_type=ResultType.SET,
            value_column=None,
            set_column=col_name,
            expects_single_row=False,
            max_rows_hint=features.limit_val,
        )

    # Fallback: aggregate on non-numeric column → SET
    return QueryShape(
        result_type=ResultType.SET,
        value_column=None,
        set_column=col_name,
        expects_single_row=False,
        max_rows_hint=features.limit_val,
    )


# ---- Public API ----


def classify(
    sql: str, *, schemas: dict[str, pa.Schema], strict: bool = True,
    engine: str = "adhoc",
) -> QueryShape:
    """Classify a SQL query into a QueryShape.

    Phases:
        1. Extract SQL features (aggregates, CTEs, limits, etc.)
        2. Validate strict-mode constraints
        3. Plan output schema via DataFusion (no execution)
        4. Infer result type from output column + SQL features

    Args:
        sql: The SQL query to classify.
        schemas: {table_name: pa.Schema} — the tables available for planning.
        strict: When True (default), reject CTEs and window functions that
            break the ad hoc engine.

    Raises:
        UnsupportedQueryError: If the query shape cannot be handled.
    """
    features = _extract_sql_features(sql)

    if strict:
        _validate_strict(features)

    output_cols = _plan_output_schema(sql, schemas, engine=engine)

    if len(output_cols) == 0:
        raise UnsupportedQueryError(
            reason="Query produces no output columns.",
            hint="Try SELECT COUNT(*) FROM user_predictions",
        )

    if len(output_cols) > 1:
        raise UnsupportedQueryError(
            reason=f"Multi-column output ({', '.join(c[0] for c in output_cols)}) is not supported.",
            hint="Try `SELECT COUNT(*) FROM user_predictions` or `SELECT user_id FROM user_predictions LIMIT 50`",
        )

    col_name, col_type = output_cols[0]
    shape = _infer_result_type(col_name, col_type, features, strict)

    # Native engine guardrail: a global LIMIT on a SET query returns top-K
    # across ALL branches instead of top-K per branch.  The correct pattern
    # is ROW_NUMBER() OVER (PARTITION BY __branch_id ...).
    if (
        engine == "native"
        and shape.result_type == ResultType.SET
        and features.has_order_by
        and features.limit_val is not None
        and not _has_partition_by_branch(sql)
    ):
        raise UnsupportedQueryError(
            reason="Global LIMIT on a set query in native mode returns top-K across all branches, not per branch.",
            hint=(
                "Use ROW_NUMBER() OVER (PARTITION BY __branch_id ORDER BY ...) "
                "to get top-K per branch. Example:\n"
                "  WITH ranked AS (\n"
                "    SELECT __branch_id, user_id,\n"
                "           ROW_NUMBER() OVER (PARTITION BY __branch_id ORDER BY conversion_prob DESC) AS rn\n"
                "    FROM user_predictions WHERE predicted_label = 1\n"
                "  )\n"
                "  SELECT __branch_id, user_id FROM ranked WHERE rn <= 50"
            ),
        )

    return shape
