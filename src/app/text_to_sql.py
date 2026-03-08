"""
Text-to-SQL translation and validation.

Uses LiteLLM to convert natural language questions into SQL.
Validates output with Apache DataFusion's query planner.
Includes a simple disk cache keyed on the normalized question text.
"""

import hashlib
import logging
import os
import re
from pathlib import Path

import pyarrow as pa
from dotenv import load_dotenv
from litellm import completion

from helpers import build_datafusion_context

logger = logging.getLogger(__name__)

# Load .env from project root
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

DEFAULT_MODEL: str = "gpt-5.2"

TABLE_DESCRIPTIONS: dict[str, str] = {
    "bronze_events": (
        "Raw e-commerce events (one row per event). HISTORICAL data only — "
        "page views, add-to-carts, and purchases that already happened. "
        "Use a DISTINCT subquery on user_id when joining with per-user tables "
        "to avoid row duplication. Do NOT use this table for forward-looking "
        "questions about tomorrow or the future."
    ),
    "user_features": (
        "Engineered features per user (one row per user). HISTORICAL data — "
        "derived from past bronze_events. Columns:\n"
        "  - user_id: unique user identifier\n"
        "  - session_count: number of browsing sessions in the observation window\n"
        "  - avg_session_duration_min: average session length in minutes\n"
        "  - total_views: number of product page views\n"
        "  - total_carts: number of add-to-cart events\n"
        "  - total_purchases_pre: number of past purchases (before the prediction cutoff)\n"
        "  - avg_price_viewed: average price of viewed products\n"
        "  - n_brands: number of distinct brands browsed\n"
        "  - converted: GROUND TRUTH label (1 = actually purchased, 0 = did not). "
        "This is a historical fact, NOT a prediction. Do NOT use this for "
        "forward-looking questions about tomorrow."
    ),
    "user_predictions": (
        "ML predictions per user (one row per user). FORWARD-LOOKING — "
        "this is the output of a predictive model about TOMORROW / the future. "
        "Use this table for any question about what WILL happen, who WILL buy, "
        "expected conversions, or targeting decisions. Columns:\n"
        "  - user_id: unique user identifier\n"
        "  - conversion_prob: predicted probability of purchasing tomorrow (0.0 to 1.0)\n"
        "  - predicted_label: binary prediction (1 = likely buyer tomorrow, "
        "0 = unlikely). Use predicted_label = 1 for 'will buy' / 'expected to buy'."
    ),
    "ecommerce_users": (
        "Per-user dimension table (one row per user, shared across all branches). "
        "Use this for demographic joins. Columns:\n"
        "  - user_id: unique user identifier\n"
        "  - category_of_interest: dominant product category "
        "(e.g. 'electronics.smartphone', 'computers.notebook')\n"
        "  - customer_segment: 'high', 'medium', or 'low' based on spending"
    ),
}


_SYSTEM_PROMPT_TEMPLATE: str = """\
You are a Text-to-SQL translator for an e-commerce analytics system.

Context: An e-commerce company deployed autonomous data agents to answer the
question "which users are most likely to convert next week?". Each agent built
its own predictive pipeline independently; their results now coexist in the
data system. The person asking questions is a business decision-maker — they
do not know how many agents ran, what branches exist, or any implementation
details. They simply have a business question and expect a direct answer from
whatever data is available. Treat every question as a straightforward business
query against the tables below.

Return ONLY a SQL query (no markdown, no comments, no explanation).

Available tables:
{table_descriptions}

Rules:
- Only output a single SELECT statement.
- Do NOT use INSERT/UPDATE/DELETE/CREATE/DROP.
- Do NOT end with a semicolon.
- CRITICAL: For any question about "tomorrow", "will buy", "expected to buy",
  "predicted", "forecast", or future-looking intent, you MUST query
  user_predictions (predicted_label, conversion_prob). NEVER use user_features
  or bronze_events for forward-looking questions — those contain only
  historical data.
- If the question asks "expected to buy" or "will buy", use predicted_label = 1
  from user_predictions.
- If the question asks for conversion rate or percentage, compute it as:
    CAST(SUM(CASE WHEN predicted_label=1 THEN 1 ELSE 0 END) AS DOUBLE) / COUNT(*)
  from user_predictions. Return it as a floating point number named `pct`.
- If the question asks "is ... above X", return a boolean column named `answer`.
- If the question asks "which customers", return a single column `user_id`.
- When joining with ecommerce_users, join ON user_id.
- When joining bronze_events with a per-user table, always deduplicate first:
    JOIN (SELECT DISTINCT user_id, col FROM bronze_events) b ON ...
- If the question is ambiguous, choose the simplest interpretation consistent with the schema.
{engine_rules}"""

_ADHOC_RULES: str = ""

_NATIVE_RULES: str = """
IMPORTANT — Native engine mode:
The user_predictions table has a hidden column called __branch_id that
identifies which data-agent branch produced each row. You MUST include
__branch_id in your SELECT list and add GROUP BY __branch_id so the query
returns one result row per branch. For set queries (returning user_id rows),
add __branch_id to the SELECT but do NOT group by it.

IMPORTANT — Top-K per branch:
If the user asks for a top-K list (e.g., "top 50 customers"), DO NOT use a
global ORDER BY ... LIMIT k. You must return top-K per branch by ranking
within each __branch_id using a window function:
  ROW_NUMBER() OVER (PARTITION BY __branch_id ORDER BY <score> DESC) AS rn
and then filtering with WHERE rn <= k.

Examples:

Q: "How many customers will buy tomorrow?"
Ad hoc SQL:
  SELECT COUNT(*) AS n_buyers FROM user_predictions WHERE predicted_label = 1
Native SQL:
  SELECT __branch_id, COUNT(*) AS n_buyers FROM user_predictions WHERE predicted_label = 1 GROUP BY __branch_id

Q: "Is tomorrow's conversion rate above 2%?"
Ad hoc SQL:
  SELECT CAST(SUM(CASE WHEN predicted_label = 1 THEN 1 ELSE 0 END) AS DOUBLE) / CAST(COUNT(*) AS DOUBLE) > 0.02 AS answer FROM user_predictions
Native SQL:
  SELECT __branch_id, CAST(SUM(CASE WHEN predicted_label = 1 THEN 1 ELSE 0 END) AS DOUBLE) / CAST(COUNT(*) AS DOUBLE) > 0.02 AS answer FROM user_predictions GROUP BY __branch_id

Q: "How many smartphone shoppers will convert tomorrow?"
Ad hoc SQL:
  SELECT COUNT(*) AS n_buyers FROM user_predictions p JOIN (SELECT DISTINCT user_id FROM ecommerce_users WHERE category_of_interest = 'electronics.smartphone') u ON p.user_id = u.user_id WHERE p.predicted_label = 1
Native SQL:
  SELECT __branch_id, COUNT(*) AS n_buyers FROM user_predictions p JOIN (SELECT DISTINCT user_id FROM ecommerce_users WHERE category_of_interest = 'electronics.smartphone') u ON p.user_id = u.user_id WHERE p.predicted_label = 1 GROUP BY __branch_id

Q: "Which customers should we target tomorrow? (top 50)"
Ad hoc SQL:
  SELECT user_id FROM user_predictions WHERE predicted_label = 1 ORDER BY conversion_prob DESC LIMIT 50
Native SQL:
  WITH ranked AS (
    SELECT __branch_id, user_id, ROW_NUMBER() OVER (PARTITION BY __branch_id ORDER BY conversion_prob DESC) AS rn
    FROM user_predictions WHERE predicted_label = 1
  )
  SELECT __branch_id, user_id FROM ranked WHERE rn <= 50

Follow this pattern: add __branch_id to SELECT, and for aggregate queries add GROUP BY __branch_id.
For set queries with ORDER BY + LIMIT (top-K), never use a global LIMIT — use ROW_NUMBER() OVER (PARTITION BY __branch_id ...) to get top-K per branch.
"""

USER_PROMPT_TEMPLATE: str = """\
Natural language question:
{question}

Write the SQL now."""

_FORBIDDEN: re.Pattern[str] = re.compile(
    r"\b(insert|update|delete|create|drop|alter|copy|grant|revoke)\b", re.I
)

# Mapping from PyArrow types to SQL-friendly names for the system prompt
_ARROW_TO_SQL_NAME: dict[pa.DataType, str] = {
    pa.int32(): "INT",
    pa.int64(): "BIGINT",
    pa.float32(): "FLOAT",
    pa.float64(): "DOUBLE",
    pa.string(): "VARCHAR",
    pa.large_string(): "VARCHAR",
    pa.bool_(): "BOOLEAN",
    pa.timestamp("us"): "TIMESTAMP",
    pa.date32(): "DATE",
}


def _arrow_type_to_sql_name(arrow_type: pa.DataType) -> str:
    """Convert a PyArrow type to a SQL-friendly name for the system prompt."""
    return _ARROW_TO_SQL_NAME.get(arrow_type, str(arrow_type).upper())


def build_system_prompt(
    schemas: dict[str, pa.Schema], engine: str = "adhoc"
) -> str:
    """Build a system prompt dynamically from table schemas and descriptions."""
    parts: list[str] = []
    for table_name, schema in schemas.items():
        cols = ", ".join(
            f"{field.name} {_arrow_type_to_sql_name(field.type)}" for field in schema
        )
        desc = TABLE_DESCRIPTIONS.get(table_name, "")
        header = f"- {table_name}({cols})"
        if desc:
            header += f"\n  {desc}"
        parts.append(header)
    table_descriptions = "\n".join(parts)
    engine_rules = _NATIVE_RULES if engine == "native" else _ADHOC_RULES
    return _SYSTEM_PROMPT_TEMPLATE.format(
        table_descriptions=table_descriptions,
        engine_rules=engine_rules,
    )


def assert_api_key() -> None:
    """Assert that OPENAI_API_KEY is set in the environment."""
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY not found. Set it in .env or as an environment variable."
        )


DEFAULT_MAX_RETRIES: int = 3


def _strip_markdown_fences(text: str) -> str:
    """Strip ```sql / ``` wrappers if the model returns fenced code."""
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [line for line in lines if not line.startswith("```")]
        return "\n".join(lines).strip()
    return text


_CACHE_DIR = Path(__file__).parent / ".sql_cache"


def _cache_key(question: str, engine: str = "adhoc") -> str:
    """Deterministic cache key from the normalized question text and engine."""
    normalized = question.strip().lower() + "|" + engine
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


def cache_get(question: str, engine: str = "adhoc") -> str | None:
    """Return cached SQL for a question, or None if not cached."""
    path = _CACHE_DIR / _cache_key(question, engine)
    if path.exists():
        sql = path.read_text().strip()
        logger.info("Cache hit for question: %s", question[:60])
        return sql
    return None


def cache_put(question: str, sql: str, engine: str = "adhoc") -> None:
    """Write a validated SQL result to the disk cache."""
    _CACHE_DIR.mkdir(exist_ok=True)
    path = _CACHE_DIR / _cache_key(question, engine)
    path.write_text(sql)
    logger.info("Cached SQL for question: %s", question[:60])


def cache_delete(question: str, engine: str = "adhoc") -> None:
    """Remove a stale cache entry."""
    path = _CACHE_DIR / _cache_key(question, engine)
    path.unlink(missing_ok=True)


def nl_to_sql(
    question: str,
    schemas: dict[str, pa.Schema],
    model: str = DEFAULT_MODEL,
    max_retries: int = DEFAULT_MAX_RETRIES,
    use_cache: bool = True,
    engine: str = "adhoc",
) -> tuple[str, bool]:
    """Translate a natural language question to validated SQL via LiteLLM.

    Runs a generate-then-validate loop: the LLM produces SQL, DataFusion
    checks it, and if validation fails the error is fed back to the LLM for
    another attempt. Returns the first SQL that passes validation.

    When ``engine="native"``, the prompt instructs the LLM to include
    ``__branch_id`` in SELECT and GROUP BY so the native engine can run
    a single query across all branches.

    Returns:
        (sql, cache_hit) tuple.

    Raises:
        ValueError: If no valid SQL is produced within *max_retries* attempts.
    """
    if use_cache:
        cached = cache_get(question, engine)
        if cached is not None:
            try:
                validate_sql(cached, schemas, engine=engine)
                return cached, True
            except ValueError:
                logger.warning("Stale cache entry for: %s", question[:60])
                cache_delete(question, engine)

    if max_retries < 1:
        raise ValueError("max_retries must be >= 1")
    assert_api_key()
    system_prompt = build_system_prompt(schemas, engine=engine)
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(question=question)},
    ]

    for attempt in range(1, max_retries + 1):
        resp = completion(model=model, messages=messages, temperature=0.0)
        sql = _strip_markdown_fences(resp["choices"][0]["message"]["content"].strip())

        try:
            validate_sql(sql, schemas, engine=engine)
            if use_cache:
                cache_put(question, sql, engine)
            return sql, False
        except ValueError as exc:
            if attempt == max_retries:
                raise ValueError(
                    f"Failed to produce valid SQL after {max_retries} attempts. "
                    f"Last error: {exc}"
                ) from exc
            # Feed the error back so the LLM can self-correct
            messages.append({"role": "assistant", "content": sql})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"That SQL failed validation with the following error:\n"
                        f"{exc}\n\n"
                        f"Fix the query and return ONLY the corrected SQL."
                    ),
                },
            )

    # Unreachable, but keeps the type checker happy
    raise ValueError("Exhausted retries without returning.")


def validate_sql(
    sql: str, schemas: dict[str, pa.Schema], engine: str = "adhoc"
) -> None:
    """Validate that SQL is a safe SELECT using DataFusion's query planner.

    Creates a DataFusion SessionContext with empty tables matching the provided
    schemas, then attempts to produce a logical plan. If DataFusion rejects the
    query (bad table, bad column, bad syntax), it raises ValueError.

    When ``engine="native"``, adds ``__branch_id`` (VARCHAR) to the
    ``user_predictions`` schema so the LLM-generated SQL can reference it.

    Raises:
        ValueError: If the query is forbidden, syntactically invalid, or
            references unknown tables/columns.
    """
    if _FORBIDDEN.search(sql):
        raise ValueError("Only SELECT queries are allowed.")

    if not schemas:
        raise ValueError("No table schemas provided for validation.")

    validation_schemas = schemas
    if engine == "native":
        # Add __branch_id to multiverse tables for validation
        validation_schemas = dict(schemas)
        for table_name in ("user_predictions",):
            if table_name in validation_schemas:
                existing = validation_schemas[table_name]
                validation_schemas[table_name] = pa.schema(
                    list(existing) + [pa.field("__branch_id", pa.string())]
                )

    ctx = build_datafusion_context(validation_schemas)
    try:
        ctx.sql(sql)
    except Exception as e:
        raise ValueError(f"SQL validation failed: {e}") from e
