"""End-to-end tests: run each demo SQL query through both engines.

Creates synthetic branch data (with deliberately different column orderings
between branches) and verifies that:
  1. Both engines produce the same per-branch results.
  2. The native engine handles schema reordering across branches.
  3. All 4 demo phases (number, boolean, join, set) work end-to-end.

No cloud credentials needed — everything runs locally with in-memory data.
"""

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import datafusion
from pathlib import Path
from multiverse_provider import MultiverseTable


# -- Synthetic data --

# Two branches with DIFFERENT column orderings to catch schema mismatch bugs.
# Branch A: columns in "natural" order
# Branch B: columns alphabetically sorted (different order)

BRANCH_A_PREDICTIONS = pa.table({
    "user_id": pa.array([1, 2, 3, 4, 5], type=pa.int64()),
    "conversion_prob": pa.array([0.9, 0.1, 0.8, 0.05, 0.7], type=pa.float64()),
    "predicted_label": pa.array([1, 0, 1, 0, 1], type=pa.int64()),
})

# Branch B: same columns, DIFFERENT ORDER + different data
BRANCH_B_PREDICTIONS = pa.table({
    "conversion_prob": pa.array([0.6, 0.2, 0.95, 0.3], type=pa.float64()),
    "predicted_label": pa.array([1, 0, 1, 0], type=pa.int64()),
    "user_id": pa.array([10, 20, 30, 40], type=pa.int64()),
})

# Shared dimension table (same for all branches)
ECOMMERCE_USERS = pa.table({
    "user_id": pa.array([1, 2, 3, 4, 5, 10, 20, 30, 40], type=pa.int64()),
    "category_of_interest": pa.array([
        "electronics.smartphone", "fashion", "electronics.smartphone",
        "home", "electronics.smartphone",
        "electronics.smartphone", "fashion", "electronics.smartphone", "home",
    ], type=pa.utf8()),
    "customer_segment": pa.array([
        "returning", "new", "returning", "new", "returning",
        "returning", "new", "returning", "new",
    ], type=pa.utf8()),
})


# -- Demo SQL (same as in test_demo_questions.py / the actual app) --

PHASE_2_SQL = "SELECT COUNT(*) AS n_buyers FROM user_predictions WHERE predicted_label = 1"

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


# -- Helpers --


def run_adhoc(sql, branch_data, shared_tables=None):
    """Run SQL per-branch (ad hoc engine), return combined Arrow table."""
    all_batches = []
    for branch_id, predictions in branch_data.items():
        ctx = datafusion.SessionContext()
        ctx.register_record_batches("user_predictions", [predictions.to_batches()])
        if shared_tables:
            for name, table in shared_tables.items():
                ctx.register_record_batches(name, [table.to_batches()])
        tagged_sql = f"SELECT *, '{branch_id}' AS __branch_id FROM ({sql})"
        batches = ctx.sql(tagged_sql).collect()
        all_batches.extend(batches)
    if not all_batches:
        return None
    return pa.Table.from_batches(all_batches)


def run_native(sql, branch_data, shared_tables=None):
    """Run SQL via MultiverseTable (native engine), return Arrow table."""
    mv = MultiverseTable([
        (branch_id, table.to_batches())
        for branch_id, table in branch_data.items()
    ])
    ctx = datafusion.SessionContext()
    ctx.register_table("user_predictions", mv)
    if shared_tables:
        for name, table in shared_tables.items():
            ctx.register_record_batches(name, [table.to_batches()])
    # Native SQL needs __branch_id GROUP BY for aggregates
    batches = ctx.sql(sql).collect()
    if not batches:
        return None
    return pa.Table.from_batches(batches)


BRANCH_DATA = {"branch_a": BRANCH_A_PREDICTIONS, "branch_b": BRANCH_B_PREDICTIONS}
SHARED = {"ecommerce_users": ECOMMERCE_USERS}


# -- Phase 2: COUNT (number) --


class TestPhase2CountBothEngines:
    """'How many customers will buy tomorrow?' — both engines."""

    def test_adhoc_returns_correct_counts(self):
        result = run_adhoc(PHASE_2_SQL, BRANCH_DATA)
        assert result is not None
        rows = result.to_pydict()
        # Branch A: 3 buyers (users 1,3,5), Branch B: 2 buyers (users 10,30)
        counts = dict(zip(rows["__branch_id"], rows["n_buyers"]))
        assert counts["branch_a"] == 3
        assert counts["branch_b"] == 2

    def test_native_returns_correct_counts(self):
        native_sql = (
            "SELECT __branch_id, COUNT(*) AS n_buyers "
            "FROM user_predictions WHERE predicted_label = 1 "
            "GROUP BY __branch_id"
        )
        result = run_native(native_sql, BRANCH_DATA)
        assert result is not None
        rows = result.to_pydict()
        counts = dict(zip(rows["__branch_id"], rows["n_buyers"]))
        assert counts["branch_a"] == 3
        assert counts["branch_b"] == 2

    def test_both_engines_agree(self):
        adhoc = run_adhoc(PHASE_2_SQL, BRANCH_DATA)
        native_sql = (
            "SELECT __branch_id, COUNT(*) AS n_buyers "
            "FROM user_predictions WHERE predicted_label = 1 "
            "GROUP BY __branch_id"
        )
        native = run_native(native_sql, BRANCH_DATA)

        adhoc_counts = dict(zip(
            adhoc.column("__branch_id").to_pylist(),
            adhoc.column("n_buyers").to_pylist(),
        ))
        native_counts = dict(zip(
            native.column("__branch_id").to_pylist(),
            native.column("n_buyers").to_pylist(),
        ))
        assert adhoc_counts == native_counts


# -- Phase 3: BOOLEAN --


class TestPhase3BooleanBothEngines:
    """'Is conversion rate above 2%?' — both engines."""

    def test_adhoc_returns_booleans(self):
        result = run_adhoc(PHASE_3_SQL, BRANCH_DATA)
        assert result is not None
        rows = result.to_pydict()
        vals = dict(zip(rows["__branch_id"], rows["answer"]))
        # Both branches have >2% conversion (3/5=60%, 2/4=50%)
        assert vals["branch_a"] is True
        assert vals["branch_b"] is True

    def test_native_returns_booleans(self):
        native_sql = (
            "SELECT __branch_id, "
            "CAST(SUM(CASE WHEN predicted_label = 1 THEN 1 ELSE 0 END) AS DOUBLE) "
            "/ COUNT(*) > 0.02 AS answer "
            "FROM user_predictions GROUP BY __branch_id"
        )
        result = run_native(native_sql, BRANCH_DATA)
        assert result is not None
        rows = result.to_pydict()
        vals = dict(zip(rows["__branch_id"], rows["answer"]))
        assert vals["branch_a"] is True
        assert vals["branch_b"] is True

    def test_both_engines_agree(self):
        adhoc = run_adhoc(PHASE_3_SQL, BRANCH_DATA)
        native_sql = (
            "SELECT __branch_id, "
            "CAST(SUM(CASE WHEN predicted_label = 1 THEN 1 ELSE 0 END) AS DOUBLE) "
            "/ COUNT(*) > 0.02 AS answer "
            "FROM user_predictions GROUP BY __branch_id"
        )
        native = run_native(native_sql, BRANCH_DATA)

        adhoc_vals = dict(zip(
            adhoc.column("__branch_id").to_pylist(),
            adhoc.column("answer").to_pylist(),
        ))
        native_vals = dict(zip(
            native.column("__branch_id").to_pylist(),
            native.column("answer").to_pylist(),
        ))
        assert adhoc_vals == native_vals


# -- Phase 4: JOIN --


class TestPhase4JoinBothEngines:
    """'How many smartphone shoppers will convert?' — both engines."""

    def test_adhoc_returns_correct_counts(self):
        result = run_adhoc(PHASE_4_SQL, BRANCH_DATA, SHARED)
        assert result is not None
        rows = result.to_pydict()
        counts = dict(zip(rows["__branch_id"], rows["n_buyers"]))
        # Branch A: users 1,3,5 are buyers; users 1,3,5 are smartphone → 3
        # Branch B: users 10,30 are buyers; users 10,30 are smartphone → 2
        assert counts["branch_a"] == 3
        assert counts["branch_b"] == 2

    def test_native_returns_correct_counts(self):
        native_sql = (
            "SELECT __branch_id, COUNT(*) AS n_buyers "
            "FROM user_predictions p "
            "JOIN (SELECT DISTINCT user_id FROM ecommerce_users "
            "WHERE category_of_interest = 'electronics.smartphone') u "
            "ON p.user_id = u.user_id "
            "WHERE p.predicted_label = 1 "
            "GROUP BY __branch_id"
        )
        result = run_native(native_sql, BRANCH_DATA, SHARED)
        assert result is not None
        rows = result.to_pydict()
        counts = dict(zip(rows["__branch_id"], rows["n_buyers"]))
        assert counts["branch_a"] == 3
        assert counts["branch_b"] == 2

    def test_both_engines_agree(self):
        adhoc = run_adhoc(PHASE_4_SQL, BRANCH_DATA, SHARED)
        native_sql = (
            "SELECT __branch_id, COUNT(*) AS n_buyers "
            "FROM user_predictions p "
            "JOIN (SELECT DISTINCT user_id FROM ecommerce_users "
            "WHERE category_of_interest = 'electronics.smartphone') u "
            "ON p.user_id = u.user_id "
            "WHERE p.predicted_label = 1 "
            "GROUP BY __branch_id"
        )
        native = run_native(native_sql, BRANCH_DATA, SHARED)

        adhoc_counts = dict(zip(
            adhoc.column("__branch_id").to_pylist(),
            adhoc.column("n_buyers").to_pylist(),
        ))
        native_counts = dict(zip(
            native.column("__branch_id").to_pylist(),
            native.column("n_buyers").to_pylist(),
        ))
        assert adhoc_counts == native_counts


# -- Phase 5: SET --


class TestPhase5SetBothEngines:
    """'Which customers should we target?' — both engines."""

    def test_adhoc_returns_user_ids(self):
        result = run_adhoc(PHASE_5_SQL, BRANCH_DATA)
        assert result is not None
        rows = result.to_pydict()
        a_ids = set(
            uid for uid, br in zip(rows["user_id"], rows["__branch_id"])
            if br == "branch_a"
        )
        b_ids = set(
            uid for uid, br in zip(rows["user_id"], rows["__branch_id"])
            if br == "branch_b"
        )
        assert a_ids == {1, 3, 5}
        assert b_ids == {10, 30}

    def test_native_returns_user_ids(self):
        native_sql = (
            "SELECT __branch_id, user_id "
            "FROM user_predictions "
            "WHERE predicted_label = 1 "
            "ORDER BY conversion_prob DESC"
        )
        result = run_native(native_sql, BRANCH_DATA)
        assert result is not None
        rows = result.to_pydict()
        a_ids = set(
            uid for uid, br in zip(rows["user_id"], rows["__branch_id"])
            if br == "branch_a"
        )
        b_ids = set(
            uid for uid, br in zip(rows["user_id"], rows["__branch_id"])
            if br == "branch_b"
        )
        assert a_ids == {1, 3, 5}
        assert b_ids == {10, 30}


# -- Schema reordering stress test --


class TestSchemaReordering:
    """Verify MultiverseTable handles branches with different column orderings."""

    def test_reordered_columns_produce_correct_results(self):
        """Branch B has columns in different order than Branch A."""
        # This is the key test — BRANCH_B_PREDICTIONS has columns in
        # (conversion_prob, predicted_label, user_id) order while
        # BRANCH_A_PREDICTIONS has (user_id, conversion_prob, predicted_label).
        mv = MultiverseTable([
            ("a", BRANCH_A_PREDICTIONS.to_batches()),
            ("b", BRANCH_B_PREDICTIONS.to_batches()),
        ])
        ctx = datafusion.SessionContext()
        ctx.register_table("t", mv)

        result = pa.Table.from_batches(
            ctx.sql("SELECT __branch_id, COUNT(*) AS n FROM t GROUP BY __branch_id").collect()
        )
        counts = dict(zip(
            result.column("__branch_id").to_pylist(),
            result.column("n").to_pylist(),
        ))
        assert counts["a"] == 5
        assert counts["b"] == 4

    def test_reordered_columns_filter_correctly(self):
        """WHERE clause on a column that's at different positions across branches."""
        mv = MultiverseTable([
            ("a", BRANCH_A_PREDICTIONS.to_batches()),
            ("b", BRANCH_B_PREDICTIONS.to_batches()),
        ])
        ctx = datafusion.SessionContext()
        ctx.register_table("t", mv)

        result = pa.Table.from_batches(
            ctx.sql(
                "SELECT __branch_id, user_id FROM t WHERE predicted_label = 1"
            ).collect()
        )
        a_ids = set(
            uid for uid, br in zip(
                result.column("user_id").to_pylist(),
                result.column("__branch_id").to_pylist(),
            ) if br == "a"
        )
        b_ids = set(
            uid for uid, br in zip(
                result.column("user_id").to_pylist(),
                result.column("__branch_id").to_pylist(),
            ) if br == "b"
        )
        assert a_ids == {1, 3, 5}
        assert b_ids == {10, 30}

    def test_join_with_reordered_branches(self):
        """JOIN query works when branch schemas have different column orders."""
        mv = MultiverseTable([
            ("a", BRANCH_A_PREDICTIONS.to_batches()),
            ("b", BRANCH_B_PREDICTIONS.to_batches()),
        ])
        ctx = datafusion.SessionContext()
        ctx.register_table("user_predictions", mv)
        ctx.register_record_batches("ecommerce_users", [ECOMMERCE_USERS.to_batches()])

        sql = (
            "SELECT __branch_id, COUNT(*) AS n_buyers "
            "FROM user_predictions p "
            "JOIN (SELECT DISTINCT user_id FROM ecommerce_users "
            "WHERE category_of_interest = 'electronics.smartphone') u "
            "ON p.user_id = u.user_id "
            "WHERE p.predicted_label = 1 "
            "GROUP BY __branch_id"
        )
        result = pa.Table.from_batches(ctx.sql(sql).collect())
        counts = dict(zip(
            result.column("__branch_id").to_pylist(),
            result.column("n_buyers").to_pylist(),
        ))
        assert counts["a"] == 3
        assert counts["b"] == 2
