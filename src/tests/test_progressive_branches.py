"""Integration test for progressive branch visualization.

Launches a real pipeline (v_10m_py_lr, which has time.sleep(10) in two models)
on a throwaway branch and polls the bauplan API to verify that:

1. The DAG branch is visible immediately (hash == trunk HEAD, no unique commits)
2. A transactional (tx) branch appears while the pipeline runs
3. The tx branch accumulates ICEBERG_TABLE commits one-by-one
4. After the pipeline finishes, the DAG branch has the merged commits

This simulates exactly what the /api/branches endpoint does, proving that
the progressive build-up logic works against the real lakehouse.

Requires cloud credentials (bauplan + AWS).  Run with:

    uv run pytest src/tests/test_progressive_branches.py -v -s
"""

import time
from concurrent.futures import ThreadPoolExecutor, Future

import bauplan
import pytest

from app.lakehouse import (
    get_all_branches,
    get_branch_commits,
    find_branches_forking_from,
    expand_run_commits,
    TRANSACTIONAL_BRANCH_MARKER,
)
from tests.conftest import TEST_PREFIX

PROJECT_DIR = "src/bpln/v_10m_py_lr"
POLL_INTERVAL = 2
MAX_POLL_SECONDS = 300
NAMESPACE = "apo_multiverse"


def _run_pipeline(client, branch_name, namespace):
    """Run the pipeline synchronously (blocking). Called in a background thread."""
    return client.run(
        project_dir=PROJECT_DIR,
        ref=branch_name,
        namespace=namespace,
        cache="off",
        client_timeout=600,
        parameters={"size": 1000000},
    )


@pytest.mark.integration
class TestProgressiveBranches:
    """Verify that in-progress pipeline commits are visible via tx branches."""

    def test_progressive_commit_buildup(self, client, unique_id, username):
        branch_name = f"{username}.{TEST_PREFIX}_progressive_{unique_id}"

        # --- Setup: create a trunk branch and a DAG branch from it ---
        trunk_name = f"{username}.{TEST_PREFIX}_trunk_{unique_id}"
        client.create_branch(trunk_name, from_ref="main")
        try:
            # Get trunk HEAD hash
            trunk_commits = get_branch_commits(
                client, trunk_name, limit=1, lookback_hours=24 * 365
            )
            assert trunk_commits and len(trunk_commits) > 0
            trunk_head_hash = trunk_commits[0]["hash"]
            print(f"\nTrunk HEAD: {trunk_head_hash[:12]}")

            # Create DAG branch from trunk
            client.create_branch(branch_name, from_ref=trunk_name)
            try:
                # ---- Phase 1: Branch exists but has no unique commits ----
                all_br = get_all_branches(client, username)
                dag_branches = [b for b in all_br if b["name"] == branch_name]
                assert len(dag_branches) == 1, f"DAG branch {branch_name} should exist"

                dag_br = dag_branches[0]
                assert dag_br["hash"] == trunk_head_hash, (
                    "Fresh DAG branch should point at trunk HEAD"
                )

                # find_branches_forking_from should NOT find it (no unique commits)
                forking = find_branches_forking_from(
                    client, trunk_head_hash, dag_branches
                )
                assert len(forking) == 0, (
                    "Branch with no unique commits should NOT be found by "
                    "find_branches_forking_from — this is the bug we're fixing"
                )
                print("Phase 1 OK: branch exists at trunk HEAD, not found by fork detection")

                # ---- Phase 2: Launch pipeline in background, poll for tx branch ----
                pool = ThreadPoolExecutor(max_workers=1)
                future: Future = pool.submit(
                    _run_pipeline, client, branch_name, NAMESPACE
                )

                tx_branch_name = None
                tx_commits_seen = []
                dag_commits_after = []
                t0 = time.monotonic()

                while time.monotonic() - t0 < MAX_POLL_SECONDS:
                    if future.done():
                        # Pipeline finished — grab result to check for errors
                        result = future.result()
                        print(f"Pipeline finished: {result.job_status}")
                        break

                    # Look for tx branches
                    all_br_with_tx = get_all_branches(
                        client, username, include_tx=True
                    )
                    tx_candidates = [
                        b for b in all_br_with_tx
                        if b["name"].startswith(branch_name + TRANSACTIONAL_BRANCH_MARKER)
                    ]

                    if tx_candidates:
                        tx_branch_name = tx_candidates[-1]["name"]  # latest
                        # Fetch commits on the tx branch
                        raw = get_branch_commits(
                            client, tx_branch_name,
                            limit=20,
                            lookback_hours=24 * 365,
                            exclude_internal_authors=False,
                        ) or []
                        model_commits = [
                            c for c in raw
                            if "ICEBERG_TABLE" in c["message"]
                            and NAMESPACE in c["message"]
                        ]
                        if len(model_commits) > len(tx_commits_seen):
                            tx_commits_seen = model_commits
                            print(
                                f"  [{time.monotonic() - t0:.0f}s] "
                                f"TX branch {tx_branch_name[-20:]}... "
                                f"has {len(model_commits)} model commit(s):"
                            )
                            for c in model_commits:
                                print(f"    {c['short_hash']} {c['message'][:60]}")

                    time.sleep(POLL_INTERVAL)

                # Wait for pipeline to finish if it hasn't already
                if not future.done():
                    print("Waiting for pipeline to complete...")
                    result = future.result(timeout=MAX_POLL_SECONDS)
                    print(f"Pipeline finished: {result.job_status}")

                pool.shutdown(wait=False)

                assert result.job_status == "SUCCESS", (
                    f"Pipeline failed: {result.error}"
                )

                # ---- Phase 3: Verify we saw progressive commits ----
                print(f"\nTx branch found: {tx_branch_name is not None}")
                print(f"Tx commits seen during execution: {len(tx_commits_seen)}")
                for c in tx_commits_seen:
                    print(f"  {c['short_hash']} {c['message'][:60]}")

                assert tx_branch_name is not None, (
                    "Should have found a tx branch while pipeline was running"
                )
                assert len(tx_commits_seen) >= 1, (
                    "Should have seen at least 1 model commit on the tx branch "
                    "while the pipeline was still running"
                )

                # ---- Phase 4: After merge, DAG branch has commits ----
                dag_commits_after = get_branch_commits(
                    client, branch_name,
                    limit=10,
                    lookback_hours=24 * 365,
                    exclude_internal_authors=False,
                ) or []

                # Should now have unique commits (the merge)
                forking_after = find_branches_forking_from(
                    client, trunk_head_hash,
                    [{"name": branch_name, "hash": "ignored"}],
                )
                assert len(forking_after) == 1, (
                    "After pipeline, DAG branch should be found by fork detection"
                )

                # Expand should give us the individual model commits
                # Filter to only commits after trunk HEAD
                unique = []
                for c in dag_commits_after:
                    if c["hash"] == trunk_head_hash:
                        break
                    unique.append(c)
                unique.reverse()  # oldest first

                expanded = expand_run_commits(
                    client, branch_name, unique, trunk_head_hash
                )
                model_expanded = [
                    c for c in expanded if "ICEBERG_TABLE" in c["message"]
                ]
                print(f"\nExpanded model commits after merge: {len(model_expanded)}")
                for c in model_expanded:
                    print(f"  {c['short_hash']} {c['message'][:60]}")

                # Pipeline has 3 models: bronze_events, user_features, user_predictions
                assert len(model_expanded) >= 3, (
                    f"Expected >= 3 model commits after expansion, got {len(model_expanded)}"
                )

            finally:
                client.delete_branch(branch_name, if_exists=True)
        finally:
            client.delete_branch(trunk_name, if_exists=True)
