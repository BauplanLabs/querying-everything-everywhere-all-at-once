"""
Integration tests for lakehouse commit/branch navigation.

These tests create real branches, tables, and tags in the bauplan lakehouse
and verify that our lakehouse module can see them correctly.

Configuration:
    Set BAUPLAN_TEST_S3_PATH to a parquet file accessible by bauplan.
    Optionally set BAUPLAN_TEST_NAMESPACE to override the default namespace.

    Either add them to .env at the project root:

        BAUPLAN_TEST_S3_PATH="s3://bucket/path/to/file.parquet"
        BAUPLAN_TEST_NAMESPACE="my_namespace"

    Or pass as env vars:

        BAUPLAN_TEST_S3_PATH=s3://... uv run pytest src/tests/ -v
"""

from datetime import datetime, timedelta, timezone

from app.lakehouse import (
    _recent_cutoff,
    get_branch_commits,
    get_all_branches,
    find_branches_forking_from,
    expand_run_commits,
)
from tests.conftest import NAMESPACE, TEST_PREFIX

SETUP_PROJECT_DIR = "src/bpln/v_10m_py_lr"


# ---------- unit tests (no client needed) ----------


def test_recent_cutoff_returns_utc_datetime():
    """_recent_cutoff should return a timezone-aware UTC datetime in the past."""
    cutoff = _recent_cutoff(24)
    now = datetime.now(timezone.utc)
    assert cutoff.tzinfo is not None
    assert now - timedelta(hours=25) < cutoff < now


def test_recent_cutoff_zero_hours():
    """_recent_cutoff(0) should return approximately now."""
    cutoff = _recent_cutoff(0)
    now = datetime.now(timezone.utc)
    assert abs((now - cutoff).total_seconds()) < 2


# ---------- branch & commit tests ----------


def test_create_branch_appears_in_listing(client, test_branch, username):
    """A freshly created branch should show up in get_all_branches."""
    branches = get_all_branches(client, username)
    names = [b["name"] for b in branches]
    assert test_branch in names


def test_branch_commits_returns_dicts_with_expected_keys(client, test_branch):
    """get_branch_commits should return dicts with the right shape."""
    commits = get_branch_commits(client, test_branch, limit=3, lookback_hours=24 * 365)
    assert len(commits) > 0
    for c in commits:
        assert "hash" in c
        assert "short_hash" in c
        assert "message" in c
        assert "author" in c
        assert "date" in c
        assert "parent_hash" in c
        assert "parent_hashes" in c
        assert len(c["short_hash"]) == 10


def test_find_branches_forking_from(client, s3_path, unique_id, username):
    """Create an isolated base branch, commit on it, fork a child branch,
    and verify find_branches_forking_from finds ONLY the child — not
    random branches that share main history.
    """
    base_name = f"{username}.{TEST_PREFIX}_base_{unique_id}"
    child_name = f"{username}.{TEST_PREFIX}_child_{unique_id}"
    table_base = f"{TEST_PREFIX}_tbase_{unique_id}"
    table_child = f"{TEST_PREFIX}_tchild_{unique_id}"

    # Step 1: create a base branch from main and make a commit on it
    # so it has a unique commit that no other branch shares.
    client.create_branch(base_name, from_ref="main")
    try:
        client.create_table(
            table=table_base,
            search_uri=s3_path,
            branch=base_name,
            namespace=NAMESPACE,
        )

        # Get the commit we just made — this is our fork point.
        base_commits = get_branch_commits(
            client, base_name, limit=1, lookback_hours=24 * 365
        )
        assert len(base_commits) > 0, "Expected at least one commit on base branch"
        fork_point_hash = base_commits[0]["hash"]

        # Step 2: create a child branch FROM the base branch (at the fork point)
        # and make a commit on it.
        client.create_branch(child_name, from_ref=f"{base_name}")
        try:
            client.create_table(
                table=table_child,
                search_uri=s3_path,
                branch=child_name,
                namespace=NAMESPACE,
            )

            # Step 3: ask find_branches_forking_from for all branches forking
            # from our fork_point_hash.  Pass ALL branches — the function must
            # return ONLY the child, not everything under the sun.
            all_branches = get_all_branches(client, username)
            forking = find_branches_forking_from(client, fork_point_hash, all_branches)
            forking_names = [b["name"] for b in forking]

            assert child_name in forking_names, (
                f"Child branch {child_name} should fork from {fork_point_hash[:10]}"
            )
            # The base branch itself must NOT appear — it contains the fork
            # point commit, it doesn't fork FROM it.
            assert base_name not in forking_names, (
                f"Base branch {base_name} should NOT appear as forking from its own commit"
            )
        finally:
            client.delete_table(
                table_child, branch=child_name, namespace=NAMESPACE, if_exists=True
            )
            client.delete_branch(child_name, if_exists=True)
    finally:
        client.delete_table(
            table_base, branch=base_name, namespace=NAMESPACE, if_exists=True
        )
        client.delete_branch(base_name, if_exists=True)


def test_find_branches_forking_from_no_match(client, test_branch, username):
    """A fake commit hash should match no branches."""
    fake_hash = "0" * 64
    all_branches = get_all_branches(client, username)
    forking = find_branches_forking_from(client, fake_hash, all_branches)
    assert forking == []


def test_known_main_commit_has_no_forks(client, username):
    """A known main commit (namespace creation) should have zero forks.

    This is a real-world regression test: the old logic falsely matched
    nearly every branch for ordinary main commits.
    """
    full_hash = "d0f0261c419279eee839bab03d06caa85c354259e252027e850e44e76421265f"
    parent_hash = "c3adaaec19b6cd1c292dd0c910350e7f1f7442ed53b148eadb573ff54c2f376d"

    # Verify the commit exists on main by searching for it as a child
    # of its known parent — no limit-based query that goes stale.
    hits = client.get_commits(
        ref="main",
        filter_by_parent_hash=parent_hash,
        filter_by_authored_date_start_at=datetime(2026, 2, 22, tzinfo=timezone.utc),
        filter_by_authored_date_end_at=datetime(2026, 2, 23, tzinfo=timezone.utc),
    )
    matching = [c for c in hits if c.ref.hash == full_hash]
    assert len(matching) == 1, (
        f"Expected commit {full_hash[:10]} on main, got {len(matching)} matches"
    )

    all_branches = get_all_branches(client, username)
    forking = find_branches_forking_from(client, full_hash, all_branches)
    assert forking == [], (
        f"Expected no branches forking from {full_hash[:10]}, got: "
        f"{[b['name'] for b in forking]}"
    )


def test_branch_commits_returns_none_for_nonexistent_branch(client):
    """get_branch_commits should return None for a branch that doesn't exist."""
    result = get_branch_commits(client, "nonexistent_branch_xyz_999")
    assert result is None


def test_expand_run_commits_after_pipeline(client, unique_id, username):
    """Run a pipeline on a throwaway branch and verify expand_run_commits
    replaces the merge commit with individual model materialization commits.

    The setup_ecommerce_users pipeline materializes one table, so we expect
    at least one 'Update ICEBERG_TABLE' commit instead of 'Run job_id=...'.
    """
    branch_name = f"{username}.{TEST_PREFIX}_expand_{unique_id}"
    client.create_branch(branch_name, from_ref="main")
    try:
        # Run the pipeline — this creates a transactional merge commit
        client.run(
            project_dir=SETUP_PROJECT_DIR,
            ref=branch_name,
        )

        # Fetch raw commits — should include a "Run job_id=..." merge commit
        raw_commits = get_branch_commits(
            client, branch_name, limit=5, lookback_hours=24 * 365
        )
        assert raw_commits is not None and len(raw_commits) > 0

        run_commits = [c for c in raw_commits if c["message"].startswith("Run job_id=")]
        assert len(run_commits) >= 1, (
            f"Expected at least one 'Run job_id=' commit, got messages: "
            f"{[c['message'] for c in raw_commits]}"
        )

        # Capture the job_id from our run's merge commit
        our_run = run_commits[0]
        our_job_id = our_run["message"].removeprefix("Run job_id=")

        # Expand — merge commits should be replaced with model commits
        expanded = expand_run_commits(client, branch_name, raw_commits)

        # Our specific merge commit should no longer appear
        our_run_remaining = [
            c for c in expanded if c["message"] == f"Run job_id={our_job_id}"
        ]
        assert our_run_remaining == [], (
            f"expand_run_commits should have replaced our run commit, "
            f"but 'Run job_id={our_job_id}' still present"
        )

        # Should have at least one model materialization commit
        model_commits = [
            c
            for c in expanded
            if "ICEBERG_TABLE" in c["message"]
        ]
        assert len(model_commits) >= 1, (
            f"Expected model commits with 'ICEBERG_TABLE', got: "
            f"{[c['message'] for c in expanded]}"
        )
    finally:
        client.delete_branch(branch_name, if_exists=True)
