"""Tests for server helpers and request validation."""

import asyncio
from unittest.mock import MagicMock, patch, AsyncMock

import pytest
from pydantic import ValidationError

from app.helpers import commit_label, short_branch_label, QueryRequest


class TestCommitLabel:
    def test_update_table(self):
        assert (
            commit_label("Update ICEBERG_TABLE ns.user_predictions")
            == "user_predictions"
        )

    def test_create_table(self):
        assert commit_label("Create ICEBERG_TABLE bronze_events") == "bronze_events"

    def test_delete_table(self):
        assert commit_label("Delete ICEBERG_TABLE ns.old_table") == "old_table"

    def test_non_table_message(self):
        assert commit_label("Run job_id=abc123") == ""

    def test_empty_message(self):
        assert commit_label("") == ""


class TestShortBranchLabel:
    def test_strips_multiverse_prefix(self):
        assert short_branch_label("apo.multiverse_v_v_30m_py_gb") == "v_30m_py_gb"

    def test_strips_tx_suffix(self):
        name = "apo.multiverse_v_v_60m_sql_gb-bpln-tx-run-20260306205340-e7c1f30b"
        assert short_branch_label(name) == "v_60m_sql_gb"

    def test_strips_username_only(self):
        assert short_branch_label("apo.some_branch") == "some_branch"

    def test_no_prefix(self):
        assert short_branch_label("bare_branch") == "bare_branch"

    def test_tx_suffix_without_multiverse(self):
        name = "apo.my_branch-bpln-tx-run-123-abc"
        assert short_branch_label(name) == "my_branch"


class TestQueryRequestValidation:
    def test_valid_adhoc(self):
        req = QueryRequest(question="How many users?", engine="adhoc")
        assert req.engine == "adhoc"

    def test_valid_native(self):
        req = QueryRequest(question="How many users?", engine="native")
        assert req.engine == "native"

    def test_default_engine(self):
        req = QueryRequest(question="How many users?")
        assert req.engine == "adhoc"

    def test_invalid_engine_rejected(self):
        with pytest.raises(ValidationError):
            QueryRequest(question="How many users?", engine="bogus")


class TestCommitOrdering:
    def test_reverse_produces_oldest_first(self):
        commits_newest_first = [
            {"hash": "ccc", "message": "third"},
            {"hash": "bbb", "message": "second"},
            {"hash": "aaa", "message": "first"},
        ]
        ordered = list(reversed(commits_newest_first))
        assert ordered[0]["hash"] == "aaa"
        assert ordered[-1]["hash"] == "ccc"

    def test_fork_hash_filtering(self):
        """Commits before the fork hash should be kept, fork hash excluded."""
        fork_hash = "fff"
        raw = [
            {"hash": "ccc"},
            {"hash": "bbb"},
            {"hash": "fff"},
            {"hash": "aaa"},
        ]
        unique = []
        found_fork = False
        for c in raw:
            if c["hash"] == fork_hash:
                found_fork = True
                break
            unique.append(c)
        assert found_fork
        assert len(unique) == 2
        assert unique[0]["hash"] == "ccc"
        assert unique[1]["hash"] == "bbb"


class _FakeBranch:
    """Minimal branch object with .name and .hash attributes."""

    def __init__(self, name, hash_val="abc123"):
        self.name = name
        self.hash = hash_val


class _StubFinder:
    """Meta path finder that stubs specific missing packages (pyiceberg,
    datafusion, multiverse_provider) so importing app.server succeeds."""

    STUB_PREFIXES = ("pyiceberg", "datafusion", "multiverse_provider", "litellm")

    def __init__(self):
        self.stubs: dict[str, object] = {}

    def find_module(self, fullname, path=None):
        if any(
            fullname == p or fullname.startswith(p + ".") for p in self.STUB_PREFIXES
        ):
            return self
        return None

    def load_module(self, fullname):
        import sys

        if fullname in sys.modules:
            return sys.modules[fullname]
        stub = MagicMock()
        stub.__path__ = []
        sys.modules[fullname] = stub
        self.stubs[fullname] = stub
        return stub


class TestWaitForBranch:
    """Tests for the async startup polling loop."""

    @pytest.fixture(autouse=True, scope="class")
    def _stub_missing_deps(self):
        """Install targeted stubs for native-extension packages that are not
        in the test environment (pyiceberg, datafusion, multiverse_provider)
        so ``from app.server import wait_for_branch`` succeeds."""
        import sys

        finder = _StubFinder()
        sys.meta_path.insert(0, finder)

        # Evict cached app modules so the next import picks up stubs.
        _app_modules = [
            "lakehouse",
            "multiverse",
            "query_shape",
            "supervaluation",
            "text_to_sql",
            "naive_multiverse",
            "native_multiverse",
            "helpers",
            "multiverse_provider",
        ]
        evicted = {}
        for m in list(sys.modules):
            if (
                m.startswith("app.")
                or m in _app_modules
                or m.startswith("multiverse_provider")
            ):
                evicted[m] = sys.modules.pop(m)

        yield

        sys.meta_path.remove(finder)
        for m in list(finder.stubs):
            sys.modules.pop(m, None)
        # Evict app modules that were imported with stubs active.
        for m in list(sys.modules):
            if m.startswith("app.") or m in _app_modules:
                sys.modules.pop(m, None)
        # Restore any previously cached app modules.
        sys.modules.update(evicted)

    def _make_mock_client(self, branches_sequence):
        """Create a mock client that returns different branches on each call.

        branches_sequence is a list of lists-of-branch-names. Each call to
        client.get_branches pops the next entry.
        """
        call_idx = {"i": 0}
        client = MagicMock()

        def fake_get_branches(user, **kwargs):
            idx = min(call_idx["i"], len(branches_sequence) - 1)
            call_idx["i"] += 1
            return [_FakeBranch(n) for n in branches_sequence[idx]]

        client.get_branches = MagicMock(side_effect=fake_get_branches)
        return client

    def test_found_immediately(self):
        from app.server import wait_for_branch

        client = self._make_mock_client([["apo.multiverse_main", "other"]])
        result = asyncio.run(
            wait_for_branch(
                client, "apo", "apo.multiverse_main", max_wait=5, poll_interval=0.1
            )
        )
        assert result is True

    def test_found_after_delay(self):
        from app.server import wait_for_branch

        # First two polls: branch missing. Third poll: branch appears.
        client = self._make_mock_client(
            [
                ["other"],
                ["other"],
                ["apo.multiverse_main", "other"],
            ]
        )
        result = asyncio.run(
            wait_for_branch(
                client, "apo", "apo.multiverse_main", max_wait=5, poll_interval=0.05
            )
        )
        assert result is True

    def test_timeout(self):
        from app.server import wait_for_branch

        # Branch never appears
        client = self._make_mock_client([["other"]])
        result = asyncio.run(
            wait_for_branch(
                client, "apo", "apo.multiverse_main", max_wait=0.2, poll_interval=0.05
            )
        )
        assert result is False

    def test_found_near_deadline(self):
        """Branch appears on the final check after deadline has passed.

        Mocks time.monotonic so the sequence is:
          1. monotonic() = 0   -> deadline set to 10
          2. monotonic() = 0   -> check: not found
          3. monotonic() = 5   -> remaining=5, sleep(5)
          4. monotonic() = 10  -> check: not found
          5. monotonic() = 11  -> remaining=-1, would exit...
             but the loop checks branch FIRST, so:
          6. monotonic() = 11  -> check: FOUND -> return True

        The old buggy loop (check deadline before re-checking branch)
        would return False here.
        """
        from app.server import wait_for_branch

        # Branch missing twice, then found on third check
        client = self._make_mock_client(
            [
                ["other"],
                ["other"],
                ["apo.multiverse_main"],
            ]
        )

        # monotonic returns: 0 (deadline calc), 0 (first remaining),
        # 5 (second remaining), 10 (check after sleep), 11 (remaining after second miss)
        # but third check finds the branch before remaining is evaluated
        clock_values = iter([0, 0, 5, 10, 11])

        import app.server as server_mod

        mock_monotonic = MagicMock(side_effect=clock_values)
        mock_time = MagicMock()
        mock_time.monotonic = mock_monotonic

        with (
            patch.object(server_mod, "time", mock_time),
            patch.object(server_mod.asyncio, "sleep", new_callable=AsyncMock),
        ):
            result = asyncio.run(
                wait_for_branch(
                    client,
                    "apo",
                    "apo.multiverse_main",
                    max_wait=10,
                    poll_interval=5,
                )
            )
        assert result is True
        # Verify all three checks happened (branch found on third)
        assert client.get_branches.call_count == 3

    def test_timeout_deterministic(self):
        """Branch never appears and deadline expires. Mocked time."""
        from app.server import wait_for_branch
        import app.server as server_mod

        client = self._make_mock_client([["other"]])
        # monotonic: 0 (deadline calc), 0 (first remaining), 11 (second remaining - past deadline)
        clock_values = iter([0, 0, 11])

        mock_monotonic = MagicMock(side_effect=clock_values)
        mock_time = MagicMock()
        mock_time.monotonic = mock_monotonic

        with (
            patch.object(server_mod, "time", mock_time),
            patch.object(server_mod.asyncio, "sleep", new_callable=AsyncMock),
        ):
            result = asyncio.run(
                wait_for_branch(
                    client,
                    "apo",
                    "apo.multiverse_main",
                    max_wait=10,
                    poll_interval=5,
                )
            )
        assert result is False
