"""
Multiverse query engine — abstract base class.

Defines the MultiverseEngine interface for running the same SQL query
across multiple bauplan branches and combining the results.
"""

from __future__ import annotations

import logging
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow as pa
from dotenv import load_dotenv

from pyiceberg.table import StaticTable

from helpers import BranchMetadata as BranchMetadata  # re-exported
from query_shape import QueryShape
from supervaluation import MetaAnswer, compute_meta_answer

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

logger = logging.getLogger(__name__)


def get_parquet_files(metadata_location: str) -> list[str]:
    """Extract Parquet file paths from an Iceberg metadata location."""
    static = StaticTable.from_metadata(metadata_location)
    return [task.file.file_path for task in static.scan().plan_files()]


NAMESPACE: str = os.environ.get("BAUPLAN_NAMESPACE", "apo_multiverse")

# Type aliases for clarity
ErrorsDict = dict[str, str]
QueryResult = tuple[pa.Table | None, ErrorsDict]
MultiverseResult = tuple[pa.Table | None, ErrorsDict, MetaAnswer | None]


@dataclass
class EngineStats:
    """Execution metrics collected by an engine run.

    Populated after ``query_multiverse`` completes.
    ``operator_bytes`` is available only when ``report_bytes=True``
    (requires a second EXPLAIN ANALYZE pass).
    """

    result_bytes: int = 0
    branches_queried: int = 0
    operator_bytes: int = 0
    per_branch_operator_bytes: dict[str, int] = field(default_factory=dict)
    # Sub-phase timing (milliseconds) within the engine
    plan_ms: int = 0
    exec_ms: int = 0
    concat_ms: int = 0


# Regex for extracting output_bytes from EXPLAIN ANALYZE plan text.
_OUTPUT_BYTES_RE = re.compile(r"output_bytes=([\d.]+)\s*([A-Za-z]*)")
_UNIT_MULTIPLIER = {
    "": 1,
    "b": 1,
    "kb": 1024,
    "kib": 1024,
    "mb": 1024**2,
    "mib": 1024**2,
    "gb": 1024**3,
    "gib": 1024**3,
    "tb": 1024**4,
    "tib": 1024**4,
}


def parse_operator_bytes(plan_text: str) -> int:
    """Sum all output_bytes values from an EXPLAIN ANALYZE plan string."""
    total = 0
    for m in _OUTPUT_BYTES_RE.finditer(plan_text):
        value = float(m.group(1))
        unit = m.group(2).strip().lower()
        total += int(value * _UNIT_MULTIPLIER.get(unit, 1))
    return total


class MultiverseEngine(ABC):
    """Abstract base for multiverse query engines.

    The only public method is ``query_multiverse``, which returns a combined
    result table, per-branch errors, and an optional MetaAnswer.

    Subclasses must implement ``_query`` (execute SQL on all branches and
    return a combined pyarrow Table).

    When ``report_bytes=True`` is passed to ``query_multiverse``, the engine
    re-executes the query with EXPLAIN ANALYZE via ``_explain_analyze`` to
    collect operator-level byte metrics.  This doubles execution time and
    should only be used for benchmarking, not in the demo UI.
    """

    def __init__(self) -> None:
        self.stats = EngineStats()

    def query_multiverse(
        self,
        sql: str,
        branches: list[str],
        shape: QueryShape | None = None,
        report_bytes: bool = False,
    ) -> MultiverseResult:
        """Execute SQL on all branches and return the final result.

        Args:
            sql: The SQL query to execute on each branch.
            branches: Branch names to query.
            shape: Optional QueryShape for supervaluation.
            report_bytes: If True, run a second EXPLAIN ANALYZE pass to
                collect operator-level output_bytes metrics.

        Returns:
            (combined_table, errors_dict, meta_answer).
        """
        combined, errors = self._query(sql, branches)

        # Basic stats (always collected, free)
        self.stats = EngineStats(
            result_bytes=combined.nbytes if combined is not None else 0,
            branches_queried=len(branches),
        )

        # Expensive: re-execute with EXPLAIN ANALYZE for operator bytes
        if report_bytes:
            op_bytes, per_branch = self._explain_analyze(sql, branches)
            self.stats.operator_bytes = op_bytes
            self.stats.per_branch_operator_bytes = per_branch
            logger.info(
                "Engine stats: %d branches, result_bytes=%d, operator_bytes=%d",
                len(branches),
                self.stats.result_bytes,
                op_bytes,
            )

        meta = self._summarize(combined, shape)
        return combined, errors, meta

    @abstractmethod
    def _query(self, sql: str, branches: list[str]) -> QueryResult:
        """Execute SQL on all branches and return combined results.

        Returns:
            (combined_table, errors_dict). combined_table is a pyarrow Table
            with a __branch_id column, or None if all branches failed.
        """

    def _explain_analyze(
        self, sql: str, branches: list[str]
    ) -> tuple[int, dict[str, int]]:
        """Re-execute with EXPLAIN ANALYZE to collect operator bytes.

        Default implementation returns zeros.  Subclasses should override
        to provide engine-specific EXPLAIN ANALYZE logic.
        """
        return 0, {}

    def _summarize(
        self, table: pa.Table | None, shape: QueryShape | None
    ) -> MetaAnswer | None:
        """Compute a supervaluationist meta-answer from the combined table.

        Returns a MetaAnswer or None if the table or shape is unavailable.
        """
        if table is None or shape is None:
            return None
        return compute_meta_answer(table, shape)
