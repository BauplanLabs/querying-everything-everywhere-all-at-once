"""Pure helper functions and shared types.

No heavy dependencies — only stdlib, pydantic, and pyarrow/datafusion.
"""

import re
from typing import Literal

import datafusion
import pyarrow as pa
from pydantic import BaseModel

# Type aliases shared across modules
BranchMetadata = dict[str, dict[str, str]]  # {branch: {table_name: metadata_location}}

TABLE_RE = re.compile(r"(?:Update|Create|Delete) ICEBERG_TABLE (?:\w+\.)?(\w+)")


def commit_label(message: str) -> str:
    """Human-friendly label: table name for model commits, empty string otherwise."""
    m = TABLE_RE.match(message)
    return m.group(1) if m else ""


def short_branch_label(name: str) -> str:
    """Strip common prefixes and transactional suffixes from a branch name.

    Turns 'apo.multiverse_v_v_30m_py_gb-bpln-tx-run-...' into 'v_30m_py_gb'.
    """
    idx = name.find("multiverse_v_")
    if idx >= 0:
        name = name[idx + len("multiverse_v_") :]
    elif "." in name:
        name = name.split(".", 1)[1]
    tx_idx = name.find("-bpln-tx-run-")
    if tx_idx >= 0:
        name = name[:tx_idx]
    return name


class QueryRequest(BaseModel):
    question: str
    engine: Literal["adhoc", "native"] = "adhoc"
    use_cache: bool = True


def build_datafusion_context(
    schemas: dict[str, pa.Schema],
) -> datafusion.SessionContext:
    """Create a DataFusion context with empty tables matching the given schemas.

    Used by both SQL validation (text_to_sql) and query shape classification
    (query_shape) to plan SQL without executing it.
    """
    ctx = datafusion.SessionContext()
    for table_name, schema in schemas.items():
        empty_batch = pa.RecordBatch.from_pydict(
            {field.name: pa.array([], type=field.type) for field in schema},
            schema=schema,
        )
        ctx.register_record_batches(table_name, [[empty_batch]])
    return ctx
