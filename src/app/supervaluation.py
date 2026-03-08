"""
Supervaluationist meta-answer layer.

Aggregates per-branch query results into a single MetaAnswer that
summarises cross-branch agreement using supervaluationist semantics:
  - A statement is *definitely true* if every branch says true.
  - A statement is *definitely false* if every branch says false.
  - Otherwise it is *indeterminate* (mixed).

For numbers, the meta-answer reports the range and whether all branches
agree.  For sets, it computes the consensus (intersection), union, and
per-branch unique elements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc

from query_shape import QueryShape


@dataclass
class MetaAnswer:
    kind: str  # "number" | "boolean" | "set"
    summary: str  # short UI text
    per_branch: dict[str, Any]  # branch -> value
    details: dict[str, Any] = field(default_factory=dict)


def meta_answer_boolean(per_branch: dict[str, bool]) -> MetaAnswer:
    """Aggregate per-branch boolean values into a MetaAnswer."""
    true_count = sum(1 for v in per_branch.values() if v)
    false_count = len(per_branch) - true_count

    if false_count == 0:
        summary = "Definitely true"
        verdict = "definitely_true"
    elif true_count == 0:
        summary = "Definitely false"
        verdict = "definitely_false"
    else:
        summary = f"Mixed ({true_count} true, {false_count} false)"
        verdict = "mixed"

    return MetaAnswer(
        kind="boolean",
        summary=summary,
        per_branch=per_branch,
        details={
            "verdict": verdict,
            "true_count": true_count,
            "false_count": false_count,
        },
    )


def meta_answer_number(per_branch: dict[str, float]) -> MetaAnswer:
    """Aggregate per-branch numeric scalars into a MetaAnswer."""
    vals = list(per_branch.values())
    mn = min(vals)
    mx = max(vals)
    mean = sum(vals) / len(vals)
    agreement = mn == mx

    if agreement:
        summary = f"All agree: {mn}"
    else:
        summary = f"Range: [{mn}, {mx}]"

    return MetaAnswer(
        kind="number",
        summary=summary,
        per_branch=per_branch,
        details={
            "min": mn,
            "max": mx,
            "mean": mean,
            "spread": mx - mn,
            "agreement": agreement,
        },
    )


def meta_answer_set(per_branch: dict[str, set]) -> MetaAnswer:
    """Aggregate per-branch sets of IDs into a MetaAnswer."""
    all_sets = list(per_branch.values())
    union = set().union(*all_sets)
    intersection = all_sets[0].copy()
    for s in all_sets[1:]:
        intersection &= s

    unique_per_branch = {
        branch: vals - intersection for branch, vals in per_branch.items()
    }
    disagreement = union - intersection

    total = len(union)
    consensus = len(intersection)
    summary = f"Consensus: {consensus} of {total} total"

    return MetaAnswer(
        kind="set",
        summary=summary,
        per_branch=per_branch,
        details={
            "intersection": intersection,
            "union": union,
            "unique_per_branch": unique_per_branch,
            "disagreement": disagreement,
        },
    )


def compute_meta_answer(table: pa.Table, shape: QueryShape) -> MetaAnswer:
    """Dispatch to the appropriate meta-answer function.

    Splits the combined Arrow table by __branch_id, extracts per-branch
    values based on shape.result_type, and calls the matching aggregator.
    """
    branch_col = table.column("__branch_id")
    branches = branch_col.to_pylist()
    unique_branches = list(dict.fromkeys(branches))  # preserve order

    rt = shape.result_type.value  # compare by value to avoid dual-import enum mismatch

    if rt == "boolean":
        col_name = shape.value_column
        col = table.column(col_name)
        per_branch: dict[str, bool] = {}
        for b in unique_branches:
            mask = pc.equal(branch_col, b)
            filtered = pc.filter(col, mask)
            if len(filtered) == 0:
                raise ValueError(f"Branch '{b}' returned no rows for boolean query")
            per_branch[b] = filtered[0].as_py()
        return meta_answer_boolean(per_branch)

    if rt == "number":
        col_name = shape.value_column
        col = table.column(col_name)
        per_branch_num: dict[str, float] = {}
        for b in unique_branches:
            mask = pc.equal(branch_col, b)
            filtered = pc.filter(col, mask)
            if len(filtered) == 0:
                raise ValueError(f"Branch '{b}' returned no rows for number query")
            per_branch_num[b] = filtered[0].as_py()
        return meta_answer_number(per_branch_num)

    # SET
    col_name = shape.set_column
    col = table.column(col_name)
    per_branch_set: dict[str, set] = {}
    for b in unique_branches:
        mask = pc.equal(branch_col, b)
        filtered = pc.filter(col, mask)
        per_branch_set[b] = set(filtered.to_pylist())
    return meta_answer_set(per_branch_set)
