"""Max-Cut evaluator: run code, parse partition, verify and score cut value."""
from __future__ import annotations

import json
from typing import Optional

import executor
from score import Score

# Fixed graph: 20 nodes, 50 weighted edges.
# Optimal cut is ~135 (found by exhaustive search on this small instance).
NUM_NODES = 20
OPTIMAL_CUT = 247

EDGES = [
    (0, 1, 7), (0, 4, 3), (0, 7, 5), (0, 12, 8), (1, 2, 6),
    (1, 5, 4), (1, 9, 9), (2, 3, 5), (2, 6, 7), (2, 10, 3),
    (3, 7, 8), (3, 11, 6), (3, 14, 4), (4, 5, 9), (4, 8, 5),
    (4, 13, 7), (5, 6, 3), (5, 9, 8), (6, 7, 6), (6, 10, 5),
    (7, 11, 4), (7, 15, 9), (8, 9, 7), (8, 12, 3), (8, 16, 6),
    (9, 10, 5), (9, 13, 8), (10, 11, 9), (10, 14, 4), (10, 17, 7),
    (11, 15, 3), (11, 18, 5), (12, 13, 6), (12, 16, 8), (13, 14, 9),
    (13, 17, 4), (14, 15, 7), (14, 18, 3), (15, 19, 6), (16, 17, 5),
    (16, 19, 9), (17, 18, 8), (17, 19, 4), (18, 19, 7), (0, 19, 5),
    (1, 18, 6), (2, 17, 4), (3, 16, 8), (5, 15, 3), (6, 14, 9),
]

TOTAL_WEIGHT = 299  # sum of all edge weights


def _compute_cut(partition: list[int]) -> int:
    """Compute cut value: sum of edge weights crossing the partition."""
    return sum(w for u, v, w in EDGES if partition[u] != partition[v])


def _format_graph() -> str:
    """Format graph for display."""
    lines = [f"Nodes: {NUM_NODES}, Edges: {len(EDGES)}, Total weight: {TOTAL_WEIGHT}"]
    return "\n".join(lines)


def evaluate_code(code: str, target: int, exec_timeout: float = 10.0) -> Score:
    """Execute code, parse partition, compute and verify cut value.

    Target is interpreted as: achieve cut_value >= (target/10) * OPTIMAL_CUT.
    E.g., target=7 means achieve >= 70% of optimal (~94).
    """
    result = executor.run_code(code, timeout=exec_timeout)

    if result.timed_out:
        return Score(
            code=code, valid=False, difficulty=0,
            exec_error=f"Timed out after {exec_timeout}s",
            feedback=f"Code timed out (>{exec_timeout}s). Simplify your algorithm.",
        )
    if result.returncode != 0:
        stderr = result.stderr[:800] if result.stderr else "(no stderr)"
        return Score(
            code=code, valid=False, difficulty=0,
            exec_error=f"Exit code {result.returncode}",
            feedback=f"Code crashed.\nStderr:\n{stderr}\nFix the error.",
        )
    if not result.stdout.strip():
        return Score(
            code=code, valid=False, difficulty=0,
            exec_error="No output",
            feedback="Code produced no output. Print a JSON object with 'partition' (list of 0/1) to stdout.",
        )

    # Parse output
    try:
        data = json.loads(result.stdout.strip())
    except json.JSONDecodeError:
        return Score(
            code=code, valid=False, difficulty=0, raw=result.stdout,
            feedback=(
                "Output is not valid JSON. Print: "
                'json.dumps({"partition": [0,1,...], "cut_value": N})'
            ),
        )

    if not isinstance(data, dict) or "partition" not in data:
        return Score(
            code=code, valid=False, difficulty=0, raw=result.stdout,
            feedback=(
                'Output must be a JSON object with "partition" key. '
                'Example: {"partition": [0,1,0,1,...], "cut_value": 42}'
            ),
        )

    partition = data["partition"]

    # Validate partition
    if not isinstance(partition, list) or len(partition) != NUM_NODES:
        return Score(
            code=code, valid=False, difficulty=0, raw=result.stdout,
            feedback=f"Partition must be a list of {NUM_NODES} elements, got {len(partition) if isinstance(partition, list) else type(partition).__name__}.",
        )
    if not all(v in (0, 1) for v in partition):
        return Score(
            code=code, valid=False, difficulty=0, raw=result.stdout,
            feedback="Partition must contain only 0s and 1s.",
        )
    if len(set(partition)) < 2:
        return Score(
            code=code, valid=False, difficulty=0, raw=result.stdout,
            feedback="Partition must have at least one node in each set (not all 0s or all 1s).",
        )

    # Compute cut value
    cut_value = _compute_cut(partition)
    claimed = data.get("cut_value", None)

    # Score: map cut_value to a 1-10 difficulty scale
    # difficulty = round(10 * cut_value / OPTIMAL_CUT), capped at 10
    achieved_level = min(10, round(10 * cut_value / OPTIMAL_CUT))
    target_threshold = round(OPTIMAL_CUT * target / 10)
    distance = max(0, target - achieved_level)

    feedback_parts = [
        f"Cut value: {cut_value}/{OPTIMAL_CUT} optimal ({100*cut_value/OPTIMAL_CUT:.0f}%).",
        f"Achieved level: {achieved_level}/10, target: {target}/10.",
    ]
    if claimed is not None and claimed != cut_value:
        feedback_parts.append(f"Note: you claimed cut_value={claimed} but actual is {cut_value}.")
    if distance == 0:
        feedback_parts.append("On target!")
    elif cut_value < target_threshold:
        feedback_parts.append(
            f"Need cut_value >= {target_threshold}. "
            "Try local search: flip each node, keep if it improves the cut. "
            "Or use simulated annealing / greedy approaches."
        )
    else:
        feedback_parts.append("Above target threshold.")

    set_sizes = (partition.count(0), partition.count(1))

    return Score(
        code=code, valid=True,
        difficulty=achieved_level,
        target_distance=distance,
        feedback=" ".join(feedback_parts),
        raw=result.stdout,
        details={
            "partition": partition,
            "cut_value": cut_value,
            "optimal": OPTIMAL_CUT,
            "pct_optimal": round(100 * cut_value / OPTIMAL_CUT, 1),
            "set_sizes": set_sizes,
        },
    )
