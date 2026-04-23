"""Evaluator: score an LLM-proposed puzzle and produce feedback."""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import solver
from proposer import _target_clues as target_clue_count


@dataclass
class Score:
    grid: list[list[int]] | None
    valid: bool
    unique: bool
    solvable: bool
    difficulty: int
    num_clues: int
    target_clues: int = 25
    techniques: dict[str, int] = field(default_factory=dict)
    backtracks: int = 0
    target_distance: float = 9.0   # lower is better
    feedback: str = ""
    raw: str = ""

    @property
    def fitness(self) -> float:
        """Higher is better. Tie-break difficulty ties by clue-count proximity so
        partial progress (fewer clues at same diff) propagates through the loop."""
        if not self.valid:
            return -100
        if not self.unique:
            return -50
        if not self.solvable:
            return -30
        # Main signal: how far from target difficulty.
        # Tie-break: how far from ideal clue count (tiny weight).
        clue_term = abs(self.num_clues - self.target_clues) * 0.03
        return -(self.target_distance + clue_term)


GRID_RE = re.compile(r"\[\s*(?:\[[^\]]*\]\s*,?\s*){9}\]", re.DOTALL)


def extract_grid(text: str) -> Optional[list[list[int]]]:
    """Pull a 9x9 int grid out of free-form LLM text.

    Supports:
      - JSON arrays like [[5,3,0,...],[...], ...]
      - 9 lines of 9 digits (spaces/dots for zero)
    """
    t = text.strip()
    # strip common code fences
    t = re.sub(r"^```(?:json|python)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)

    m = GRID_RE.search(t)
    if m:
        blob = m.group(0)
        try:
            grid = json.loads(blob)
            if isinstance(grid, list):
                return grid
        except json.JSONDecodeError:
            pass

    # Fallback: 9 lines of 9 tokens
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    digit_lines = []
    for ln in lines:
        tokens = re.findall(r"[0-9.\-_]", ln)
        if len(tokens) == 9:
            digit_lines.append([0 if t in ".-_" else int(t) for t in tokens])
        if len(digit_lines) == 9:
            return digit_lines
    return None


def evaluate(raw_text: str, target_difficulty: int) -> Score:
    tgt_clues = target_clue_count(target_difficulty)
    grid = extract_grid(raw_text)
    if grid is None:
        return Score(
            grid=None, valid=False, unique=False, solvable=False,
            difficulty=0, num_clues=0, target_clues=tgt_clues,
            feedback="Could not extract a 9x9 grid. Respond with a JSON array of 9 rows, each 9 ints (0=blank).",
            raw=raw_text,
        )

    ok, msg = solver.validate_structure(grid)
    if not ok:
        return Score(
            grid=grid, valid=False, unique=False, solvable=False,
            difficulty=0, num_clues=_count_clues(grid),
            feedback=f"Grid is structurally invalid: {msg}",
            raw=raw_text,
        )

    clues = _count_clues(grid)

    try:
        sols = solver.count_solutions(grid, limit=2)
    except solver.SolverBudgetExceeded:
        return Score(
            grid=grid, valid=True, unique=False, solvable=False,
            difficulty=0, num_clues=clues, target_clues=tgt_clues,
            feedback=(
                f"Uniqueness check exceeded budget ({clues} clues - likely far too sparse, "
                "search space too large). Use more clues (24-30)."
            ),
            raw=raw_text,
        )
    if sols == 0:
        return Score(
            grid=grid, valid=True, unique=False, solvable=False,
            difficulty=0, num_clues=clues, target_clues=tgt_clues,
            feedback=f"No solution exists. Placed clues contradict each other (has {clues} clues).",
            raw=raw_text,
        )
    if sols > 1:
        return Score(
            grid=grid, valid=True, unique=False, solvable=True,
            difficulty=0, num_clues=clues, target_clues=tgt_clues,
            feedback=f"Multiple solutions ({clues} clues; need more/different clues to force uniqueness).",
            raw=raw_text,
        )

    try:
        result = solver.solve_with_techniques(grid)
    except solver.SolverBudgetExceeded:
        return Score(
            grid=grid, valid=True, unique=True, solvable=True,
            difficulty=10, num_clues=clues, target_clues=tgt_clues,
            techniques={}, backtracks=999,
            target_distance=abs(10 - target_difficulty),
            feedback=(
                f"Puzzle is extremely hard (solver budget exceeded while solving; {clues} clues). "
                "Treated as difficulty 10. Add more clues to bring difficulty down."
            ),
            raw=raw_text,
        )
    difficulty = solver.rate_difficulty(result["techniques"], result["backtracks"])
    distance = abs(difficulty - target_difficulty)

    return Score(
        grid=grid,
        valid=True, unique=True, solvable=True,
        difficulty=difficulty,
        num_clues=clues, target_clues=tgt_clues,
        techniques=dict(result["techniques"]),
        backtracks=result["backtracks"],
        target_distance=distance,
        feedback=_feedback(difficulty, target_difficulty, result["techniques"], result["backtracks"], clues),
        raw=raw_text,
    )


def _count_clues(grid: list[list[int]]) -> int:
    return sum(1 for row in grid for v in row if v != 0)


def _feedback(diff: int, target: int, techniques: Counter, backtracks: int, clues: int) -> str:
    tech_summary = ", ".join(f"{k}={v}" for k, v in sorted(techniques.items())) or "none"
    base = (
        f"clues={clues}, difficulty={diff}, target={target}. "
        f"Solver used: {tech_summary}; backtracks={backtracks}. "
    )
    if diff == target:
        return base + "On target."
    if diff < target:
        if diff <= 3:
            return base + (
                "Too easy - solver only needs naked singles. "
                "Remove more clues (aim for ~24-28) and place them so at least some cells "
                "require hidden singles or pair/locked-candidates deductions."
            )
        if diff <= 5:
            return base + (
                "Needs hidden singles but not pair/locked-candidates logic. "
                "Harder puzzles at difficulty 7 require naked pairs or locked candidates "
                "(candidates of a digit within a box all in one row/col). "
                "Try fewer clues (around 24-26) and place them so solving requires these deductions."
            )
        return base + (
            "Close. Needs a few more pair/locked-candidates deductions to reach target. "
            "Try removing one or two more clues carefully."
        )
    # diff > target
    return base + (
        "Too hard - the solver had to guess/backtrack or apply advanced techniques many times. "
        "Add a few more clues to reduce required deductions."
    )
