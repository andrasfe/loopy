"""Sudoku evaluator: run code, extract grid, validate, rate difficulty."""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

# Add project root so we can import solver
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import executor
import solver
from score import Score

TARGET_CLUE_COUNT = {
    1: 52, 2: 46, 3: 40, 4: 34, 5: 30, 6: 27, 7: 25, 8: 23, 9: 22, 10: 21,
}

GRID_RE = re.compile(r"\[\s*(?:\[[^\]]*\]\s*,?\s*){9}\]", re.DOTALL)


def _target_clues(target: int) -> int:
    return TARGET_CLUE_COUNT.get(target, 25)


def _extract_grid(text: str) -> Optional[list[list[int]]]:
    t = text.strip()
    t = re.sub(r"^```(?:json|python)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    m = GRID_RE.search(t)
    if m:
        try:
            grid = json.loads(m.group(0))
            if isinstance(grid, list):
                return grid
        except json.JSONDecodeError:
            pass
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    digit_lines = []
    for ln in lines:
        tokens = re.findall(r"[0-9.\-_]", ln)
        if len(tokens) == 9:
            digit_lines.append([0 if tok in ".-_" else int(tok) for tok in tokens])
        if len(digit_lines) == 9:
            return digit_lines
    return None


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
            return base + "Too easy. Remove more clues (aim for ~24-28)."
        if diff <= 5:
            return base + "Needs harder techniques. Try fewer clues (around 24-26)."
        return base + "Close. Try removing one or two more clues carefully."
    return base + "Too hard. Add a few more clues."


def evaluate_code(code: str, target: int, exec_timeout: float = 10.0) -> Score:
    """Execute code, extract Sudoku grid, validate, rate difficulty."""
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
            feedback="Code produced no output. Print a 9x9 JSON array to stdout.",
        )

    grid = _extract_grid(result.stdout)
    if grid is None:
        return Score(
            code=code, valid=False, difficulty=0, raw=result.stdout,
            feedback="Output not parseable as a 9x9 grid. Use: print(json.dumps(grid))",
        )

    ok, msg = solver.validate_structure(grid)
    if not ok:
        return Score(
            code=code, valid=False, difficulty=0, raw=result.stdout,
            feedback=f"Invalid grid: {msg}",
            details={"grid": grid, "clues": _count_clues(grid)},
        )

    clues = _count_clues(grid)

    try:
        sols = solver.count_solutions(grid, limit=2)
    except solver.SolverBudgetExceeded:
        return Score(
            code=code, valid=True, difficulty=0, raw=result.stdout,
            target_distance=9.0,
            feedback=f"Uniqueness check exceeded budget ({clues} clues). Use more clues.",
            details={"grid": grid, "clues": clues},
        )
    if sols != 1:
        return Score(
            code=code, valid=True, difficulty=0, raw=result.stdout,
            target_distance=8.0,
            feedback=f"{'No solution' if sols == 0 else 'Multiple solutions'} ({clues} clues).",
            details={"grid": grid, "clues": clues, "solutions": sols},
        )

    try:
        res = solver.solve_with_techniques(grid)
    except solver.SolverBudgetExceeded:
        difficulty = 10
        distance = abs(10 - target)
        return Score(
            code=code, valid=True, difficulty=10, raw=result.stdout,
            target_distance=distance,
            feedback=f"Extremely hard (budget exceeded, {clues} clues). Treated as difficulty 10.",
            details={"grid": grid, "clues": clues},
        )

    difficulty = solver.rate_difficulty(res["techniques"], res["backtracks"])
    distance = abs(difficulty - target)

    return Score(
        code=code, valid=True, difficulty=difficulty, raw=result.stdout,
        target_distance=distance,
        feedback=_feedback(difficulty, target, res["techniques"], res["backtracks"], clues),
        details={
            "grid": grid, "clues": clues,
            "techniques": dict(res["techniques"]),
            "backtracks": res["backtracks"],
        },
    )
