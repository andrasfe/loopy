"""Sudoku solver with technique-tiered difficulty rating.

Implements:
  - Structural validation
  - Solution-counting (for uniqueness, capped at 2)
  - Technique-based solver: naked single, hidden single, locked candidates, naked pair
  - Backtracking fallback with guess counting
  - 1-10 difficulty rating derived from which techniques were needed
"""
from __future__ import annotations

import random
from collections import Counter
from copy import deepcopy

Grid = list[list[int]]


class SolverBudgetExceeded(Exception):
    """Raised when the solver exceeds its node budget."""


DEFAULT_NODE_BUDGET = 200_000

ROWS = [[(r, c) for c in range(9)] for r in range(9)]
COLS = [[(r, c) for r in range(9)] for c in range(9)]
BOXES = [
    [(3 * (b // 3) + i, 3 * (b % 3) + j) for i in range(3) for j in range(3)]
    for b in range(9)
]
UNITS = ROWS + COLS + BOXES


def _peers(r: int, c: int) -> set[tuple[int, int]]:
    p: set[tuple[int, int]] = set()
    for cc in range(9):
        if cc != c:
            p.add((r, cc))
    for rr in range(9):
        if rr != r:
            p.add((rr, c))
    br, bc = 3 * (r // 3), 3 * (c // 3)
    for i in range(3):
        for j in range(3):
            rr, cc = br + i, bc + j
            if (rr, cc) != (r, c):
                p.add((rr, cc))
    return p


PEERS = {(r, c): _peers(r, c) for r in range(9) for c in range(9)}


def validate_structure(grid: object) -> tuple[bool, str]:
    """Check shape, types, and absence of same-unit conflicts."""
    if not isinstance(grid, list) or len(grid) != 9:
        return False, "grid must be a list of 9 rows"
    for r, row in enumerate(grid):
        if not isinstance(row, list) or len(row) != 9:
            return False, f"row {r} is not a list of 9"
        for c, v in enumerate(row):
            if not isinstance(v, int) or v < 0 or v > 9:
                return False, f"cell ({r},{c}) has invalid value {v!r} (need int 0..9)"
    for unit in UNITS:
        seen: dict[int, tuple[int, int]] = {}
        for (r, c) in unit:
            v = grid[r][c]
            if v == 0:
                continue
            if v in seen:
                return False, f"duplicate {v} at ({r},{c}) and {seen[v]}"
            seen[v] = (r, c)
    return True, "ok"


def count_solutions(grid: Grid, limit: int = 2, node_budget: int = DEFAULT_NODE_BUDGET) -> int:
    """Return number of solutions, capped at `limit`. Raises SolverBudgetExceeded."""
    g = deepcopy(grid)
    count = 0
    nodes = 0

    def safe(r: int, c: int, v: int) -> bool:
        for pr, pc in PEERS[(r, c)]:
            if g[pr][pc] == v:
                return False
        return True

    def pick_cell():
        best = None
        best_opts = 10
        for r in range(9):
            for c in range(9):
                if g[r][c] == 0:
                    opts = sum(1 for v in range(1, 10) if safe(r, c, v))
                    if opts < best_opts:
                        best_opts = opts
                        best = (r, c)
                        if opts <= 1:
                            return best
        return best

    def solve():
        nonlocal count, nodes
        if count >= limit:
            return
        nodes += 1
        if nodes > node_budget:
            raise SolverBudgetExceeded()
        cell = pick_cell()
        if cell is None:
            count += 1
            return
        r, c = cell
        for v in range(1, 10):
            if safe(r, c, v):
                g[r][c] = v
                solve()
                if count >= limit:
                    return
                g[r][c] = 0

    solve()
    return count


# ---------- technique-tiered solver ----------

def _compute_candidates(grid: Grid) -> list[list[set[int]]]:
    cands = [[set() for _ in range(9)] for _ in range(9)]
    for r in range(9):
        for c in range(9):
            if grid[r][c] == 0:
                used = {grid[pr][pc] for pr, pc in PEERS[(r, c)] if grid[pr][pc]}
                cands[r][c] = set(range(1, 10)) - used
    return cands


def _place(grid, cands, r, c, v):
    grid[r][c] = v
    cands[r][c] = set()
    for pr, pc in PEERS[(r, c)]:
        cands[pr][pc].discard(v)


def _naked_single(grid, cands) -> bool:
    for r in range(9):
        for c in range(9):
            if grid[r][c] == 0 and len(cands[r][c]) == 1:
                _place(grid, cands, r, c, next(iter(cands[r][c])))
                return True
    return False


def _hidden_single(grid, cands) -> bool:
    for unit in UNITS:
        for v in range(1, 10):
            positions = [(r, c) for (r, c) in unit if v in cands[r][c]]
            if len(positions) == 1:
                r, c = positions[0]
                _place(grid, cands, r, c, v)
                return True
    return False


def _locked_candidates(grid, cands) -> bool:
    """A value's candidates within a box lie in a single row or column:
    eliminate that value from the rest of that row/column outside the box."""
    for box in BOXES:
        box_set = set(box)
        for v in range(1, 10):
            positions = [(r, c) for (r, c) in box if v in cands[r][c]]
            if not positions:
                continue
            rows = {r for (r, _) in positions}
            cols = {c for (_, c) in positions}
            if len(rows) == 1:
                r = next(iter(rows))
                changed = False
                for c in range(9):
                    if (r, c) not in box_set and v in cands[r][c]:
                        cands[r][c].discard(v)
                        changed = True
                if changed:
                    return True
            if len(cols) == 1:
                c = next(iter(cols))
                changed = False
                for r in range(9):
                    if (r, c) not in box_set and v in cands[r][c]:
                        cands[r][c].discard(v)
                        changed = True
                if changed:
                    return True
    return False


def _naked_pair(grid, cands) -> bool:
    """Two cells in a unit share the same 2-candidate set:
    remove those two values from the rest of the unit."""
    for unit in UNITS:
        pairs: dict[frozenset[int], list[tuple[int, int]]] = {}
        for (r, c) in unit:
            if len(cands[r][c]) == 2:
                key = frozenset(cands[r][c])
                pairs.setdefault(key, []).append((r, c))
        for pair_vals, cells in pairs.items():
            if len(cells) == 2:
                changed = False
                cell_set = set(cells)
                for (r, c) in unit:
                    if (r, c) not in cell_set:
                        before = len(cands[r][c])
                        cands[r][c] -= pair_vals
                        if len(cands[r][c]) < before:
                            changed = True
                if changed:
                    return True
    return False


def _backtrack(grid, cands, budget: list[int]) -> tuple[bool, int]:
    """Brute-force from current state. Returns (solved, guess_count). Uses shared budget list."""
    best = None
    best_n = 10
    for r in range(9):
        for c in range(9):
            if grid[r][c] == 0:
                n = len(cands[r][c])
                if n == 0:
                    return False, 0
                if n < best_n:
                    best_n = n
                    best = (r, c)
    if best is None:
        return True, 0

    r, c = best
    guesses = 0
    for v in sorted(cands[r][c]):
        guesses += 1
        budget[0] -= 1
        if budget[0] <= 0:
            raise SolverBudgetExceeded()
        grid_bak = [row[:] for row in grid]
        cands_bak = [[s.copy() for s in row] for row in cands]
        _place(grid, cands, r, c, v)
        ok, sub = _backtrack(grid, cands, budget)
        guesses += sub
        if ok:
            return True, guesses
        for rr in range(9):
            for cc in range(9):
                grid[rr][cc] = grid_bak[rr][cc]
                cands[rr][cc] = cands_bak[rr][cc]
    return False, guesses


def solve_with_techniques(grid: Grid, node_budget: int = DEFAULT_NODE_BUDGET) -> dict:
    g = deepcopy(grid)
    cands = _compute_candidates(g)
    techniques: Counter[str] = Counter()

    steps = [
        ("naked_single", _naked_single),
        ("hidden_single", _hidden_single),
        ("locked_candidates", _locked_candidates),
        ("naked_pair", _naked_pair),
    ]

    while any(0 in row for row in g):
        progressed = False
        for name, fn in steps:
            if fn(g, cands):
                techniques[name] += 1
                progressed = True
                break
        if not progressed:
            budget = [node_budget]
            ok, guesses = _backtrack(g, cands, budget)
            return {
                "solved": ok,
                "grid": g,
                "techniques": techniques,
                "backtracks": guesses,
            }

    return {"solved": True, "grid": g, "techniques": techniques, "backtracks": 0}


def random_solved(seed: int | None = None) -> Grid:
    """Generate a random valid completed 9x9 Sudoku."""
    rng = random.Random(seed)
    grid: Grid = [[0] * 9 for _ in range(9)]
    # Seed the three independent diagonal boxes with random permutations.
    for b in (0, 4, 8):
        cells = BOXES[b]
        vals = list(range(1, 10))
        rng.shuffle(vals)
        for (r, c), v in zip(cells, vals):
            grid[r][c] = v

    def safe(r, c, v):
        return all(grid[pr][pc] != v for pr, pc in PEERS[(r, c)])

    def fill() -> bool:
        for r in range(9):
            for c in range(9):
                if grid[r][c] == 0:
                    vals = list(range(1, 10))
                    rng.shuffle(vals)
                    for v in vals:
                        if safe(r, c, v):
                            grid[r][c] = v
                            if fill():
                                return True
                            grid[r][c] = 0
                    return False
        return True

    fill()
    return grid


def rate_difficulty(techniques: Counter, backtracks: int) -> int:
    """1..10 based on which techniques were needed."""
    if backtracks >= 20:
        return 10
    if backtracks >= 8:
        return 9
    if backtracks >= 1:
        return 8
    advanced = techniques.get("naked_pair", 0) + techniques.get("locked_candidates", 0)
    if advanced >= 3:
        return 7
    if advanced >= 1:
        return 6
    if techniques.get("hidden_single", 0) >= 5:
        return 5
    if techniques.get("hidden_single", 0) >= 1:
        return 4
    ns = techniques.get("naked_single", 0)
    if ns >= 40:
        return 3
    if ns >= 25:
        return 2
    return 1
