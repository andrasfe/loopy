# Agents

This document describes the agent architecture in loopy.

## Overview

Loopy is a single-agent system with a Monte-Carlo evolutionary loop.
The agent (an LLM accessed via OpenRouter) writes Python code that
generates Sudoku puzzles. A deterministic evaluator scores the output
and provides feedback. The agent rewrites its code each generation
until the target difficulty is reached.

## Agent: Proposer (LLM)

**Role:** Write and iteratively improve Python code that generates
Sudoku puzzles at a specified difficulty level.

**Input:**
- Target difficulty (1-10)
- In fresh mode: just the target and solver API docs
- In mutate mode: previous code + evaluator feedback + execution errors

**Output:** A Python script that prints a 9x9 JSON grid to stdout.

**Capabilities:** The generated code must be fully self-contained,
using only the Python standard library. It must implement everything
from scratch: grid generation, validation, solution counting, and
difficulty assessment. No project-local modules (solver.py, etc.)
are available — the executor runs code in an isolated temp directory.

**Modes:**
- `fresh` — cold start, writes code from scratch
- `mutate` — rewrites previous code guided by evaluator feedback

**Concurrency:** K proposals are sampled in parallel per generation
with jittered temperatures for diversity.

## Evaluator (deterministic code)

**Role:** Execute LLM-generated code, score the output puzzle, and
produce actionable feedback.

**Pipeline:**
1. Execute code in subprocess (10s timeout)
2. Extract 9x9 grid from stdout
3. Validate structure (shape, types, no conflicts)
4. Count solutions (must be exactly 1)
5. Rate difficulty using technique-tiered solver
6. Produce feedback: metrics + concrete improvement direction

**Fitness tiers** (higher is better):
| Condition | Fitness |
|---|---|
| Code crashed / timed out | -150 |
| No grid in stdout | -100 |
| Invalid grid structure | -80 |
| Multiple solutions | -50 |
| Valid, unique, solvable | `-(distance_to_target)` |

## Orchestrator (bootstrap.py)

**Role:** Manage the evolutionary loop.

- Maintains a population of top-K scoring code+puzzle pairs
- Runs fresh proposals when starting or after stagnation
- Runs mutations from the best parents otherwise
- Detects stagnation (no improvement for 2 generations) and reboots
- Tracks best-ever result across reboots
- Logs all events to JSONL

## Data Flow

```
proposer.fresh(k, target)
    → [Proposal(text, code, tokens)]

proposer.mutate(parent_code, feedback, ...)
    → [Proposal(text, code, tokens)]

evaluator.evaluate_code(code, target)
    → executor.run_code(code)
        → ExecutionResult(stdout, stderr, returncode, timed_out)
    → extract_grid(stdout)
    → solver.validate_structure(grid)
    → solver.count_solutions(grid)
    → solver.solve_with_techniques(grid)
    → solver.rate_difficulty(techniques, backtracks)
    → Score(code, grid, fitness, feedback, ...)
```

## Configuration

Set via `.env` file or environment variables:

- `OPENROUTER_API_KEY` — API key for OpenRouter
- `LLM_MODEL` — model identifier (e.g., `google/gemini-3.1-flash-lite-preview`)
- `OPENROUTER_BASE_URL` — API base URL (default: OpenRouter)
- `LOOPY_MODEL` — override for model (takes precedence over LLM_MODEL)
- `LOOPY_TEMPERATURE` — override for temperature (default: 0.7)
