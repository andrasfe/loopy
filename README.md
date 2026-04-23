# loopy

A recursive self-rewriting agent that converges on a goal by iterating
proposal → evaluation → rewrite. Each generation, the agent reads its own
previous output and the evaluator's feedback, then rewrites that output to
move closer to the target. The concrete task here is generating a Sudoku
puzzle at a specified difficulty level.

## The loop

```
   seed_grid  ─────────────────┐
                               ▼
   ┌────────────── PROPOSER (LLM) ──────────────┐
   │ reads: seed_grid, parent_puzzle, feedback  │
   │ writes: K variant puzzles                  │◄──── rewrites previous
   └────────────────────────────────────────────┘      output based on
                               │                       evaluator feedback
                               ▼
   ┌─────────────── EVALUATOR (code) ───────────┐
   │ validates structure, counts solutions,     │
   │ runs technique-tiered solver, rates 1-10   │
   │ produces imperative feedback               │
   └────────────────────────────────────────────┘
                               │
                               ▼
            top-K retained → next generation
            stagnant?  → reboot (discard parents)
            on target? → done
```

The agent is "recursive" in the sense that each generation's input is
itself produced by the previous generation's output — the LLM is rewriting
its own prior draft, not starting fresh every round. The evaluator is
entirely deterministic code; the LLM never judges its own success.

The same scaffold works for any task where candidates can be scored cheaply
and programmatically (algorithmic problems against test cases, regex
inference from labeled examples, strategy search with a simulator, etc.).
Swap the proposer prompt and the evaluator, keep the loop.

## Architecture

| File | Role |
|---|---|
| `bootstrap.py` | MC loop, population, stagnation reboot, jsonl logging |
| `proposer.py` | OpenRouter LLM calls (OpenAI SDK); fresh + mutate modes with numeric action directives |
| `evaluator.py` | Parses LLM output, validates, scores, produces feedback |
| `solver.py` | Uniqueness check, technique-tiered solver (naked/hidden singles, naked pair, locked candidates), budgeted backtracking, 1-10 difficulty rating, `random_solved()` seed generator |
| `config.py` | Loads `../specter/.env` for OpenRouter API credentials |

## Key design choices

1. **LLM is seeded with a fully solved Sudoku, not asked to invent one.**
   "Generate a valid unique Sudoku from scratch" fails almost always. "Mask
   cells of this solved grid" succeeds. The MC + evaluator pattern is
   intact, the task is just tractable.

2. **Numeric directives beat vague advice.** The evaluator feedback says
   "remove ~15 cells, go from 40 clues to 25" rather than "too easy, try
   harder."

3. **Tie-break fitness.** Two candidates at the same difficulty are ranked
   by clue-count distance to target. Progress propagates through plateaus
   where difficulty refuses to move.

4. **Stagnation reboot.** If the best doesn't improve for 2 generations,
   parents are abandoned and a fresh batch is seeded. Best-ever is tracked
   across reboots. Most runs' final answer came from a post-reboot lineage.

5. **Budget the solver.** Sparse LLM outputs can balloon the backtracking
   search to minutes. `SolverBudgetExceeded` caps the work at 200k nodes
   and returns a useful "too hard, add clues" signal.

## Running

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# uses ../specter/.env for OPENROUTER_API_KEY; override with LOOPY_MODEL, LOOPY_TEMPERATURE
LOOPY_MODEL=anthropic/claude-haiku-4.5 \
.venv/bin/python -u bootstrap.py --target 7 -k 8 --keep 2 --max-gen 15
```

Flags:

- `--target` target difficulty 1–10
- `-k` proposals per generation (MC width)
- `--keep` parents retained between generations
- `--max-gen` max generations
- `--seed` RNG seed for the solved reference grid
- `--log` path for the jsonl event log (default `runs/run-{ts}.jsonl`)

Use `python -u` — plain `python` buffers stdout and the progress stream
won't appear until exit.

## Honest limitations

- Convergence is probabilistic. A 15-generation run targeting difficulty 7
  lands on 6 or 7 most of the time, occasionally 5. Longer runs or wider
  K help.
- The difficulty rating is a hand-picked ladder (naked_single → hidden_single
  → locked_candidates → naked_pair → backtrack). It's not the same as any
  published rating, but it's internally consistent.
- The LLM never rewrites `bootstrap.py` itself — only the puzzle. For a
  true code-rewriting agent (same loop, Python source as the artifact),
  the scaffold is identical; swap the proposer prompt from "Sudoku mask"
  to "code diff" and the evaluator from "solver" to "pytest".
