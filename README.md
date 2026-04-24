# loopy

A recursive self-rewriting agent that converges on a goal by iterating
proposal → evaluation → rewrite. The LLM writes **Python code** that
generates Sudoku puzzles; the evaluator executes the code, scores the
output puzzle, and feeds metrics back. The LLM then rewrites its own
code to move closer to the target difficulty.

## The loop

```
   ┌────────────── PROPOSER (LLM) ──────────────┐
   │ reads: previous code, evaluator feedback    │
   │ writes: K variant Python scripts            │◄──── rewrites previous
   └────────────────────────────────────────────┘      code based on
                               │                       evaluator feedback
                               ▼
   ┌─────────────── EXECUTOR (subprocess) ─────────┐
   │ runs LLM-generated code in sandboxed process  │
   │ captures stdout (the 9x9 grid)                │
   └────────────────────────────────────────────────┘
                               │
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

The LLM is rewriting its own prior code, not starting fresh every round.
The evaluator is entirely deterministic; the LLM never judges its own
success. The generated code can `import solver` to use the project's
solver API (random_solved, count_solutions, rate_difficulty, etc.).

The same scaffold works for any task where candidates can be scored
cheaply and programmatically. Swap the proposer prompt and the evaluator,
keep the loop.

## Architecture

| File | Role |
|---|---|
| `bootstrap.py` | MC loop, population, stagnation reboot, jsonl logging |
| `proposer.py` | OpenRouter LLM calls (OpenAI SDK); fresh + mutate modes; extracts Python code from responses |
| `evaluator.py` | Executes code, extracts grid from stdout, validates, scores, produces feedback |
| `executor.py` | Runs LLM-generated Python in a subprocess with timeout |
| `solver.py` | Uniqueness check, technique-tiered solver (naked/hidden singles, naked pair, locked candidates), budgeted backtracking, 1-10 difficulty rating, `random_solved()` generator |
| `config.py` | Loads `.env` for OpenRouter API credentials |

## Key design choices

1. **LLM writes code, not data.** The LLM produces a Python script that
   generates puzzles, not a raw grid. This makes it a true self-rewriting
   agent — the artifact being iterated is code, and the LLM learns to
   write better generation algorithms through feedback.

2. **Code has access to the solver API.** The generated code can
   `import solver` and use `random_solved()`, `count_solutions()`,
   `rate_difficulty()`, etc. This makes the task tractable — the LLM
   doesn't need to reinvent constraint propagation.

3. **Sandboxed execution.** Generated code runs in a subprocess with a
   10-second timeout. Process isolation prevents the LLM code from
   corrupting the orchestrator's state.

4. **Incremental fitness tiers.** Fitness rewards progress:
   code runs (-150 if crash) → grid extracted (-100) → valid (-80) →
   unique (-50) → on-target (0). This lets the loop recover from
   broken code through the feedback cycle.

5. **Stagnation reboot.** If the best doesn't improve for 2 generations,
   parents are abandoned and a fresh batch is seeded. Best-ever is tracked
   across reboots.

6. **Budget the solver.** `SolverBudgetExceeded` caps work at 200k nodes
   and returns a useful "too hard, add clues" signal.

## Running

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# configure .env with OPENROUTER_API_KEY and LLM_MODEL
python -u bootstrap.py --target 7 -k 4 --keep 2 --max-gen 8
```

Flags:

- `--target` target difficulty 1-10
- `-k` proposals per generation (MC width)
- `--keep` parents retained between generations
- `--max-gen` max generations
- `--seed` RNG seed
- `--log` path for the jsonl event log (default `runs/run-{ts}.jsonl`)

Use `python -u` — plain `python` buffers stdout and the progress stream
won't appear until exit.

## Results

### google/gemini-3.1-flash-lite-preview — code-generation mode

Target difficulty 7/10, K=4, keep=2, max-gen=8.

Converged at generation 2. All proposals produced valid, executable code
from generation 0. The LLM iterated from difficulty 5 → 6 → 7 across
three generations.

```
gen  0 fresh   exec=4/4  valid=4  unique=4  on-target=0  best=diff=5  clues=25  fit=-2   dt=3.5s
gen  1 mutate  exec=4/4  valid=4  unique=4  on-target=0  best=diff=6  clues=25  fit=-1   dt=14.4s
gen  2 mutate  exec=4/4  valid=4  unique=4  on-target=2  best=diff=7  clues=25  fit=0    dt=11.8s
```

Winning puzzle (difficulty 7/10, 25 clues):

```
. . . | 6 9 . | . 3 .
4 . 1 | . . . | 8 . .
. . . | . . . | 6 . .
---------------------
. . . | . 2 . | . . 3
1 . . | . . . | . 8 7
8 . 3 | 5 . 7 | 9 2 1
---------------------
. . . | . 3 . | . 4 6
5 . . | . . . | . . .
. . 8 | . . . | 2 . .
```

### moonshotai/kimi-k2.6 — code-generation mode

Target difficulty 7/10, K=4, keep=2, max-gen=8.

Converged at generation 0. All 4 proposals produced working code that
hit difficulty 7 on the first try. The reasoning model wrote a
sophisticated two-phase removal algorithm with a 300-attempt retry loop.
Slow (491s for gen 0) but accurate.

```
gen  0 fresh   exec=4/4  valid=4  unique=4  on-target=4  best=diff=7  clues=25  fit=0    dt=491.2s
```

Winning puzzle (difficulty 7/10, 25 clues):

```
. . . | . . 8 | . . 4
4 . 9 | . 1 . | . . 6
. . . | 2 . . | 7 . .
---------------------
2 . . | . 4 1 | . . .
. . . | . . 9 | . 7 5
. . 4 | 6 . . | . 3 .
---------------------
. 7 . | . . . | . . .
6 . . | . . 3 | 5 . 1
. 8 . | . . . | . 6 7
```

### google/gemini-3.1-flash-lite-preview — v1 data-generation mode

Target difficulty 7/10, K=6, keep=2, max-gen=8.

The original v1 architecture had the LLM directly output puzzle grids
(masking a seed solved grid). Reached max generations without hitting
target. Best puzzle found at generation 4: difficulty 3/10, 33 clues.

```
gen  0 fresh   valid=6/6  unique=0  on-target=0  best=NON-UNIQUE  fit=-50  clues=21
gen  1 mutate  valid=6/6  unique=0  on-target=0  best=NON-UNIQUE  fit=-50  clues=21
gen  2 mutate  valid=6/6  unique=0  on-target=0  best=NON-UNIQUE  fit=-50  clues=21
gen  3 reboot  valid=6/6  unique=0  on-target=0  best=NON-UNIQUE  fit=-50  clues=22
gen  4 mutate  valid=6/6  unique=1  on-target=0  best=diff=3  clues=33  fit=-4
gen  5 mutate  valid=6/6  unique=1  on-target=0  best=diff=3  clues=33  fit=-4
gen  6 mutate  valid=6/6  unique=0  on-target=0  best=diff=3  clues=33  fit=-4
gen  7 reboot  valid=6/6  unique=0  on-target=0  best=NON-UNIQUE  fit=-50  clues=22
```

Best puzzle (difficulty 3/10, 33 clues, solved with naked singles only):

```
4 . 3 | . . 6 | 7 . .
. 1 . | 7 . . | . 4 3
7 . . | . 5 . | 6 . .
---------------------
9 . . | 5 . 1 | . . 7
. 8 . | . 2 . | . 3 .
5 . . | 9 . 3 | . . 6
---------------------
. . 1 | . 3 . | . . 4
8 6 . | . . 5 | . 9 .
. . 9 | . . 2 | 8 . 1
```

### Comparison: data-generation vs code-generation mode

With gemini-3.1-flash-lite-preview targeting difficulty 7, v1 failed
after 8 generations (best: difficulty 3). The code-generation v2
converged in 2 generations with the same model.

| Mode | Model | Target | Result | Generations |
|---|---|---|---|---|
| v1 (grid data) | gemini-flash-lite | 7 | Failed (best: 3) | 8 (max) |
| v2 (code gen) | gemini-flash-lite | 7 | Converged (7) | 2 |
| v2 (code gen) | kimi-k2.6 | 7 | Converged (7) | 0 |

## Honest limitations

- Convergence is probabilistic. Results vary between runs.
- The difficulty rating is a hand-picked ladder (naked_single →
  hidden_single → locked_candidates → naked_pair → backtrack). It's
  not the same as any published rating, but it's internally consistent.
- Generated code runs unsandboxed beyond a timeout. For production use,
  add network isolation and resource limits.
- Reasoning models (kimi-k2.6) produce better first-attempt code but
  are 15-20x slower per generation than flash models.
