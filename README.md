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
success. The generated code must be fully self-contained — it runs in
an isolated temp directory with only the Python standard library
available. No pre-built solver or helpers are provided.

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

2. **Nothing pre-coded.** The generated code must implement everything
   from scratch: grid generation, validation, solution counting,
   difficulty heuristics. Only the Python standard library is available.
   The evaluator (solver.py) is used only by the harness to score output.

3. **Isolated execution.** Generated code runs in an isolated temp
   directory with a 10-second timeout. No project-local modules are
   importable. Process isolation prevents the LLM code from corrupting
   the orchestrator's state.

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

All results below use self-contained code generation: the LLM writes
everything from scratch (grid generation, validation, solution counting)
using only the Python standard library. No pre-built solver is available.

### google/gemini-3.1-flash-lite-preview — self-contained code gen

Target difficulty 7/10, K=4, keep=2, max-gen=8.

Converged at generation 6. Total tokens: 24,826 in + 18,517 out = 43,343.

```
gen  0 fresh   exec=4/4  valid=4  unique=4  on-target=0  best=diff=6  clues=25  tok=1874+2371   dt=7.4s
gen  1 mutate  exec=4/4  valid=4  unique=4  on-target=0  best=diff=6  clues=25  tok=4676+2784   dt=15.1s
gen  2 mutate  exec=3/4  valid=3  unique=3  on-target=0  best=diff=6  clues=25  tok=4674+2643   dt=26.2s
gen  3 reboot  exec=4/4  valid=4  unique=4  on-target=0  best=diff=5  clues=25  tok=1874+2553   dt=17.8s
gen  4 mutate  exec=3/4  valid=3  unique=3  on-target=0  best=diff=5  clues=25  tok=4926+2895   dt=31.2s
gen  5 mutate  exec=3/4  valid=3  unique=3  on-target=0  best=diff=5  clues=25  tok=4928+2854   dt=18.7s
gen  6 reboot  exec=4/4  valid=4  unique=4  on-target=1  best=diff=7  clues=25  tok=1874+2417   dt=4.7s
```

Winning puzzle (difficulty 7/10, 25 clues):

```
. 5 . | 2 3 . | . 8 7
. . . | . . 6 | . . .
. 3 . | 5 4 . | . . 6
---------------------
9 . 2 | . 6 . | . . .
. . . | 9 . . | . . 1
4 6 . | . . . | . . 3
---------------------
. . 7 | . . . | . 4 .
6 . 9 | . . 1 | . . .
8 . . | . 2 . | . . .
```

### moonshotai/kimi-k2.6 — self-contained code gen

Target difficulty 7/10, K=4, keep=2, max-gen=8.

Killed after gen 0 due to excessive token usage. Gen 0 alone consumed
1,790 in + 38,510 out = 40,300 tokens (mostly extended reasoning) and
took 601s. The code it generated produced difficulty 10 (too hard) — it
would have needed further iterations to dial down to 7.

```
gen  0 fresh   exec=3/4  valid=3  unique=3  on-target=0  best=diff=10  clues=25  tok=1790+38510  dt=601.1s
(killed — token budget exceeded)
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

### Comparison

| Mode | Model | Target | Result | Gens | Total Tokens | Time |
|---|---|---|---|---|---|---|
| v1 (grid data) | gemini-flash-lite | 7 | Failed (best: 3) | 8 (max) | n/a | ~24s |
| v2 (code, w/ solver) | gemini-flash-lite | 7 | Converged (7) | 2 | ~12K | ~30s |
| v2 (code, w/ solver) | kimi-k2.6 | 7 | Converged (7) | 0 | ~40K | ~491s |
| v3 (self-contained) | gemini-flash-lite | 7 | Converged (7) | 6 | 43,343 | ~121s |
| v3 (self-contained) | kimi-k2.6 | 7 | Killed (best: 10) | 0 | 40,300+ | 601s+ |

Key findings:
- Self-contained mode (v3) is harder — gemini needed 6 gens vs 2 with solver access
- Kimi's reasoning tokens (38K out in gen 0) make it impractical for iterative loops
- Gemini flash-lite is 10-100x more token-efficient per generation

## Honest limitations

- Convergence is probabilistic. Results vary between runs.
- The difficulty rating is a hand-picked ladder (naked_single →
  hidden_single → locked_candidates → naked_pair → backtrack). It's
  not the same as any published rating, but it's internally consistent.
- Generated code runs in an isolated temp dir with a timeout, but
  without network or filesystem sandboxing.
- Reasoning models (kimi-k2.6) produce better first-attempt code but
  consume 15x more tokens per generation due to extended thinking.
