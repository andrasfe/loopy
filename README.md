# loopy

A generic scaffold for self-improving LLM agents. Given a problem
statement and a deterministic evaluator, the LLM writes code to solve
the problem, the evaluator scores the output and produces feedback, and
the LLM rewrites its own code to improve. The loop runs for N
generations with Monte-Carlo sampling (K proposals per generation),
elitist selection, and stagnation reboots.

The scaffold is task-agnostic: swap the proposer prompt and the
evaluator, keep the loop. Two tasks ship out of the box: Sudoku
puzzle generation and Max-Cut optimization.

## Why this works for algorithm design

Greedy algos are a natural fit. The loop already works this way — the
LLM writes a greedy algorithm in gen 0, the evaluator scores it, and
the feedback tells it how far off it is. The LLM then rewrites the
code to use a better approach. We saw this with Max-Cut: the LLM
started with random partitions + local search and found the optimal
cut on the first try.

Algorithm swapping through iterations happens organically. The LLM
receives its previous code + evaluator feedback like "your greedy
approach scored 60%, try local search or simulated annealing." It can
completely rewrite the algorithm — nothing constrains it to incremental
edits. In practice we've seen:

- Gen 0: random/naive approach
- Gen 1: greedy with the same structure
- Gen 2: completely different algorithm (e.g., switching from greedy
  to backtracking with pruning)

The feedback loop is what drives it. If the evaluator says "your O(n!)
brute force timed out" the LLM switches to a polynomial approximation.
If it says "your greedy gets 70% but target is 90%" the LLM might
switch to simulated annealing or branch-and-bound.

Good candidate problems for this pattern:

- **Knapsack** (greedy → DP → branch-and-bound)
- **TSP** (nearest neighbor → 2-opt → simulated annealing)
- **Graph coloring** (greedy → backtracking → tabu search)
- **Bin packing** (first-fit → best-fit → metaheuristics)

The key requirement is a fast deterministic evaluator — if you can
score the output in milliseconds, the loop can iterate quickly. The
10-second execution timeout already pressures the LLM away from brute
force toward smarter algorithms.

## The loop

```
   ┌────────────── PROPOSER (LLM) ──────────────┐
   │ reads: previous code, evaluator feedback    │
   │ writes: K variant Python scripts            │◄──── rewrites previous
   └────────────────────────────────────────────┘      code based on
                               │                       evaluator feedback
                               ▼
   ┌─────────────── EXECUTOR (subprocess) ─────────┐
   │ runs LLM-generated code in isolated sandbox   │
   │ captures stdout                               │
   └────────────────────────────────────────────────┘
                               │
                               ▼
   ┌─────────────── EVALUATOR (deterministic) ──┐
   │ scores output, produces imperative feedback │
   └────────────────────────────────────────────┘
                               │
                               ▼
            top-K retained → next generation
            stagnant?  → reboot (discard parents)
            on target? → done
```

The LLM rewrites its own prior code each generation. The evaluator is
entirely deterministic — the LLM never judges its own success. Generated
code must be fully self-contained (only the Python standard library is
available). No pre-built solver or helpers are provided.

## Architecture

| File | Role |
|---|---|
| `bootstrap.py` | MC loop, population, stagnation reboot, jsonl logging |
| `proposer.py` | LLM calls (OpenAI SDK via OpenRouter); fresh + mutate modes |
| `evaluator.py` | Executes code, scores output, produces feedback |
| `executor.py` | Runs generated Python in an isolated subprocess with timeout |
| `solver.py` | Sudoku-specific: uniqueness check, technique-tiered solver, difficulty rating (used by evaluator only) |
| `config.py` | Loads `.env` for API credentials |

## Key design choices

1. **LLM writes code, not data.** The artifact being iterated is source
   code. The LLM learns to write better algorithms through feedback.

2. **Nothing pre-coded.** Generated code must implement everything from
   scratch. The evaluator uses its own solver only for scoring.

3. **Isolated execution.** Code runs in an isolated temp directory with
   a 10-second timeout. No project-local modules are importable.

4. **Incremental fitness tiers.** Fitness rewards progress: code runs
   → output parsed → valid → unique → on-target. This lets the loop
   recover from broken code through the feedback cycle.

5. **Stagnation reboot.** If the best doesn't improve for 2 generations,
   parents are abandoned and a fresh batch is seeded.

## Running

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# configure .env with OPENROUTER_API_KEY and LLM_MODEL
python -u bootstrap.py --target 7 -k 4 --keep 2 --max-gen 8
```

Flags: `--target` (difficulty 1-10), `-k` (proposals per generation),
`--keep` (parents retained), `--max-gen`, `--seed`, `--log`.

## Results

All runs target difficulty 7/10, K=4, keep=2, max-gen=8. Generated code
is fully self-contained (standard library only).

| Model | Result | Gens | Total Tokens |
|---|---|---|---|
| openai/gpt-oss-20b | Converged (7) | 0 | 9,422 |
| google/gemma-4-26b-a4b-it | Converged (7) | 4 | 42,456 |
| google/gemini-3.1-flash-lite | Converged (7) | 6 | 43,343 |
| qwen/qwen-turbo | Converged (7) | 7 | 61,450 |
| openai/gpt-5-nano | Converged (7) | 3 | 159,199 |
| x-ai/grok-4.1-fast | Converged (7) | 7 | 272,555 |
| nvidia/nemotron-nano-9b-v2 | Failed (best: 5) | 8 | 352,498 |
| inclusionai/ling-2.6-flash | Failed (best: 5) | 8 | 90,578 |
| liquid/lfm-2-24b-a2b | Failed (best: 3) | 8 | 74,335 |
| amazon/nova-micro-v1 | Failed (best: 1) | 8 | 54,378 |
| mistralai/mistral-nemo | Failed (best: 0) | 8 | 51,231 |
| ibm-granite/granite-4.0-h-micro | Failed (best: 0) | 8 | 48,549 |
| moonshotai/kimi-k2.6 | Killed (best: 10) | 0 | 40,300+ |

Key findings:

- **GPT-OSS-20B** converged on the first generation with only 9K tokens
- Non-reasoning models (gemma-4, gemini-flash-lite) are the most
  token-efficient when they converge
- Reasoning models (gpt-5-nano, grok, kimi) use 15-35K output tokens
  per generation due to hidden thinking — expensive for iterative loops
- Many small/cheap models fail entirely — they cannot write correct
  self-contained Sudoku solvers
- The difficulty 5→7 gap is the hardest barrier: models plateau at 5-6

## Honest limitations

- Convergence is probabilistic. Results vary between runs.
- The difficulty rating is a hand-picked ladder, not a published standard.
- Generated code runs in an isolated temp dir with a timeout, but
  without network or filesystem sandboxing.
- Reasoning models produce better first-attempt code but consume
  15-30x more tokens per generation.
