"""LLM proposer: asks an OpenRouter-hosted model for Python code that generates Sudoku puzzles.

Two modes:
  - fresh():  cold proposal — write code from scratch for a target difficulty
  - mutate(): rewrite previous code, guided by evaluator feedback

Runs K concurrent requests per round for Monte-Carlo sampling over variants.
"""
from __future__ import annotations

import concurrent.futures as futures
import re
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI

import config


SYSTEM_PROMPT = """You are a Sudoku puzzle ENGINEER. You write Python scripts that generate Sudoku puzzles.

Your script:
  - Can `import solver` which provides these functions:
      solver.random_solved(seed=None) -> list[list[int]]
        Returns a random fully-solved 9x9 Sudoku grid.
      solver.validate_structure(grid) -> (bool, str)
        Checks shape, types, and no same-unit conflicts.
      solver.count_solutions(grid, limit=2) -> int
        Counts solutions up to `limit`. Raises solver.SolverBudgetExceeded if too sparse.
      solver.solve_with_techniques(grid) -> dict
        Returns {"solved": bool, "grid": ..., "techniques": Counter, "backtracks": int}
      solver.rate_difficulty(techniques, backtracks) -> int
        Returns difficulty rating 1-10.
  - Can use any Python standard library module (json, random, copy, etc.)
  - Must print exactly one 9x9 JSON array to stdout: print(json.dumps(grid))
  - Zeros = blank cells, 1-9 = clues

DIFFICULTY SCALE (what the solver rates):
  1-3 : trivial (~40-55 clues, solver only needs naked singles)
  4-5 : medium  (~30-38 clues, requires hidden singles)
  6-7 : hard    (~24-28 clues, requires naked pairs / locked candidates)
  8-10: expert  (~20-24 clues, requires backtracking)

STRATEGY HINTS:
  - Start with solver.random_solved() to get a valid complete grid.
  - Remove cells one at a time, checking solver.count_solutions(grid, limit=2) == 1.
  - If removing a cell breaks uniqueness, skip it and try another.
  - Use solver.rate_difficulty() to check if you've reached the target.
  - Fewer clues generally means harder puzzles, but placement matters too.

OUTPUT: Respond with ONLY a Python code block. No prose, no explanation.
"""


USER_FRESH = """Write a Python script that generates a Sudoku puzzle at difficulty {target}/10.

Target approximately {target_clues} clues.
The script must print the 9x9 grid as a JSON array to stdout.

Output ONLY the Python code."""


USER_MUTATE = """Your previous code produced a puzzle with {current_clues} clues at difficulty {current_diff}/10.
Target difficulty is {target}/10 (approximately {target_clues} clues).

Evaluator feedback:
{feedback}

{exec_error_section}Previous code:
```python
{parent_code}
```

Rewrite the code to fix any issues and move toward the target difficulty.
Output ONLY the improved Python code."""


@dataclass
class Proposal:
    text: str
    code: Optional[str] = None
    tokens_in: int = 0
    tokens_out: int = 0
    error: Optional[str] = None


TARGET_CLUE_COUNT = {
    1: 52, 2: 46, 3: 40, 4: 34, 5: 30, 6: 27, 7: 25, 8: 23, 9: 22, 10: 21,
}


def _target_clues(target: int) -> int:
    return TARGET_CLUE_COUNT.get(target, 25)


def extract_code(text: str) -> Optional[str]:
    """Pull Python code from fenced code block or treat entire response as code."""
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    stripped = text.strip()
    if stripped and ("import " in stripped or "print(" in stripped or "def " in stripped):
        return stripped
    return None


def _client() -> OpenAI:
    s = config.load()
    return OpenAI(api_key=s.api_key, base_url=s.base_url)


def _call(client: OpenAI, model: str, temperature: float, user_msg: str) -> Proposal:
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            extra_headers={
                "HTTP-Referer": "https://github.com/andrasf/loopy",
                "X-Title": "loopy",
            },
        )
        text = resp.choices[0].message.content or ""
        usage = resp.usage
        code = extract_code(text)
        return Proposal(
            text=text,
            code=code,
            tokens_in=getattr(usage, "prompt_tokens", 0) if usage else 0,
            tokens_out=getattr(usage, "completion_tokens", 0) if usage else 0,
        )
    except Exception as e:
        return Proposal(text="", error=f"{type(e).__name__}: {e}")


def fresh(
    k: int,
    target: int,
    temperature: Optional[float] = None,
) -> list[Proposal]:
    """K cold proposals — ask LLM to write code from scratch."""
    settings = config.load()
    client = _client()
    base_t = temperature if temperature is not None else settings.temperature
    temps = [min(1.5, max(0.1, base_t + 0.1 * i - 0.2)) for i in range(k)]
    user = USER_FRESH.format(
        target=target,
        target_clues=_target_clues(target),
    )
    with futures.ThreadPoolExecutor(max_workers=k) as ex:
        jobs = [ex.submit(_call, client, settings.model, t, user) for t in temps]
        return [j.result() for j in jobs]


def mutate(
    parent_code: str,
    parent_feedback: str,
    parent_exec_error: Optional[str],
    parent_clues: int,
    parent_difficulty: int,
    k: int,
    target: int,
    temperature: Optional[float] = None,
) -> list[Proposal]:
    """K variations — send previous code + feedback, ask for rewrite."""
    settings = config.load()
    client = _client()
    base_t = temperature if temperature is not None else settings.temperature
    temps = [min(1.5, max(0.3, base_t + 0.2 * i)) for i in range(k)]

    exec_error_section = ""
    if parent_exec_error:
        exec_error_section = f"Execution error:\n{parent_exec_error}\n\n"

    user = USER_MUTATE.format(
        parent_code=parent_code,
        current_clues=parent_clues,
        current_diff=parent_difficulty,
        target=target,
        target_clues=_target_clues(target),
        feedback=parent_feedback,
        exec_error_section=exec_error_section,
    )
    with futures.ThreadPoolExecutor(max_workers=k) as ex:
        jobs = [ex.submit(_call, client, settings.model, t, user) for t in temps]
        return [j.result() for j in jobs]
