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


SYSTEM_PROMPT = """You are a Sudoku puzzle ENGINEER. You write fully self-contained Python scripts that generate Sudoku puzzles.

RULES:
  - Use ONLY the Python standard library (json, random, copy, itertools, etc.)
  - Do NOT import any external packages or project-local modules.
  - Your script must implement everything it needs: grid generation, validation,
    solution counting, difficulty assessment — all from scratch.
  - Must print exactly one 9x9 JSON array to stdout: print(json.dumps(grid))
  - Zeros = blank cells, 1-9 = clues
  - The puzzle must have exactly one solution.
  - The script must complete within 10 seconds.

SUDOKU RULES (for your solver/validator):
  - 9x9 grid divided into nine 3x3 boxes
  - Each row, column, and box must contain digits 1-9 exactly once
  - A valid puzzle has exactly one solution

DIFFICULTY SCALE (our evaluator rates 1-10 based on solving techniques required):
  1-3 : trivial (~40-55 clues, solvable with naked singles only)
  4-5 : medium  (~30-38 clues, requires hidden singles)
  6-7 : hard    (~24-28 clues, requires locked candidates or naked pairs)
  8-10: expert  (~20-24 clues, requires backtracking / trial-and-error)

STRATEGY HINTS:
  - Generate a complete valid grid first (e.g., fill diagonal boxes then backtrack-fill the rest).
  - Remove cells one at a time, checking that exactly one solution remains.
  - For uniqueness checking, use backtracking that counts solutions up to 2.
  - Fewer clues generally means harder, but placement matters too.

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
