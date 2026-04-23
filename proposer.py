"""LLM proposer: asks an OpenRouter-hosted model for Sudoku puzzles.

Two modes:
  - fresh():  cold proposal at a target difficulty
  - mutate(): variation from a parent puzzle, given critic feedback

Runs K concurrent requests per round for Monte-Carlo sampling over variants.
"""
from __future__ import annotations

import concurrent.futures as futures
import json
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI

import config


SYSTEM_PROMPT = """You are a Sudoku puzzle CRAFTER working from a fully-solved reference grid.

Your task per turn: choose a MASK - which cells of the reference grid remain as
clues (non-zero) and which become 0 (blanks). The non-zero cells you output MUST
match the reference grid exactly; you only decide keep-or-blank per cell.

DIFFICULTY SCALE (inferred from how many cells must remain):
  diff 1-3 : ~40-55 clues  (trivial, solver only needs naked singles)
  diff 4-5 : ~30-38 clues  (requires hidden singles)
  diff 6-7 : ~24-28 clues  (requires naked pairs / locked candidates)  <- HARD
  diff 8-10: ~20-24 clues  (requires backtracking / expert techniques)

HARD RULES:
  - Output ONLY a 9x9 JSON array of integers (0 = blank, else matches reference).
  - No prose, no markdown, no explanation.
  - Every row, column, and 3x3 box must keep at least 2 clues.
  - The resulting puzzle must still have exactly ONE solution.
"""


USER_FRESH = """REFERENCE SOLVED GRID:
{seed_grid}

TARGET difficulty: {target}/10  -->  keep approximately {target_clues} cells, blank the rest.

Produce the masked grid (81 cells, of which ~{target_clues} match the reference and the rest are 0).
Output ONLY the 9x9 JSON array."""


USER_MUTATE = """REFERENCE SOLVED GRID (all non-zero cells must match this exactly):
{seed_grid}

Your previous puzzle had {current_clues} clues and difficulty {current_diff}.
TARGET difficulty is {target}, which needs approximately {target_clues} clues.

{action_directive}

Previous puzzle:
{parent_grid}

Produce the IMPROVED masked grid. Output ONLY the 9x9 JSON array."""


@dataclass
class Proposal:
    text: str
    tokens_in: int = 0
    tokens_out: int = 0
    error: Optional[str] = None


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
        return Proposal(
            text=text,
            tokens_in=getattr(usage, "prompt_tokens", 0) if usage else 0,
            tokens_out=getattr(usage, "completion_tokens", 0) if usage else 0,
        )
    except Exception as e:
        return Proposal(text="", error=f"{type(e).__name__}: {e}")


def _format_grid(grid: list[list[int]]) -> str:
    return "[\n" + ",\n".join("  " + json.dumps(row) for row in grid) + "\n]"


TARGET_CLUE_COUNT = {
    1: 52, 2: 46, 3: 40, 4: 34, 5: 30, 6: 27, 7: 25, 8: 23, 9: 22, 10: 21,
}


def _target_clues(target: int) -> int:
    return TARGET_CLUE_COUNT.get(target, 25)


def fresh(
    seed_grid: list[list[int]],
    k: int,
    target: int,
    temperature: Optional[float] = None,
) -> list[Proposal]:
    """K cold proposals. Temperature jittered per sample to diversify."""
    settings = config.load()
    client = _client()
    base_t = temperature if temperature is not None else settings.temperature
    temps = [min(1.5, max(0.1, base_t + 0.1 * i - 0.2)) for i in range(k)]
    tgt_clues = _target_clues(target)
    user = USER_FRESH.format(
        target=target,
        target_clues=tgt_clues,
        seed_grid=_format_grid(seed_grid),
    )
    with futures.ThreadPoolExecutor(max_workers=k) as ex:
        jobs = [ex.submit(_call, client, settings.model, t, user) for t in temps]
        return [j.result() for j in jobs]


def _action_directive(current_clues: int, current_diff: int, target_diff: int) -> str:
    target_clues = _target_clues(target_diff)
    delta = current_clues - target_clues
    if current_diff == target_diff:
        return f"Status: ON TARGET. Produce a structurally different mask at similar clue count (~{target_clues})."
    if current_diff < target_diff:
        return (
            f"Status: TOO EASY. You MUST REMOVE approximately {delta} more cells "
            f"(blank them to 0). Go from {current_clues} clues down to ~{target_clues}."
        )
    # current_diff > target_diff
    return (
        f"Status: TOO HARD. You MUST ADD approximately {-delta} more cells "
        f"(restore them from the reference). Go from {current_clues} clues up to ~{target_clues}."
    )


def mutate(
    seed_grid: list[list[int]],
    parent_grid: list[list[int]],
    parent_feedback: str,
    parent_clues: int,
    parent_difficulty: int,
    k: int,
    target: int,
    temperature: Optional[float] = None,
) -> list[Proposal]:
    """K variations of a parent, guided by concrete action directive."""
    settings = config.load()
    client = _client()
    base_t = temperature if temperature is not None else settings.temperature
    temps = [min(1.5, max(0.3, base_t + 0.2 * i)) for i in range(k)]
    user = USER_MUTATE.format(
        seed_grid=_format_grid(seed_grid),
        parent_grid=_format_grid(parent_grid),
        current_clues=parent_clues,
        current_diff=parent_difficulty,
        target=target,
        target_clues=_target_clues(target),
        action_directive=_action_directive(parent_clues, parent_difficulty, target),
    )
    with futures.ThreadPoolExecutor(max_workers=k) as ex:
        jobs = [ex.submit(_call, client, settings.model, t, user) for t in temps]
        return [j.result() for j in jobs]
