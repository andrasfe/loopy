"""LLM proposer: asks an OpenRouter-hosted model for code that solves a task.

Two modes:
  - fresh():  cold proposal — write code from scratch
  - mutate(): rewrite previous code, guided by evaluator feedback

Runs K concurrent requests per round for Monte-Carlo sampling over variants.
"""
from __future__ import annotations

import concurrent.futures as futures
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from openai import OpenAI

import config


def load_task_prompt(task_path: str | Path) -> str:
    """Load a task prompt from a .md file, stripping YAML frontmatter if present."""
    text = Path(task_path).read_text()
    # Strip YAML frontmatter (--- ... ---)
    if text.startswith("---"):
        end = text.find("---", 3)
        if end != -1:
            text = text[end + 3:]
    return text.strip()


# Module-level prompt, set by init() before any calls.
_system_prompt: str = ""


def init(task_path: str | Path) -> None:
    """Initialize the proposer with a task prompt file."""
    global _system_prompt
    _system_prompt = load_task_prompt(task_path)


USER_FRESH = """Target: {target}/10.

Output ONLY the Python code."""


USER_MUTATE = """Your previous code scored {current_diff}/10. Target is {target}/10.

Evaluator feedback:
{feedback}

{exec_error_section}Previous code:
```python
{parent_code}
```

Rewrite the code to fix any issues and move toward the target.
Output ONLY the improved Python code."""


@dataclass
class Proposal:
    text: str
    code: Optional[str] = None
    tokens_in: int = 0
    tokens_out: int = 0
    error: Optional[str] = None



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
                {"role": "system", "content": _system_prompt},
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
    user = USER_FRESH.format(target=target)
    with futures.ThreadPoolExecutor(max_workers=k) as ex:
        jobs = [ex.submit(_call, client, settings.model, t, user) for t in temps]
        return [j.result() for j in jobs]


def mutate(
    parent_code: str,
    parent_feedback: str,
    parent_exec_error: Optional[str],
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
        current_diff=parent_difficulty,
        target=target,
        feedback=parent_feedback,
        exec_error_section=exec_error_section,
    )
    with futures.ThreadPoolExecutor(max_workers=k) as ex:
        jobs = [ex.submit(_call, client, settings.model, t, user) for t in temps]
        return [j.result() for j in jobs]
