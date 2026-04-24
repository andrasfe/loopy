"""Execute LLM-generated Python code in a subprocess with timeout."""
from __future__ import annotations

import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

PROJECT_DIR = Path(__file__).resolve().parent


@dataclass
class ExecutionResult:
    stdout: str
    stderr: str
    returncode: int
    timed_out: bool
    error: Optional[str] = None


def run_code(code: str, timeout: float = 10.0) -> ExecutionResult:
    """Write code to a temp file and run it in a subprocess."""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", dir=PROJECT_DIR, delete=False
        ) as f:
            f.write(code)
            tmp_path = Path(f.name)
        try:
            proc = subprocess.run(
                [sys.executable, str(tmp_path)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(PROJECT_DIR),
            )
            return ExecutionResult(
                stdout=proc.stdout,
                stderr=proc.stderr,
                returncode=proc.returncode,
                timed_out=False,
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                stdout="",
                stderr="",
                returncode=-1,
                timed_out=True,
                error=f"Timed out after {timeout}s",
            )
        finally:
            tmp_path.unlink(missing_ok=True)
    except Exception as e:
        return ExecutionResult(
            stdout="",
            stderr=str(e),
            returncode=-1,
            timed_out=False,
            error=f"{type(e).__name__}: {e}",
        )
