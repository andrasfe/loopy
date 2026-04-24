"""Execute LLM-generated Python code in an isolated subprocess with timeout."""
from __future__ import annotations

import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ExecutionResult:
    stdout: str
    stderr: str
    returncode: int
    timed_out: bool
    error: Optional[str] = None


def run_code(code: str, timeout: float = 10.0) -> ExecutionResult:
    """Write code to a temp dir and run it in a subprocess.

    Runs in an isolated temp directory so the generated code cannot
    import project-local modules (solver.py, etc.). Only the standard
    library is available.
    """
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / "generated.py"
            tmp_path.write_text(code)
            try:
                proc = subprocess.run(
                    [sys.executable, str(tmp_path)],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=tmpdir,
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
    except Exception as e:
        return ExecutionResult(
            stdout="",
            stderr=str(e),
            returncode=-1,
            timed_out=False,
            error=f"{type(e).__name__}: {e}",
        )
