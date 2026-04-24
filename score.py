"""Generic score dataclass used by all evaluators."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Score:
    code: Optional[str]
    valid: bool
    difficulty: int                    # achieved level (task-specific meaning)
    target_distance: float = 9.0      # |achieved - target|, lower is better
    feedback: str = ""
    raw: str = ""                      # stdout from code execution
    exec_error: Optional[str] = None
    details: dict[str, Any] = field(default_factory=dict)  # task-specific extras

    @property
    def fitness(self) -> float:
        """Higher is better. Generic tiers:
        code ran > output valid > on-target."""
        if self.exec_error:
            return -150
        if not self.valid:
            return -100
        return -self.target_distance
