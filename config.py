"""Loads .env from ../specter and exposes LLM settings."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parent.parent / "specter" / ".env"
LOCAL_ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(ENV_PATH)
load_dotenv(LOCAL_ENV_PATH, override=True)


@dataclass(frozen=True)
class Settings:
    api_key: str
    base_url: str
    model: str
    temperature: float


def load() -> Settings:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            f"OPENROUTER_API_KEY not found. Looked at {ENV_PATH}."
        )
    return Settings(
        api_key=api_key,
        base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        model=os.environ.get("LOOPY_MODEL", os.environ.get("LLM_MODEL", "google/gemini-3-flash-preview")),
        temperature=float(os.environ.get("LOOPY_TEMPERATURE", os.environ.get("LLM_TEMPERATURE", "0.7"))),
    )
