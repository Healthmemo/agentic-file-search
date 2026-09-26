"""LM provider configuration for the DSPy spike.

Reads configuration from environment variables (never from committed secrets).

Env vars
--------
DSPY_LM_PROVIDER : ``ollama`` (default) or ``bedrock``
DSPY_OLLAMA_MODEL : Ollama model id (default ``llama3.2``)
DSPY_OLLAMA_BASE_URL : Ollama API base (default ``http://localhost:11434``)
DSPY_BEDROCK_MODEL : Bedrock model id (default ``amazon.nova-lite-v1:0``)
DSPY_BEDROCK_REGION / AWS_REGION / AWS_DEFAULT_REGION : AWS region
AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_PROFILE : standard AWS creds
DSPY_LM_TEMPERATURE : optional float
DSPY_LM_MAX_TOKENS : optional int (default 2048)
"""

from __future__ import annotations

import os
from typing import Any, Literal

ProviderName = Literal["ollama", "bedrock"]

DEFAULT_PROVIDER: ProviderName = "ollama"
DEFAULT_OLLAMA_MODEL = "llama3.2"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_BEDROCK_MODEL = "amazon.nova-lite-v1:0"
DEFAULT_BEDROCK_REGION = "us-east-1"
DEFAULT_MAX_TOKENS = 2048


class LMConfigError(ValueError):
    """Raised when LM provider configuration is invalid or incomplete."""


def _optional_float(name: str) -> float | None:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    try:
        return float(raw)
    except ValueError as exc:
        raise LMConfigError(f"{name} must be a float, got {raw!r}") from exc


def _optional_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise LMConfigError(f"{name} must be an int, got {raw!r}") from exc


def resolve_provider(explicit: str | None = None) -> ProviderName:
    """Resolve LM provider from argument or ``DSPY_LM_PROVIDER``."""
    raw = (
        (explicit or os.getenv("DSPY_LM_PROVIDER") or DEFAULT_PROVIDER).strip().lower()
    )
    if raw not in ("ollama", "bedrock"):
        raise LMConfigError(
            f"DSPY_LM_PROVIDER must be 'ollama' or 'bedrock', got {raw!r}"
        )
    return raw  # type: ignore[return-value]


def build_lm(
    provider: str | None = None,
    *,
    model: str | None = None,
    configure: bool = True,
) -> Any:
    """Build a ``dspy.LM`` for Ollama or Bedrock and optionally configure DSPy.

    Parameters
    ----------
    provider:
        ``ollama`` or ``bedrock``. Defaults to ``DSPY_LM_PROVIDER`` / ollama.
    model:
        Override model id. Defaults to provider-specific env vars.
    configure:
        If True, call ``dspy.configure(lm=...)``.
    """
    try:
        import dspy
    except ImportError as exc:
        raise LMConfigError(
            "dspy is not installed. Install with: uv pip install -e '.[dspy]' "
            "or uv pip install dspy"
        ) from exc

    resolved = resolve_provider(provider)
    temperature = _optional_float("DSPY_LM_TEMPERATURE")
    max_tokens = _optional_int("DSPY_LM_MAX_TOKENS", DEFAULT_MAX_TOKENS)

    kwargs: dict[str, Any] = {"max_tokens": max_tokens}
    if temperature is not None:
        kwargs["temperature"] = temperature

    if resolved == "ollama":
        model_id = model or os.getenv("DSPY_OLLAMA_MODEL") or DEFAULT_OLLAMA_MODEL
        api_base = (
            os.getenv("DSPY_OLLAMA_BASE_URL") or DEFAULT_OLLAMA_BASE_URL
        ).rstrip("/")
        # DSPy / LiteLLM Ollama chat route
        lm_model = (
            model_id if model_id.startswith("ollama") else f"ollama_chat/{model_id}"
        )
        lm = dspy.LM(lm_model, api_base=api_base, api_key="", **kwargs)
    else:
        model_id = model or os.getenv("DSPY_BEDROCK_MODEL") or DEFAULT_BEDROCK_MODEL
        region = (
            os.getenv("DSPY_BEDROCK_REGION")
            or os.getenv("AWS_REGION")
            or os.getenv("AWS_DEFAULT_REGION")
            or DEFAULT_BEDROCK_REGION
        )
        lm_model = (
            model_id if model_id.startswith("bedrock/") else f"bedrock/{model_id}"
        )
        # LiteLLM reads AWS_* env vars; region_name is also accepted by many versions.
        lm = dspy.LM(lm_model, region_name=region, **kwargs)

    if configure:
        dspy.configure(lm=lm)
    return lm
