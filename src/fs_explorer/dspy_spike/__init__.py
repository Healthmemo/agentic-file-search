"""Minimal DSPy spike for medicolegal retrieve → categorize → extract.

Production LM path: Amazon Bedrock. Local/dev default: Ollama.
Postgres is not required — use the in-memory fixture page store.

See README.md in this package for env vars and how to run.
"""

from .modules import MedicolegalRetrieveExtract, PipelineResult
from .retrieval import FixturePageStore, PageRecord, PageStore, SearchHit

__all__ = [
    "FixturePageStore",
    "MedicolegalRetrieveExtract",
    "PageRecord",
    "PageStore",
    "PipelineResult",
    "SearchHit",
]
