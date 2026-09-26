"""Minimal DSPy spike for medicolegal retrieve → categorize → extract.

Production LM path: Amazon Bedrock. Local/dev default: Ollama.
Page store: FixturePageStore (default) or PostgresPageStore via DSPY_PAGE_STORE.

See README.md in this package for env vars and how to run.
"""

from .modules import MedicolegalRetrieveExtract, PipelineResult
from .postgres_store import PostgresPageStore, resolve_database_url
from .retrieval import FixturePageStore, PageRecord, PageStore, SearchHit
from .store_factory import build_page_store, resolve_page_store_kind

__all__ = [
    "FixturePageStore",
    "MedicolegalRetrieveExtract",
    "PageRecord",
    "PageStore",
    "PipelineResult",
    "PostgresPageStore",
    "SearchHit",
    "build_page_store",
    "resolve_database_url",
    "resolve_page_store_kind",
]
