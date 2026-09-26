"""Minimal DSPy spike for medicolegal plan → retrieve → extract → link (+ optimize).

Production LM path: Amazon Bedrock. Local/dev default: Ollama.
Page store: FixturePageStore (default) or PostgresPageStore via DSPY_PAGE_STORE.

See README.md in this package for env vars and how to run.
"""

from .modules import (
    CasePlan,
    CasePlanner,
    CrossPageLinker,
    MedicolegalRetrieveExtract,
    PipelineResult,
    QuestionFactLinker,
    SectionMapper,
)
from .optimize import load_optimized_question_fact_linker
from .postgres_store import PostgresPageStore, resolve_database_url
from .retrieval import FixturePageStore, PageRecord, PageStore, SearchHit
from .store_factory import build_page_store, resolve_page_store_kind

__all__ = [
    "CasePlan",
    "CasePlanner",
    "CrossPageLinker",
    "FixturePageStore",
    "MedicolegalRetrieveExtract",
    "PageRecord",
    "PageStore",
    "PipelineResult",
    "PostgresPageStore",
    "QuestionFactLinker",
    "SearchHit",
    "SectionMapper",
    "build_page_store",
    "load_optimized_question_fact_linker",
    "resolve_database_url",
    "resolve_page_store_kind",
]
