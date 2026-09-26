"""Select FixturePageStore vs PostgresPageStore from env or CLI flag."""

from __future__ import annotations

import os
from typing import Literal

from .fixtures import build_fixture_store
from .retrieval import FixturePageStore, PageStore

StoreKind = Literal["fixture", "postgres"]

DEFAULT_PAGE_STORE: StoreKind = "fixture"


class PageStoreConfigError(ValueError):
    """Invalid page-store selection."""


def resolve_page_store_kind(explicit: str | None = None) -> StoreKind:
    raw = (
        (explicit or os.getenv("DSPY_PAGE_STORE") or DEFAULT_PAGE_STORE).strip().lower()
    )
    if raw not in ("fixture", "postgres"):
        raise PageStoreConfigError(
            f"DSPY_PAGE_STORE must be 'fixture' or 'postgres', got {raw!r}"
        )
    return raw  # type: ignore[return-value]


def build_page_store(
    kind: str | None = None,
    *,
    database_url: str | None = None,
) -> PageStore:
    """Build the configured page store (default: in-memory fixture)."""
    resolved = resolve_page_store_kind(kind)
    if resolved == "fixture":
        store: FixturePageStore = build_fixture_store()
        return store

    from .postgres_store import PostgresPageStore

    return PostgresPageStore(database_url)
