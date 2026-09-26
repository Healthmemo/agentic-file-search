"""Unit tests for Postgres PageStore wiring (no live Postgres required)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from fs_explorer.dspy_spike.postgres_store import (
    PostgresPageStore,
    resolve_database_url,
)
from fs_explorer.dspy_spike.retrieval import SearchQuery
from fs_explorer.dspy_spike.store_factory import (
    PageStoreConfigError,
    build_page_store,
    resolve_page_store_kind,
)


def test_resolve_database_url_prefers_database_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@db:5432/app")
    monkeypatch.setenv("DSPY_PG_HOST", "ignored")
    assert resolve_database_url() == "postgresql://u:p@db:5432/app"
    assert resolve_database_url("postgresql://x:y@h:1/d") == "postgresql://x:y@h:1/d"


def test_resolve_database_url_from_pieces(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("DSPY_PG_HOST", "pg.local")
    monkeypatch.setenv("DSPY_PG_PORT", "6543")
    monkeypatch.setenv("DSPY_PG_USER", "alice")
    monkeypatch.setenv("DSPY_PG_PASSWORD", "secret")
    monkeypatch.setenv("DSPY_PG_DATABASE", "cases")
    assert resolve_database_url() == "postgresql://alice:secret@pg.local:6543/cases"


def test_resolve_page_store_kind(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DSPY_PAGE_STORE", raising=False)
    assert resolve_page_store_kind() == "fixture"
    assert resolve_page_store_kind("postgres") == "postgres"
    monkeypatch.setenv("DSPY_PAGE_STORE", "POSTGRES")
    assert resolve_page_store_kind() == "postgres"
    with pytest.raises(PageStoreConfigError):
        resolve_page_store_kind("duckdb")


def test_build_page_store_defaults_to_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DSPY_PAGE_STORE", raising=False)
    store = build_page_store()
    assert hasattr(store, "search_pages")
    assert "page_a91" in store.all_page_ids()  # type: ignore[attr-defined]


def test_postgres_search_pages_returns_snippets_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("psycopg")

    full_text = "GP review. Medications: tramadol 50mg BD prescribed by Dr Patel. " + (
        "padding " * 80
    )
    rows = [
        {
            "page_id": "page_a91",
            "doc_id": "doc_gp_2023_01",
            "path": "gp/2023-01-12.pdf",
            "page_number": 2,
            "text": full_text,
            "category_hints": ["medications"],
            "fts_score": 0.4,
        },
        {
            "page_id": "page_c11",
            "doc_id": "doc_invoice",
            "path": "admin/invoice.pdf",
            "page_number": 1,
            "text": "TAX INVOICE account payment tramadol incidental",
            "category_hints": ["admin"],
            "fts_score": 0.1,
        },
    ]

    cursor = MagicMock()
    cursor.__enter__.return_value = cursor
    cursor.__exit__.return_value = None
    cursor.fetchall.return_value = rows

    conn = MagicMock()
    conn.closed = False
    conn.cursor.return_value = cursor

    store = PostgresPageStore("postgresql://u:p@localhost:5432/db")
    monkeypatch.setattr(store, "connect", lambda: conn)

    hits = store.search_pages(
        [
            SearchQuery(
                query_id="q1",
                text="medications tramadol prescribed",
                category="medications",
                filters={"exclude_keywords": ["invoice", "payment"]},
            )
        ],
        max_hits=10,
        snippet_chars=80,
    )

    assert hits
    assert all(h.page_id != "page_c11" for h in hits)
    assert hits[0].page_id == "page_a91"
    assert "tramadol" in hits[0].snippet.lower()
    assert len(hits[0].snippet) <= 90
    # Snippet must not be the full page body
    assert hits[0].snippet != full_text


def test_postgres_get_pages_preserves_request_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("psycopg")

    rows = [
        {
            "page_id": "page_b",
            "doc_id": "doc_b",
            "path": "b.pdf",
            "page_number": 1,
            "text": "bbb",
            "category_hints": [],
            "page_date_hint": None,
        },
        {
            "page_id": "page_a",
            "doc_id": "doc_a",
            "path": "a.pdf",
            "page_number": 2,
            "text": "aaa",
            "category_hints": ["medications"],
            "page_date_hint": "2023-01-01",
        },
    ]
    cursor = MagicMock()
    cursor.__enter__.return_value = cursor
    cursor.__exit__.return_value = None
    cursor.fetchall.return_value = rows
    conn = MagicMock()
    conn.closed = False
    conn.cursor.return_value = cursor

    store = PostgresPageStore("postgresql://u:p@localhost:5432/db")
    monkeypatch.setattr(store, "connect", lambda: conn)

    pages = store.get_pages(["page_a", "missing", "page_b"])
    assert [p.page_id for p in pages] == ["page_a", "page_b"]
    assert pages[0].text == "aaa"


def test_schema_sql_exists_and_mentions_pages() -> None:
    from fs_explorer.dspy_spike.postgres_store import SCHEMA_PATH

    sql = SCHEMA_PATH.read_text(encoding="utf-8")
    assert "CREATE TABLE IF NOT EXISTS pages" in sql
    assert "to_tsvector" in sql
