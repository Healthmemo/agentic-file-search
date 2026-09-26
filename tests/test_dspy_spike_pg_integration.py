"""Optional live Postgres integration for the DSPy spike PageStore.

Skipped unless ``DSPY_PG_INTEGRATION=1`` and Postgres is reachable.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("psycopg")

from fs_explorer.dspy_spike.fixtures import build_fixture_pages
from fs_explorer.dspy_spike.postgres_store import (
    PostgresConfigError,
    PostgresPageStore,
    resolve_database_url,
)
from fs_explorer.dspy_spike.retrieval import SearchQuery

_RUN = os.getenv("DSPY_PG_INTEGRATION", "").strip() == "1"


def _pg_available() -> bool:
    if not _RUN:
        return False
    try:
        store = PostgresPageStore(resolve_database_url())
        conn = store.connect()
        with conn.cursor() as cur:
            cur.execute("SELECT 1 AS ok")
            row = cur.fetchone()
        store.close()
        return bool(row and (row.get("ok") == 1 or row.get("ok") is True))
    except Exception:  # noqa: BLE001 - any connect/query failure means unavailable
        return False


pytestmark = pytest.mark.skipif(
    not _pg_available(),
    reason="Set DSPY_PG_INTEGRATION=1 and start docker/postgres to run",
)


def test_live_postgres_search_and_get_roundtrip() -> None:
    store = PostgresPageStore(resolve_database_url())
    try:
        store.init_schema()
        n = store.upsert_pages(build_fixture_pages())
        assert n >= 1
        hits = store.search_pages(
            [
                SearchQuery(
                    query_id="q_med",
                    text="tramadol medications prescribed",
                    category="medications",
                    filters={"exclude_keywords": ["invoice"]},
                )
            ],
            max_hits=10,
            snippet_chars=120,
        )
        assert hits
        assert any(h.page_id == "page_a91" for h in hits)
        for hit in hits:
            assert hit.snippet
            assert "TAX INVOICE" not in hit.snippet or hit.page_id != "page_a91"

        pages = store.get_pages([hits[0].page_id])
        assert len(pages) == 1
        assert pages[0].text
    except PostgresConfigError as exc:  # pragma: no cover
        pytest.skip(str(exc))
    finally:
        store.close()
