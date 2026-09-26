"""Postgres-backed PageStore for the DSPy spike.

Implements the same contract as FixturePageStore:
- ``search_pages`` returns snippet-only hits (FTS / keyword)
- ``get_pages`` returns full text for selected page_ids only

Connection config via ``DATABASE_URL`` or ``DSPY_PG_*`` env vars (os.getenv).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Self

from .retrieval import PageRecord, SearchHit, SearchQuery, make_snippet

DEFAULT_PG_HOST = "localhost"
DEFAULT_PG_PORT = "5432"
DEFAULT_PG_USER = "fs_explorer"
DEFAULT_PG_PASSWORD = "devpassword"
DEFAULT_PG_DATABASE = "fs_explorer"
DEFAULT_CORPUS_ID = "corpus_dspy_spike"
DEFAULT_CORPUS_ROOT = "dspy_spike_fixture"

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


class PostgresConfigError(ValueError):
    """Raised when Postgres connection settings are invalid."""


def resolve_database_url(explicit: str | None = None) -> str:
    """Build a Postgres DSN from ``DATABASE_URL`` or ``DSPY_PG_*`` pieces."""
    if explicit and explicit.strip():
        return explicit.strip()
    env_url = os.getenv("DATABASE_URL")
    if env_url and env_url.strip():
        return env_url.strip()

    host = os.getenv("DSPY_PG_HOST") or DEFAULT_PG_HOST
    port = os.getenv("DSPY_PG_PORT") or DEFAULT_PG_PORT
    user = os.getenv("DSPY_PG_USER") or DEFAULT_PG_USER
    password = os.getenv("DSPY_PG_PASSWORD") or DEFAULT_PG_PASSWORD
    database = os.getenv("DSPY_PG_DATABASE") or DEFAULT_PG_DATABASE
    return f"postgresql://{user}:{password}@{host}:{port}/{database}"


def _require_psycopg() -> Any:
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError as exc:  # pragma: no cover
        raise PostgresConfigError(
            "psycopg is not installed. Install with: uv pip install -e '.[postgres]' "
            "or uv pip install 'psycopg[binary]'"
        ) from exc
    return psycopg, dict_row


def _terms_from_query(text: str) -> list[str]:
    return [t for t in re.split(r"\W+", text.lower()) if len(t) > 2]


class PostgresPageStore:
    """Keyword/FTS page search against the spike Postgres schema."""

    def __init__(self, conninfo: str | None = None, *, autocommit: bool = True) -> None:
        psycopg, dict_row = _require_psycopg()
        self._psycopg = psycopg
        self._dict_row = dict_row
        self.conninfo = resolve_database_url(conninfo)
        self._autocommit = autocommit
        self._conn: Any | None = None

    @classmethod
    def from_env(cls) -> PostgresPageStore:
        return cls(resolve_database_url())

    def connect(self) -> Any:
        if self._conn is None or self._conn.closed:
            self._conn = self._psycopg.connect(
                self.conninfo, row_factory=self._dict_row, autocommit=self._autocommit
            )
        return self._conn

    def close(self) -> None:
        if self._conn is not None and not self._conn.closed:
            self._conn.close()
        self._conn = None

    def __enter__(self) -> Self:
        self.connect()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def init_schema(self) -> None:
        """Apply ``schema.sql`` (CREATE IF NOT EXISTS)."""
        sql = SCHEMA_PATH.read_text(encoding="utf-8")
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute(sql)

    def upsert_pages(
        self,
        pages: Sequence[PageRecord],
        *,
        corpus_id: str = DEFAULT_CORPUS_ID,
        corpus_root: str = DEFAULT_CORPUS_ROOT,
    ) -> int:
        """Upsert documents + pages for spike seeding / tests."""
        if not pages:
            return 0
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO corpora (id, root_path)
                VALUES (%s, %s)
                ON CONFLICT (id) DO NOTHING
                """,
                (corpus_id, corpus_root),
            )
            for page in pages:
                text_sha = hashlib.sha256(page.text.encode("utf-8")).hexdigest()
                cur.execute(
                    """
                    INSERT INTO documents (
                      id, corpus_id, relative_path, absolute_path,
                      content_sha256, page_count, extractor, is_deleted
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, FALSE)
                    ON CONFLICT (id) DO UPDATE SET
                      relative_path = EXCLUDED.relative_path,
                      is_deleted = FALSE
                    """,
                    (
                        page.doc_id,
                        corpus_id,
                        page.path,
                        page.path,
                        text_sha,
                        page.page_number,
                        "dspy_spike_fixture",
                    ),
                )
                cur.execute(
                    """
                    INSERT INTO pages (
                      id, doc_id, corpus_id, page_number, text, text_sha256,
                      char_count, category_hints, page_date_hint
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                      text = EXCLUDED.text,
                      text_sha256 = EXCLUDED.text_sha256,
                      char_count = EXCLUDED.char_count,
                      category_hints = EXCLUDED.category_hints,
                      page_date_hint = EXCLUDED.page_date_hint
                    """,
                    (
                        page.page_id,
                        page.doc_id,
                        corpus_id,
                        page.page_number,
                        page.text,
                        text_sha,
                        len(page.text),
                        list(page.category_hints),
                        page.page_date_hint,
                    ),
                )
        return len(pages)

    def search_pages(
        self,
        queries: list[SearchQuery],
        *,
        max_hits: int = 50,
        snippet_chars: int = 500,
    ) -> list[SearchHit]:
        if not queries:
            return []

        scored: dict[str, SearchHit] = {}
        conn = self.connect()
        for query in sorted(queries, key=lambda q: q.priority):
            terms = _terms_from_query(query.text)
            if not terms and not query.text.strip():
                continue
            fts_query = " ".join(terms) if terms else query.text.strip()
            limit = min(max(1, query.limit_hint), max_hits)
            exclude = [
                str(x).lower()
                for x in (query.filters.get("exclude_keywords") or [])
                if str(x).strip()
            ]

            sql = """
                SELECT
                  p.id AS page_id,
                  p.doc_id,
                  d.relative_path AS path,
                  p.page_number,
                  p.text,
                  p.category_hints,
                  COALESCE(
                    ts_rank(
                      to_tsvector('english', p.text),
                      plainto_tsquery('english', %s)
                    ),
                    0
                  ) AS fts_score
                FROM pages p
                JOIN documents d ON d.id = p.doc_id
                WHERE d.is_deleted = FALSE
                  AND (
                    to_tsvector('english', p.text) @@ plainto_tsquery('english', %s)
                    OR p.text ILIKE %s
                  )
                ORDER BY fts_score DESC
                LIMIT %s
            """
            like_pat = f"%{terms[0]}%" if terms else f"%{query.text.strip()}%"
            with conn.cursor() as cur:
                cur.execute(sql, (fts_query, fts_query, like_pat, limit))
                rows = cur.fetchall()

            for row in rows:
                text = str(row["text"] or "")
                hay = text.lower()
                if exclude and any(x in hay for x in exclude):
                    continue
                term_hits = sum(1 for t in terms if t in hay) if terms else 1
                denom = float(len(terms) or 1)
                category_hints = tuple(row.get("category_hints") or ())
                category_boost = (
                    0.5 if query.category and query.category in category_hints else 0.0
                )
                score = (
                    float(row["fts_score"] or 0) + (term_hits / denom) + category_boost
                )
                snippet = make_snippet(
                    text, terms or [query.text.lower()], snippet_chars
                )
                # Never pass full page body through SearchHit.snippet
                if len(snippet) > snippet_chars + 2:
                    snippet = snippet[: snippet_chars + 2]
                page_id = str(row["page_id"])
                existing = scored.get(page_id)
                if existing is None or score > existing.score:
                    scored[page_id] = SearchHit(
                        page_id=page_id,
                        doc_id=str(row["doc_id"]),
                        path=str(row["path"]),
                        page_number=int(row["page_number"]),
                        score=round(score, 4),
                        snippet=snippet,
                        matched_query_id=query.query_id,
                        category=query.category,
                    )

        hits = sorted(scored.values(), key=lambda h: h.score, reverse=True)
        return hits[:max_hits]

    def get_pages(self, page_ids: list[str]) -> list[PageRecord]:
        if not page_ids:
            return []
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                  p.id AS page_id,
                  p.doc_id,
                  d.relative_path AS path,
                  p.page_number,
                  p.text,
                  p.category_hints,
                  p.page_date_hint
                FROM pages p
                JOIN documents d ON d.id = p.doc_id
                WHERE p.id = ANY(%s)
                  AND d.is_deleted = FALSE
                """,
                (list(page_ids),),
            )
            rows = {str(r["page_id"]): r for r in cur.fetchall()}

        out: list[PageRecord] = []
        for page_id in page_ids:
            row = rows.get(page_id)
            if row is None:
                continue
            out.append(
                PageRecord(
                    page_id=page_id,
                    doc_id=str(row["doc_id"]),
                    path=str(row["path"]),
                    page_number=int(row["page_number"]),
                    text=str(row["text"] or ""),
                    category_hints=tuple(row.get("category_hints") or ()),
                    page_date_hint=row.get("page_date_hint"),
                )
            )
        return out


def _cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DSPy spike Postgres helpers")
    parser.add_argument(
        "--init-schema",
        action="store_true",
        help="Apply schema.sql to the configured database",
    )
    parser.add_argument(
        "--seed-fixture",
        action="store_true",
        help="Upsert the in-repo fixture pages into Postgres",
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="Override DATABASE_URL / DSPY_PG_*",
    )
    args = parser.parse_args(argv)
    if not args.init_schema and not args.seed_fixture:
        parser.error("specify --init-schema and/or --seed-fixture")

    try:
        store = PostgresPageStore(args.database_url)
        if args.init_schema:
            store.init_schema()
            print("schema applied")
        if args.seed_fixture:
            from .fixtures import build_fixture_pages

            n = store.upsert_pages(build_fixture_pages())
            print(f"seeded_pages={n}")
        store.close()
    except (OSError, PostgresConfigError, ValueError) as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - surface driver/connect failures in CLI
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
