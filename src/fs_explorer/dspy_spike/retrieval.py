"""Page-store protocol and in-memory fixture retrieval for the DSPy spike.

Hard rule: search returns snippets only; full page text is fetched only for
selected DEEP_REVIEW page_ids. Never dump the full corpus into the LM.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class PageRecord:
    """Full page body available only after categorizer selection."""

    page_id: str
    doc_id: str
    path: str
    page_number: int
    text: str
    category_hints: tuple[str, ...] = ()
    page_date_hint: str | None = None


@dataclass(frozen=True)
class SearchHit:
    """Snippet-only search hit passed to the categorizer."""

    page_id: str
    doc_id: str
    path: str
    page_number: int
    score: float
    snippet: str
    matched_query_id: str
    category: str


@dataclass
class SearchQuery:
    """Normalized query emitted by SearchTermGenerator (after JSON parse)."""

    query_id: str
    text: str
    mode: str = "keyword"
    category: str = "general"
    priority: int = 1
    filters: dict[str, Any] = field(default_factory=dict)
    limit_hint: int = 25


@runtime_checkable
class PageStore(Protocol):
    """DB adapter contract — Postgres later; fixture store for the spike."""

    def search_pages(
        self,
        queries: list[SearchQuery],
        *,
        max_hits: int = 50,
        snippet_chars: int = 500,
    ) -> list[SearchHit]:
        """Return ranked snippets; never full corpus text."""

    def get_pages(self, page_ids: list[str]) -> list[PageRecord]:
        """Return full page bodies for selected ids only."""


def parse_queries_json(queries_json: str | list[dict[str, Any]]) -> list[SearchQuery]:
    """Parse SearchTermGenerator output into SearchQuery objects."""
    if isinstance(queries_json, str):
        raw = json.loads(queries_json) if queries_json.strip() else []
    else:
        raw = queries_json
    if not isinstance(raw, list):
        raise TypeError("queries_json must be a JSON list")

    out: list[SearchQuery] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        out.append(
            SearchQuery(
                query_id=str(item.get("query_id") or f"q{i + 1}"),
                text=text,
                mode=str(item.get("mode") or "keyword"),
                category=str(item.get("category") or "general"),
                priority=int(item.get("priority") or i + 1),
                filters=dict(item.get("filters") or {}),
                limit_hint=int(item.get("limit_hint") or 25),
            )
        )
    return out


def parse_deep_review_page_ids(
    selections_json: str | list[dict[str, Any]],
) -> list[str]:
    """Extract ordered DEEP_REVIEW page_ids from categorizer output."""
    if isinstance(selections_json, str):
        raw = json.loads(selections_json) if selections_json.strip() else []
    else:
        raw = selections_json
    if not isinstance(raw, list):
        raise TypeError("selections_json must be a JSON list")

    ranked: list[tuple[int, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        if str(item.get("bucket") or "").upper() != "DEEP_REVIEW":
            continue
        page_id = str(item.get("page_id") or "").strip()
        if not page_id:
            continue
        rank = int(item.get("rank") or 999)
        ranked.append((rank, page_id))
    ranked.sort(key=lambda pair: pair[0])
    # Preserve order, drop duplicates
    seen: set[str] = set()
    ordered: list[str] = []
    for _, page_id in ranked:
        if page_id not in seen:
            seen.add(page_id)
            ordered.append(page_id)
    return ordered


def hits_to_json(hits: list[SearchHit]) -> str:
    return json.dumps(
        [
            {
                "page_id": h.page_id,
                "doc_id": h.doc_id,
                "path": h.path,
                "page_number": h.page_number,
                "score": h.score,
                "snippet": h.snippet,
                "matched_query_id": h.matched_query_id,
                "category": h.category,
            }
            for h in hits
        ]
    )


def pages_to_json(pages: list[PageRecord]) -> str:
    return json.dumps(
        [
            {
                "page_id": p.page_id,
                "doc_id": p.doc_id,
                "path": p.path,
                "page_number": p.page_number,
                "text": p.text,
                "page_date_hint": p.page_date_hint,
            }
            for p in pages
        ]
    )


class FixturePageStore:
    """In-memory keyword page store for offline spike / tests."""

    def __init__(self, pages: list[PageRecord] | None = None) -> None:
        self._pages: dict[str, PageRecord] = {p.page_id: p for p in (pages or [])}

    def add_page(self, page: PageRecord) -> None:
        self._pages[page.page_id] = page

    def all_page_ids(self) -> list[str]:
        return list(self._pages.keys())

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
        for query in sorted(queries, key=lambda q: q.priority):
            terms = [t for t in re.split(r"\W+", query.text.lower()) if len(t) > 2]
            if not terms:
                continue
            for page in self._pages.values():
                hay = page.text.lower()
                matches = sum(1 for t in terms if t in hay)
                if matches == 0:
                    continue
                # Soft boost if query category matches page hints
                category_boost = (
                    0.5
                    if query.category and query.category in page.category_hints
                    else 0.0
                )
                score = float(matches) / float(len(terms)) + category_boost
                exclude = [
                    str(x).lower()
                    for x in (query.filters.get("exclude_keywords") or [])
                ]
                if exclude and any(x and x in hay for x in exclude):
                    score *= 0.25

                snippet = _make_snippet(page.text, terms, snippet_chars)
                existing = scored.get(page.page_id)
                if existing is None or score > existing.score:
                    scored[page.page_id] = SearchHit(
                        page_id=page.page_id,
                        doc_id=page.doc_id,
                        path=page.path,
                        page_number=page.page_number,
                        score=round(score, 4),
                        snippet=snippet,
                        matched_query_id=query.query_id,
                        category=query.category,
                    )

        hits = sorted(scored.values(), key=lambda h: h.score, reverse=True)
        return hits[:max_hits]

    def get_pages(self, page_ids: list[str]) -> list[PageRecord]:
        out: list[PageRecord] = []
        for page_id in page_ids:
            page = self._pages.get(page_id)
            if page is not None:
                out.append(page)
        return out


def _make_snippet(text: str, terms: list[str], max_chars: int) -> str:
    lower = text.lower()
    pos = -1
    for term in terms:
        pos = lower.find(term)
        if pos >= 0:
            break
    if pos < 0:
        return text[:max_chars]
    start = max(0, pos - max_chars // 4)
    end = min(len(text), start + max_chars)
    snippet = text[start:end].strip()
    if start > 0:
        snippet = "…" + snippet
    if end < len(text):
        snippet = snippet + "…"
    return snippet
