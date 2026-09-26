"""Tests for the DSPy medicolegal spike (mocked LM — no live Ollama/Bedrock)."""

from __future__ import annotations

import json
from typing import Any

import pytest

dspy = pytest.importorskip("dspy")

from fs_explorer.dspy_spike.fixtures import (
    FIXTURE_ALLOWED_FACT_TYPES,
    FIXTURE_DATE_WINDOWS,
    FIXTURE_ENTITIES,
    FIXTURE_QUESTION_TEXT,
    FIXTURE_SECTION_HINT,
    FIXTURE_TARGET_CATEGORY,
    FIXTURE_TASK_GOAL,
    build_fixture_store,
)
from fs_explorer.dspy_spike.lm import (
    LMConfigError,
    build_lm,
    resolve_provider,
)
from fs_explorer.dspy_spike.modules import MedicolegalRetrieveExtract
from fs_explorer.dspy_spike.retrieval import (
    SearchQuery,
    parse_deep_review_page_ids,
    parse_queries_json,
)


def test_resolve_provider_defaults_to_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DSPY_LM_PROVIDER", raising=False)
    assert resolve_provider() == "ollama"
    assert resolve_provider("bedrock") == "bedrock"
    monkeypatch.setenv("DSPY_LM_PROVIDER", "BEDROCK")
    assert resolve_provider() == "bedrock"


def test_resolve_provider_rejects_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DSPY_LM_PROVIDER", "openai")
    with pytest.raises(LMConfigError):
        resolve_provider()


def test_build_lm_ollama_uses_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DSPY_LM_PROVIDER", "ollama")
    monkeypatch.setenv("DSPY_OLLAMA_MODEL", "llama3.2:1b")
    monkeypatch.setenv("DSPY_OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    lm = build_lm(configure=False)
    assert "ollama" in str(lm.model).lower() or "llama" in str(lm.model).lower()


def test_build_lm_bedrock_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DSPY_LM_PROVIDER", "bedrock")
    monkeypatch.setenv("DSPY_BEDROCK_MODEL", "amazon.nova-lite-v1:0")
    monkeypatch.setenv("DSPY_BEDROCK_REGION", "us-west-2")
    lm = build_lm(configure=False)
    assert str(lm.model).startswith("bedrock/")


def test_fixture_store_search_returns_snippets_not_full_corpus() -> None:
    store = build_fixture_store()
    queries = [
        SearchQuery(
            query_id="q_med_1",
            text="medications tramadol prescribed",
            category="medications",
            priority=1,
            filters={"exclude_keywords": ["invoice"]},
        )
    ]
    hits = store.search_pages(queries, max_hits=10, snippet_chars=120)
    assert hits
    assert any(h.page_id == "page_a91" for h in hits)
    for hit in hits:
        assert len(hit.snippet) <= 150  # ellipsis may add a char or two
        # Snippet must not be the entire fixture corpus concatenated
        assert "TAX INVOICE" not in hit.snippet or hit.page_id == "page_c11"

    pages = store.get_pages(["page_a91"])
    assert len(pages) == 1
    assert "tramadol" in pages[0].text
    assert store.get_pages(["missing"]) == []


def test_parse_queries_and_deep_review_ids() -> None:
    queries = parse_queries_json(
        json.dumps(
            [
                {
                    "query_id": "q1",
                    "text": "medication list",
                    "mode": "keyword",
                    "category": "medications",
                    "priority": 1,
                }
            ]
        )
    )
    assert len(queries) == 1
    assert queries[0].text == "medication list"

    page_ids = parse_deep_review_page_ids(
        [
            {"page_id": "page_b", "bucket": "DEEP_REVIEW", "rank": 2, "reason": "b"},
            {"page_id": "page_a", "bucket": "DEEP_REVIEW", "rank": 1, "reason": "a"},
            {"page_id": "page_c", "bucket": "SKIP", "rank": 99, "reason": "c"},
        ]
    )
    assert page_ids == ["page_a", "page_b"]


class _FakeTerms:
    def __call__(self, **kwargs: Any) -> dspy.Prediction:
        return dspy.Prediction(
            queries_json=json.dumps(
                [
                    {
                        "query_id": "q_med_1",
                        "text": "medications tramadol medication list prescribed",
                        "mode": "keyword",
                        "category": "medications",
                        "priority": 1,
                        "filters": {"exclude_keywords": ["invoice", "account payment"]},
                    }
                ]
            ),
            stop_conditions_json=json.dumps(
                {"max_pages_to_collect": 20, "min_distinct_docs": 1}
            ),
        )


class _FakeCategorizer:
    def __call__(self, **kwargs: Any) -> dspy.Prediction:
        hits = json.loads(kwargs["search_hits_json"])
        assert isinstance(hits, list)
        # Categorizer must only see snippets — no full multi-page dump
        for hit in hits:
            assert "snippet" in hit
            assert "text" not in hit
        selections = []
        for i, hit in enumerate(hits):
            bucket = (
                "DEEP_REVIEW"
                if "tramadol" in hit["snippet"].lower()
                or "medications" in hit["snippet"].lower()
                else "SKIP"
            )
            if (
                "invoice" in hit["snippet"].lower()
                or "payment" in hit["snippet"].lower()
            ):
                bucket = "SKIP"
            selections.append(
                {
                    "page_id": hit["page_id"],
                    "bucket": bucket,
                    "rank": i + 1,
                    "reason": "fixture mock",
                }
            )
        deep = sum(1 for s in selections if s["bucket"] == "DEEP_REVIEW")
        return dspy.Prediction(
            selections_json=json.dumps(selections),
            stats_json=json.dumps(
                {
                    "deep_review_count": deep,
                    "maybe_count": 0,
                    "skip_count": len(selections) - deep,
                    "coverage_risk": "low",
                }
            ),
        )


class _FakeExtractor:
    def __call__(self, **kwargs: Any) -> dspy.Prediction:
        pages = json.loads(kwargs["pages_json"])
        assert pages, "extractor should receive selected pages only"
        # Must not receive the whole fixture corpus blindly
        assert len(pages) <= int(kwargs.get("max_deep_review") or 8) or len(pages) <= 4
        facts = []
        for i, page in enumerate(pages):
            if "tramadol" in page["text"].lower():
                facts.append(
                    {
                        "local_fact_key": f"F_med_{i + 1}",
                        "fact_text": "Claimant was taking tramadol 50 mg.",
                        "fact_type": "medication",
                        "confidence": 0.9,
                        "canonical_page_id": page["page_id"],
                        "page_number": page["page_number"],
                        "evidence_quote": "tramadol 50mg",
                        "needs_cross_page": False,
                    }
                )
        return dspy.Prediction(
            facts_json=json.dumps(facts),
            extraction_notes="mocked extraction",
            needs_review=False,
        )


def test_pipeline_forward_with_mocked_lm_modules() -> None:
    store = build_fixture_store()
    pipeline = MedicolegalRetrieveExtract(store, max_deep_review=3)
    pipeline.gen_terms = _FakeTerms()  # type: ignore[method-assign]
    pipeline.categorize = _FakeCategorizer()  # type: ignore[method-assign]
    pipeline.extract = _FakeExtractor()  # type: ignore[method-assign]

    result = pipeline(
        task_goal=FIXTURE_TASK_GOAL,
        question_text=FIXTURE_QUESTION_TEXT,
        entities_json=json.dumps(FIXTURE_ENTITIES),
        date_windows_json=json.dumps(FIXTURE_DATE_WINDOWS),
        target_category=FIXTURE_TARGET_CATEGORY,
        section_hint=FIXTURE_SECTION_HINT,
        allowed_fact_types=FIXTURE_ALLOWED_FACT_TYPES,
        max_deep_review=3,
    )

    assert result.queries
    assert result.search_hits
    assert result.selected_page_ids
    assert "page_c11" not in result.selected_page_ids
    assert all(pid.startswith("page_") for pid in result.selected_page_ids)
    # Full corpus has 4 pages; extractor must not see all unless all selected
    assert len(result.selected_page_ids) <= 3
    assert result.facts
    assert result.facts[0]["fact_type"] == "medication"
    assert result.facts[0]["canonical_page_id"] in result.selected_page_ids
    assert result.needs_review is False
