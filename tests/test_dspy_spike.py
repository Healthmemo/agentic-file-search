"""Tests for the DSPy medicolegal spike (mocked LM — no live Ollama/Bedrock)."""

from __future__ import annotations

import json
from typing import Any

import pytest

dspy = pytest.importorskip("dspy")

from fs_explorer.dspy_spike.fixtures import (
    FIXTURE_ALLOWED_FACT_TYPES,
    FIXTURE_CASE_ID,
    FIXTURE_DATE_WINDOWS,
    FIXTURE_ENTITIES,
    FIXTURE_QUESTION_KEY,
    FIXTURE_QUESTION_TEXT,
    FIXTURE_QUESTIONS,
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
from fs_explorer.dspy_spike.modules import (
    MedicolegalRetrieveExtract,
    apply_cross_page_merges,
    normalize_facts_for_linking,
    normalize_questions,
)
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
                    },
                    {
                        "query_id": "q_hist_1",
                        "text": "slipped oil warehouse injury lumbar",
                        "mode": "keyword",
                        "category": "history",
                        "priority": 2,
                        "filters": {},
                    },
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
            snippet = hit["snippet"].lower()
            bucket = "SKIP"
            if any(
                tok in snippet
                for tok in (
                    "tramadol",
                    "medications",
                    "slipped",
                    "warehouse",
                    "lbp",
                    "oil",
                )
            ):
                bucket = "DEEP_REVIEW"
            if "invoice" in snippet or "payment" in snippet:
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
        assert len(pages) <= 8
        facts = []
        med_i = 0
        for page in pages:
            text = page["text"].lower()
            if "tramadol" in text:
                med_i += 1
                facts.append(
                    {
                        "local_fact_key": f"F_med_{med_i}",
                        "fact_text": "Claimant was taking tramadol 50 mg.",
                        "fact_type": "medication",
                        "confidence": 0.9,
                        "canonical_page_id": page["page_id"],
                        "page_number": page["page_number"],
                        "evidence_quote": "tramadol 50mg",
                        "needs_cross_page": False,
                    }
                )
            if page["page_id"] == "page_ed_1" and "slipped" in text:
                facts.append(
                    {
                        "local_fact_key": "F_hist_1",
                        "fact_text": "Claimant slipped on oil in the warehouse.",
                        "fact_type": "injury_mechanism",
                        "confidence": 0.85,
                        "canonical_page_id": "page_ed_1",
                        "page_number": 1,
                        "evidence_quote": "Slipped on oil in warehouse today",
                        "needs_cross_page": True,
                    }
                )
            if page["page_id"] == "page_ed_2" and (
                "lbp" in text or "radiating" in text
            ):
                facts.append(
                    {
                        "local_fact_key": "F_hist_2",
                        "fact_text": "Immediate low back pain radiating to the left leg.",
                        "fact_type": "presentation",
                        "confidence": 0.88,
                        "canonical_page_id": "page_ed_2",
                        "page_number": 2,
                        "evidence_quote": "Immediate LBP radiating to left leg",
                        "needs_cross_page": True,
                    }
                )
        return dspy.Prediction(
            facts_json=json.dumps(facts),
            extraction_notes="mocked extraction with cross-page candidates",
            needs_review=False,
        )


class _FakeCrossPageLinker:
    def __call__(self, **kwargs: Any) -> dspy.Prediction:
        candidates = json.loads(kwargs["candidate_facts_json"])
        pages = json.loads(kwargs["pages_json"])
        page_ids = {str(p["page_id"]) for p in pages}
        # Cross-page stage must only see pages referenced by candidates
        for fact in candidates:
            pid = str(fact.get("canonical_page_id") or "")
            if pid:
                assert pid in page_ids

        keys = {str(f.get("local_fact_key") or f.get("fact_id")) for f in candidates}
        span_keys = {"F_hist_1", "F_hist_2"}
        merged = []
        unmerged = []
        page_links = []
        if span_keys.issubset(keys):
            merged.append(
                {
                    "local_fact_key": "F_hist_merged_1",
                    "fact_text": (
                        "Claimant reported slipping on oil in the warehouse on "
                        "14 March 2022, with immediate low back pain radiating "
                        "to the left leg."
                    ),
                    "fact_type": "injury_mechanism",
                    "canonical_page_id": "page_ed_2",
                    "supporting_page_ids": ["page_ed_1"],
                    "confidence": 0.9,
                    "event_date": "2022-03-14",
                    "evidence_quotes": [
                        {
                            "page_id": "page_ed_1",
                            "quote": "Slipped on oil in warehouse today",
                        },
                        {
                            "page_id": "page_ed_2",
                            "quote": "Immediate LBP radiating to left leg",
                        },
                    ],
                    "source_local_fact_keys": ["F_hist_1", "F_hist_2"],
                }
            )
            page_links.append(
                {
                    "local_fact_key": "F_hist_merged_1",
                    "page_ids": ["page_ed_1", "page_ed_2"],
                    "reason": "Continuation of injury narrative across ED pages",
                }
            )
            unmerged = sorted(keys - span_keys)
        else:
            unmerged = sorted(keys)

        return dspy.Prediction(
            merged_facts_json=json.dumps(merged),
            unmerged_fact_keys_json=json.dumps(unmerged),
            page_link_requests_json=json.dumps(page_links),
        )


class _FakeLinker:
    def __call__(self, **kwargs: Any) -> dspy.Prediction:
        questions = json.loads(kwargs["questions_json"])
        facts = json.loads(kwargs["facts_json"])
        min_rel = float(kwargs["min_relevance"])
        assert isinstance(questions, list) and questions
        assert all("question_key" in q and "text" in q for q in questions)
        fact_ids = {str(f.get("fact_id") or f.get("local_fact_key")) for f in facts}

        links = []
        for fact in facts:
            fact_id = str(fact.get("fact_id") or fact.get("local_fact_key"))
            ftype = fact.get("fact_type")
            if ftype == "medication":
                relevance = 0.91
                if relevance >= min_rel:
                    links.append(
                        {
                            "question_key": "Q2",
                            "fact_id": fact_id,
                            "relevance": relevance,
                            "role": "support",
                            "reason": "Medication dose answers Q2",
                        }
                    )
            if ftype in {"injury_mechanism", "presentation"}:
                relevance = 0.95
                if relevance >= min_rel:
                    links.append(
                        {
                            "question_key": "Q1",
                            "fact_id": fact_id,
                            "relevance": relevance,
                            "role": "support",
                            "reason": "Injury narrative answers Q1",
                        }
                    )
        assert all(link["fact_id"] in fact_ids for link in links)

        coverage = []
        for q in questions:
            key = q["question_key"]
            linked = [lnk for lnk in links if lnk["question_key"] == key]
            if not linked:
                coverage.append(
                    {
                        "question_key": key,
                        "status": "uncovered",
                        "gap_note": "No linked facts in this spike pass",
                    }
                )
            else:
                coverage.append(
                    {
                        "question_key": key,
                        "status": "partial",
                        "gap_note": "Evidence linked; gaps may remain",
                    }
                )
        return dspy.Prediction(
            links_json=json.dumps(links),
            question_coverage_json=json.dumps(coverage),
        )


def test_normalize_facts_and_questions_helpers() -> None:
    facts = normalize_facts_for_linking(
        [{"local_fact_key": "F1", "fact_text": "x"}, {"fact_text": "y"}]
    )
    assert facts[0]["fact_id"] == "F1"
    assert facts[1]["fact_id"] == "fact_2"

    qs = normalize_questions(
        None, fallback_key="Q9", fallback_text="Fallback question?"
    )
    assert qs == [{"question_key": "Q9", "text": "Fallback question?", "ordinal": 1}]


def test_apply_cross_page_merges_prefers_merged_and_unmerged() -> None:
    candidates = [
        {
            "local_fact_key": "F_hist_1",
            "fact_text": "a",
            "canonical_page_id": "page_ed_1",
        },
        {
            "local_fact_key": "F_hist_2",
            "fact_text": "b",
            "canonical_page_id": "page_ed_2",
        },
        {
            "local_fact_key": "F_med_1",
            "fact_text": "med",
            "canonical_page_id": "page_a91",
        },
    ]
    merged = [
        {
            "local_fact_key": "F_hist_merged_1",
            "fact_text": "merged claim",
            "canonical_page_id": "page_ed_2",
            "supporting_page_ids": ["page_ed_1"],
            "source_local_fact_keys": ["F_hist_1", "F_hist_2"],
            "evidence_quotes": [{"page_id": "page_ed_2", "quote": "Immediate LBP"}],
        }
    ]
    out = apply_cross_page_merges(candidates, merged, ["F_med_1"])
    keys = {f["fact_id"] for f in out}
    assert keys == {"F_hist_merged_1", "F_med_1"}
    merged_fact = next(f for f in out if f["fact_id"] == "F_hist_merged_1")
    assert merged_fact["evidence_quote"] == "Immediate LBP"
    assert "F_hist_1" not in keys


def test_pipeline_forward_with_mocked_lm_modules() -> None:
    store = build_fixture_store()
    pipeline = MedicolegalRetrieveExtract(
        store, max_deep_review=5, case_id=FIXTURE_CASE_ID
    )
    pipeline.gen_terms = _FakeTerms()  # type: ignore[method-assign]
    pipeline.categorize = _FakeCategorizer()  # type: ignore[method-assign]
    pipeline.extract = _FakeExtractor()  # type: ignore[method-assign]
    pipeline.cross_page = _FakeCrossPageLinker()  # type: ignore[method-assign]
    pipeline.link = _FakeLinker()  # type: ignore[method-assign]

    result = pipeline(
        task_goal=FIXTURE_TASK_GOAL,
        question_text=FIXTURE_QUESTION_TEXT,
        entities_json=json.dumps(FIXTURE_ENTITIES),
        date_windows_json=json.dumps(FIXTURE_DATE_WINDOWS),
        target_category=FIXTURE_TARGET_CATEGORY,
        section_hint=FIXTURE_SECTION_HINT,
        allowed_fact_types=FIXTURE_ALLOWED_FACT_TYPES,
        max_deep_review=5,
        questions=FIXTURE_QUESTIONS,
        question_key=FIXTURE_QUESTION_KEY,
        case_id=FIXTURE_CASE_ID,
    )

    assert result.queries
    assert result.search_hits
    assert result.selected_page_ids
    assert "page_c11" not in result.selected_page_ids
    assert all(pid.startswith("page_") for pid in result.selected_page_ids)
    assert len(result.selected_page_ids) <= 5

    assert result.merged_facts
    assert result.merged_facts[0]["local_fact_key"] == "F_hist_merged_1"
    assert result.merged_facts[0]["canonical_page_id"] == "page_ed_2"
    assert "page_ed_1" in result.merged_facts[0]["supporting_page_ids"]
    assert "F_hist_1" not in {
        f.get("local_fact_key") for f in result.facts
    } or "F_hist_merged_1" in {f["fact_id"] for f in result.facts}
    assert "F_hist_merged_1" in {f["fact_id"] for f in result.facts}
    assert result.page_link_requests
    assert result.unmerged_fact_keys
    assert any(k.startswith("F_med_") for k in result.unmerged_fact_keys)

    assert result.facts
    assert any(f["fact_type"] == "medication" for f in result.facts)
    assert result.needs_review is False

    assert result.links
    linked_qs = {lnk["question_key"] for lnk in result.links}
    assert "Q2" in linked_qs
    assert "Q1" in linked_qs
    assert all(
        lnk["fact_id"] in {f["fact_id"] for f in result.facts} for lnk in result.links
    )
    statuses = {c["question_key"]: c["status"] for c in result.question_coverage}
    assert statuses.get("Q2") in {"covered", "partial"}
    assert statuses.get("Q1") in {"covered", "partial"}
