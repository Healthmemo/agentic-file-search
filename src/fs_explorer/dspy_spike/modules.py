"""DSPy modules: search → categorize → extract → question–fact link orchestrator."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import dspy

from .retrieval import (
    PageStore,
    SearchHit,
    SearchQuery,
    hits_to_json,
    pages_to_json,
    parse_deep_review_page_ids,
    parse_queries_json,
)
from .signatures import (
    CategorizeHits,
    ExtractFacts,
    GenerateSearchTerms,
    LinkQuestionsToFacts,
)

DEFAULT_MIN_RELEVANCE = 0.55


@dataclass
class PipelineResult:
    """Structured result of retrieve→categorize→extract→link."""

    queries: list[dict[str, Any]] = field(default_factory=list)
    stop_conditions: dict[str, Any] = field(default_factory=dict)
    search_hits: list[dict[str, Any]] = field(default_factory=list)
    selections: list[dict[str, Any]] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    selected_page_ids: list[str] = field(default_factory=list)
    facts: list[dict[str, Any]] = field(default_factory=list)
    extraction_notes: str = ""
    needs_review: bool = False
    links: list[dict[str, Any]] = field(default_factory=list)
    question_coverage: list[dict[str, Any]] = field(default_factory=list)
    raw: Any | None = None


def _loads_obj(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    text = str(value).strip()
    if not text:
        return default
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return default


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def normalize_facts_for_linking(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ensure each fact has a stable ``fact_id`` for the linker (from local_fact_key)."""
    out: list[dict[str, Any]] = []
    for i, fact in enumerate(facts):
        if not isinstance(fact, dict):
            continue
        item = dict(fact)
        fact_id = str(
            item.get("fact_id") or item.get("local_fact_key") or f"fact_{i + 1}"
        )
        item["fact_id"] = fact_id
        if "local_fact_key" not in item:
            item["local_fact_key"] = fact_id
        out.append(item)
    return out


def normalize_questions(
    questions: list[dict[str, Any]] | None,
    *,
    fallback_key: str,
    fallback_text: str,
) -> list[dict[str, Any]]:
    """Build a questions list; fall back to a single question from pipeline inputs."""
    if questions:
        out: list[dict[str, Any]] = []
        for i, q in enumerate(questions):
            if not isinstance(q, dict):
                continue
            key = str(q.get("question_key") or f"Q{i + 1}")
            text = str(q.get("text") or q.get("question_text") or "").strip()
            if not text:
                continue
            item = dict(q)
            item["question_key"] = key
            item["text"] = text
            out.append(item)
        if out:
            return out
    return [{"question_key": fallback_key, "text": fallback_text, "ordinal": 1}]


class SearchTermGenerator(dspy.Module):
    """Thin wrapper around GenerateSearchTerms."""

    def __init__(self) -> None:
        super().__init__()
        self.predict = dspy.ChainOfThought(GenerateSearchTerms)

    def forward(
        self,
        task_goal: str,
        question_text: str,
        entities_json: str = "{}",
        date_windows_json: str = "[]",
        already_tried_json: str = "[]",
    ) -> dspy.Prediction:
        return self.predict(
            task_goal=task_goal,
            question_text=question_text,
            entities_json=entities_json,
            date_windows_json=date_windows_json,
            already_tried_json=already_tried_json,
        )


class PageCategorizer(dspy.Module):
    """Thin wrapper around CategorizeHits."""

    def __init__(self) -> None:
        super().__init__()
        self.predict = dspy.ChainOfThought(CategorizeHits)

    def forward(
        self,
        task_goal: str,
        target_category: str,
        search_hits_json: str,
        max_deep_review: int = 8,
    ) -> dspy.Prediction:
        return self.predict(
            task_goal=task_goal,
            target_category=target_category,
            max_deep_review=max_deep_review,
            search_hits_json=search_hits_json,
        )


class FactExtractor(dspy.Module):
    """Thin wrapper around ExtractFacts."""

    def __init__(self) -> None:
        super().__init__()
        self.predict = dspy.Predict(ExtractFacts)

    def forward(
        self,
        focus: str,
        section_hint: str,
        allowed_fact_types: str,
        pages_json: str,
    ) -> dspy.Prediction:
        return self.predict(
            focus=focus,
            section_hint=section_hint,
            allowed_fact_types=allowed_fact_types,
            pages_json=pages_json,
        )


class QuestionFactLinker(dspy.Module):
    """Thin wrapper around LinkQuestionsToFacts."""

    def __init__(self) -> None:
        super().__init__()
        self.predict = dspy.ChainOfThought(LinkQuestionsToFacts)

    def forward(
        self,
        case_id: str,
        questions_json: str,
        facts_json: str,
        min_relevance: float = DEFAULT_MIN_RELEVANCE,
    ) -> dspy.Prediction:
        return self.predict(
            case_id=case_id,
            questions_json=questions_json,
            facts_json=facts_json,
            min_relevance=min_relevance,
        )


class MedicolegalRetrieveExtract(dspy.Module):
    """Orchestrator: search → snippets → categorize → extract → question–fact link.

    The LM never receives the full corpus; Python owns hit lists and page budgets.
    """

    def __init__(
        self,
        page_store: PageStore,
        *,
        max_hits: int = 50,
        snippet_chars: int = 500,
        max_deep_review: int = 8,
        min_relevance: float = DEFAULT_MIN_RELEVANCE,
        case_id: str = "case_spike",
    ) -> None:
        super().__init__()
        self.gen_terms = SearchTermGenerator()
        self.categorize = PageCategorizer()
        self.extract = FactExtractor()
        self.link = QuestionFactLinker()
        self.page_store = page_store
        self.max_hits = max_hits
        self.snippet_chars = snippet_chars
        self.max_deep_review = max_deep_review
        self.min_relevance = min_relevance
        self.case_id = case_id

    def forward(
        self,
        task_goal: str,
        question_text: str,
        entities_json: str = "{}",
        date_windows_json: str = "[]",
        already_tried_json: str = "[]",
        target_category: str = "general",
        section_hint: str = "history",
        allowed_fact_types: str = "other",
        focus: str | None = None,
        max_deep_review: int | None = None,
        questions: list[dict[str, Any]] | None = None,
        question_key: str = "Q1",
        min_relevance: float | None = None,
        case_id: str | None = None,
    ) -> PipelineResult:
        budget = self.max_deep_review if max_deep_review is None else max_deep_review
        relevance = self.min_relevance if min_relevance is None else min_relevance
        resolved_case_id = case_id or self.case_id

        terms_pred = self.gen_terms(
            task_goal=task_goal,
            question_text=question_text,
            entities_json=entities_json,
            date_windows_json=date_windows_json,
            already_tried_json=already_tried_json,
        )
        queries_raw = _loads_obj(getattr(terms_pred, "queries_json", None), [])
        stop_conditions = _loads_obj(
            getattr(terms_pred, "stop_conditions_json", None), {}
        )
        queries: list[SearchQuery] = parse_queries_json(queries_raw)

        hits: list[SearchHit] = self.page_store.search_pages(
            queries,
            max_hits=self.max_hits,
            snippet_chars=self.snippet_chars,
        )
        hits_json = hits_to_json(hits)

        cats_pred = self.categorize(
            task_goal=task_goal,
            target_category=target_category,
            search_hits_json=hits_json,
            max_deep_review=budget,
        )
        selections = _loads_obj(getattr(cats_pred, "selections_json", None), [])
        stats = _loads_obj(getattr(cats_pred, "stats_json", None), {})
        selected_ids = parse_deep_review_page_ids(selections)[:budget]

        pages = self.page_store.get_pages(selected_ids)
        pages_json = pages_to_json(pages)

        extract_focus = focus or f"Extract for: {task_goal}"
        facts_pred = self.extract(
            focus=extract_focus,
            section_hint=section_hint,
            allowed_fact_types=allowed_fact_types,
            pages_json=pages_json,
        )
        facts_raw = _loads_obj(getattr(facts_pred, "facts_json", None), [])
        facts = normalize_facts_for_linking(
            facts_raw if isinstance(facts_raw, list) else []
        )
        notes = str(getattr(facts_pred, "extraction_notes", "") or "")
        needs_review = _as_bool(getattr(facts_pred, "needs_review", False))

        questions_list = normalize_questions(
            questions,
            fallback_key=question_key,
            fallback_text=question_text,
        )
        link_pred = self.link(
            case_id=resolved_case_id,
            questions_json=json.dumps(questions_list),
            facts_json=json.dumps(facts),
            min_relevance=relevance,
        )
        links = _loads_obj(getattr(link_pred, "links_json", None), [])
        coverage = _loads_obj(getattr(link_pred, "question_coverage_json", None), [])

        return PipelineResult(
            queries=queries_raw if isinstance(queries_raw, list) else [],
            stop_conditions=stop_conditions
            if isinstance(stop_conditions, dict)
            else {},
            search_hits=json.loads(hits_json),
            selections=selections if isinstance(selections, list) else [],
            stats=stats if isinstance(stats, dict) else {},
            selected_page_ids=selected_ids,
            facts=facts,
            extraction_notes=notes,
            needs_review=needs_review,
            links=links if isinstance(links, list) else [],
            question_coverage=coverage if isinstance(coverage, list) else [],
            raw={
                "terms": terms_pred,
                "categorize": cats_pred,
                "extract": facts_pred,
                "link": link_pred,
            },
        )
