"""DSPy modules: SearchTermGenerator → PageCategorizer → FactExtractor orchestrator."""

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
from .signatures import CategorizeHits, ExtractFacts, GenerateSearchTerms


@dataclass
class PipelineResult:
    """Structured result of one retrieve→categorize→extract forward pass."""

    queries: list[dict[str, Any]] = field(default_factory=list)
    stop_conditions: dict[str, Any] = field(default_factory=dict)
    search_hits: list[dict[str, Any]] = field(default_factory=list)
    selections: list[dict[str, Any]] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    selected_page_ids: list[str] = field(default_factory=list)
    facts: list[dict[str, Any]] = field(default_factory=list)
    extraction_notes: str = ""
    needs_review: bool = False
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


class MedicolegalRetrieveExtract(dspy.Module):
    """Orchestrator: search terms → DB snippets → categorizer → selected pages → facts.

    The LM never receives the full corpus; Python owns hit lists and page budgets.
    """

    def __init__(
        self,
        page_store: PageStore,
        *,
        max_hits: int = 50,
        snippet_chars: int = 500,
        max_deep_review: int = 8,
    ) -> None:
        super().__init__()
        self.gen_terms = SearchTermGenerator()
        self.categorize = PageCategorizer()
        self.extract = FactExtractor()
        self.page_store = page_store
        self.max_hits = max_hits
        self.snippet_chars = snippet_chars
        self.max_deep_review = max_deep_review

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
    ) -> PipelineResult:
        budget = self.max_deep_review if max_deep_review is None else max_deep_review

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
        facts = _loads_obj(getattr(facts_pred, "facts_json", None), [])
        notes = str(getattr(facts_pred, "extraction_notes", "") or "")
        needs_review = _as_bool(getattr(facts_pred, "needs_review", False))

        return PipelineResult(
            queries=queries_raw if isinstance(queries_raw, list) else [],
            stop_conditions=stop_conditions
            if isinstance(stop_conditions, dict)
            else {},
            search_hits=json.loads(hits_json),
            selections=selections if isinstance(selections, list) else [],
            stats=stats if isinstance(stats, dict) else {},
            selected_page_ids=selected_ids,
            facts=facts if isinstance(facts, list) else [],
            extraction_notes=notes,
            needs_review=needs_review,
            raw={
                "terms": terms_pred,
                "categorize": cats_pred,
                "extract": facts_pred,
            },
        )
