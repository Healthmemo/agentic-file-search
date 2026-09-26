"""DSPy modules: plan → search → categorize → extract → cross-page → section → Q–fact.

Orchestrator order:
**plan** (optional, when letter provided) → search terms → categorize → extract →
cross-page merge → **section map** → **question–fact link**.

Section mapping and question linking both consume the same post-merge ``facts``.
Section map runs first so drafting tags are available before question coverage;
question linking does not depend on section tags.
"""

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
    LinkCrossPageFacts,
    LinkQuestionsToFacts,
    MapFactSections,
    PlanCase,
)

DEFAULT_MIN_RELEVANCE = 0.55
DEFAULT_HOUSE_RULE_CANONICAL = (
    "Prefer the page where the claim is most complete; if equal, "
    "prefer the lowest page_number within the same document."
)
DEFAULT_CROSS_PAGE_TASK_ID = "T_cross_page"
DEFAULT_INDEX_EVENT_DATE = ""
REPORT_SECTIONS = (
    "history",
    "past_history",
    "social",
    "medications",
    "treatment",
    "employment",
)


@dataclass
class CasePlan:
    """Structured planner output (prompt-pack case plan, JSON-string fields parsed)."""

    case_id: str = ""
    run_id: str = ""
    entities: dict[str, Any] = field(default_factory=dict)
    date_windows: list[dict[str, Any]] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    normalized_questions: list[dict[str, Any]] = field(default_factory=list)
    section_priorities: list[dict[str, Any]] = field(default_factory=list)
    initial_tasks: list[dict[str, Any]] = field(default_factory=list)
    planner_notes: str = ""


@dataclass
class SearchDrivers:
    """Values derived from a CasePlan that drive GenerateSearchTerms + extract focus."""

    task_goal: str
    question_text: str
    question_key: str
    entities_json: str
    date_windows_json: str
    questions: list[dict[str, Any]]
    target_category: str
    section_hint: str
    allowed_fact_types: str
    index_event_date: str
    focus: str


@dataclass
class PipelineResult:
    """Structured result of plan→retrieve→extract→cross-page→section→question link."""

    case_plan: CasePlan | None = None
    queries: list[dict[str, Any]] = field(default_factory=list)
    stop_conditions: dict[str, Any] = field(default_factory=dict)
    search_hits: list[dict[str, Any]] = field(default_factory=list)
    selections: list[dict[str, Any]] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    selected_page_ids: list[str] = field(default_factory=list)
    facts: list[dict[str, Any]] = field(default_factory=list)
    extraction_notes: str = ""
    needs_review: bool = False
    merged_facts: list[dict[str, Any]] = field(default_factory=list)
    unmerged_fact_keys: list[str] = field(default_factory=list)
    page_link_requests: list[dict[str, Any]] = field(default_factory=list)
    mappings: list[dict[str, Any]] = field(default_factory=list)
    section_summaries: list[dict[str, Any]] = field(default_factory=list)
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


def pages_for_candidate_facts(
    pages_json: str,
    candidate_facts: list[dict[str, Any]],
) -> str:
    """Filter selected pages down to those referenced by candidate facts only."""
    all_pages = _loads_obj(pages_json, [])
    if not isinstance(all_pages, list):
        return "[]"
    needed: set[str] = set()
    for fact in candidate_facts:
        for key in ("canonical_page_id",):
            pid = str(fact.get(key) or "").strip()
            if pid:
                needed.add(pid)
        for pid in fact.get("supporting_page_ids") or []:
            if pid:
                needed.add(str(pid))
    if not needed:
        return pages_json
    filtered = [p for p in all_pages if str(p.get("page_id")) in needed]
    return json.dumps(filtered)


def apply_cross_page_merges(
    candidate_facts: list[dict[str, Any]],
    merged_facts: list[dict[str, Any]],
    unmerged_fact_keys: list[str],
) -> list[dict[str, Any]]:
    """Collapse merge output into the fact list for section map + question link."""
    by_key = {
        str(f.get("local_fact_key") or f.get("fact_id")): f for f in candidate_facts
    }
    out: list[dict[str, Any]] = []

    for merged in merged_facts:
        if not isinstance(merged, dict):
            continue
        item = dict(merged)
        key = str(item.get("local_fact_key") or item.get("fact_id") or "").strip()
        if not key:
            continue
        item["local_fact_key"] = key
        item["fact_id"] = key
        quotes = item.get("evidence_quotes") or []
        if not item.get("evidence_quote") and quotes and isinstance(quotes[0], dict):
            item["evidence_quote"] = str(quotes[0].get("quote") or "")
        out.append(item)

    for key in unmerged_fact_keys:
        k = str(key)
        if k in by_key:
            out.append(by_key[k])

    # Fallback: no merge plan → keep candidates
    if not out:
        return normalize_facts_for_linking(candidate_facts)
    return normalize_facts_for_linking(out)


def empty_section_summaries(note: str = "No facts to map") -> list[dict[str, Any]]:
    """All six report sections with empty coverage (prompt-pack empty-facts behavior)."""
    return [
        {
            "section": section,
            "fact_count": 0,
            "coverage": "empty",
            "note": note,
        }
        for section in REPORT_SECTIONS
    ]


def coverage_label(fact_count: int) -> str:
    if fact_count <= 0:
        return "empty"
    if fact_count == 1:
        return "sparse"
    if fact_count <= 3:
        return "adequate"
    return "rich"


def parse_case_plan(
    pred: Any,
    *,
    case_id: str = "",
    run_id: str = "",
) -> CasePlan:
    """Parse a PlanCase prediction into a CasePlan dataclass."""
    entities = _loads_obj(getattr(pred, "entities_json", None), {})
    date_windows = _loads_obj(getattr(pred, "date_windows_json", None), [])
    constraints = _loads_obj(getattr(pred, "constraints_json", None), [])
    questions = _loads_obj(getattr(pred, "normalized_questions_json", None), [])
    section_priorities = _loads_obj(getattr(pred, "section_priorities_json", None), [])
    initial_tasks = _loads_obj(getattr(pred, "initial_tasks_json", None), [])
    notes = str(getattr(pred, "planner_notes", "") or "")

    constraint_list = (
        [str(c) for c in constraints] if isinstance(constraints, list) else []
    )
    return CasePlan(
        case_id=case_id,
        run_id=run_id,
        entities=entities if isinstance(entities, dict) else {},
        date_windows=date_windows if isinstance(date_windows, list) else [],
        constraints=constraint_list,
        normalized_questions=questions if isinstance(questions, list) else [],
        section_priorities=section_priorities
        if isinstance(section_priorities, list)
        else [],
        initial_tasks=initial_tasks if isinstance(initial_tasks, list) else [],
        planner_notes=notes,
    )


def _index_event_from_windows(date_windows: list[dict[str, Any]]) -> str:
    for window in date_windows:
        if not isinstance(window, dict):
            continue
        label = str(window.get("label") or "").lower()
        if "index" in label:
            start = window.get("start")
            if start:
                return str(start)
    for window in date_windows:
        if isinstance(window, dict) and window.get("start"):
            return str(window["start"])
    return ""


def _section_from_priorities(
    section_priorities: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    question_key: str,
) -> str:
    for q in questions:
        if str(q.get("question_key")) == question_key:
            sections = q.get("likely_sections") or []
            if sections:
                return str(sections[0])
    high = [
        str(p.get("section"))
        for p in section_priorities
        if isinstance(p, dict) and str(p.get("priority")) == "high" and p.get("section")
    ]
    if high:
        return high[0]
    return "history"


def _fact_types_for_section(section: str) -> str:
    mapping = {
        "history": "injury_mechanism,presentation,other",
        "past_history": "past_history,other",
        "social": "social,other",
        "medications": "medication,other",
        "treatment": "treatment,presentation,other",
        "employment": "employment_status,capacity,other",
    }
    return mapping.get(section, "other")


def drivers_from_case_plan(
    plan: CasePlan,
    *,
    fallback_task_goal: str = "",
    fallback_question_text: str = "",
    preferred_question_key: str | None = None,
) -> SearchDrivers:
    """Derive search/extract drivers from planner output (light task skeleton only)."""
    questions = normalize_questions(
        plan.normalized_questions,
        fallback_key="Q1",
        fallback_text=fallback_question_text or "Answer the insurer questions.",
    )
    by_key = {str(q["question_key"]): q for q in questions}

    obtain = [
        t
        for t in plan.initial_tasks
        if isinstance(t, dict) and str(t.get("task_type")) == "obtain_pages"
    ]
    obtain.sort(key=lambda t: (int(t.get("priority") or 99), str(t.get("task_key"))))

    chosen: dict[str, Any] | None = None
    if preferred_question_key:
        for task in obtain:
            if str(task.get("parent_question_key")) == preferred_question_key:
                chosen = task
                break
    if chosen is None and obtain:
        chosen = obtain[0]

    if chosen is not None:
        task_goal = str(chosen.get("goal") or fallback_task_goal).strip()
        q_key = str(chosen.get("parent_question_key") or questions[0]["question_key"])
    else:
        task_goal = fallback_task_goal or (
            plan.planner_notes[:200] if plan.planner_notes else "Obtain relevant pages"
        )
        q_key = preferred_question_key or (
            questions[0]["question_key"] if questions else "Q1"
        )

    q_text = str(by_key.get(q_key, questions[0]).get("text") or fallback_question_text)
    section = _section_from_priorities(plan.section_priorities, questions, str(q_key))
    # Broaden search when multiple obtain_pages exist: join goals for recall
    if len(obtain) > 1:
        goals = [str(t.get("goal") or "").strip() for t in obtain if t.get("goal")]
        if goals:
            task_goal = "; ".join(goals)

    fact_type_parts: list[str] = []
    for q in questions:
        for sec in q.get("likely_sections") or []:
            for part in _fact_types_for_section(str(sec)).split(","):
                if part and part not in fact_type_parts:
                    fact_type_parts.append(part)
    if not fact_type_parts:
        fact_type_parts = _fact_types_for_section(section).split(",")

    return SearchDrivers(
        task_goal=task_goal or fallback_task_goal,
        question_text=q_text,
        question_key=str(q_key),
        entities_json=json.dumps(plan.entities or {}),
        date_windows_json=json.dumps(plan.date_windows or []),
        questions=questions,
        target_category=section,
        section_hint=section,
        allowed_fact_types=",".join(fact_type_parts),
        index_event_date=_index_event_from_windows(plan.date_windows),
        focus=f"Extract for: {task_goal or fallback_task_goal}",
    )


class CasePlanner(dspy.Module):
    """Letter/brief/question planner — front of the medicolegal pipeline."""

    def __init__(self) -> None:
        super().__init__()
        self.predict = dspy.ChainOfThought(PlanCase)

    def forward(
        self,
        case_id: str,
        run_id: str,
        today_iso: str,
        letter: str,
        brief: str = "",
        questions_raw: str = "[]",
    ) -> dspy.Prediction:
        return self.predict(
            case_id=case_id,
            run_id=run_id,
            today_iso=today_iso,
            letter=letter,
            brief=brief,
            questions_raw=questions_raw,
        )


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


class CrossPageLinker(dspy.Module):
    """Thin wrapper around LinkCrossPageFacts."""

    def __init__(self) -> None:
        super().__init__()
        self.predict = dspy.ChainOfThought(LinkCrossPageFacts)

    def forward(
        self,
        case_id: str,
        task_id: str,
        house_rule_canonical: str,
        candidate_facts_json: str,
        pages_json: str,
    ) -> dspy.Prediction:
        return self.predict(
            case_id=case_id,
            task_id=task_id,
            house_rule_canonical=house_rule_canonical,
            candidate_facts_json=candidate_facts_json,
            pages_json=pages_json,
        )


class SectionMapper(dspy.Module):
    """Thin wrapper around MapFactSections."""

    def __init__(self) -> None:
        super().__init__()
        self.predict = dspy.Predict(MapFactSections)

    def forward(
        self,
        case_id: str,
        index_event_date: str,
        facts_json: str,
    ) -> dspy.Prediction:
        return self.predict(
            case_id=case_id,
            index_event_date=index_event_date,
            facts_json=facts_json,
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
    """Orchestrator: plan → search → categorize → extract → cross-page → section → Q–fact.

    When ``letter`` is provided, ``CasePlanner`` runs first and drives search terms /
    questions. Without a letter, callers pass task_goal/entities directly (compat).

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
        run_id: str = "run_spike",
        house_rule_canonical: str = DEFAULT_HOUSE_RULE_CANONICAL,
        cross_page_task_id: str = DEFAULT_CROSS_PAGE_TASK_ID,
        index_event_date: str = DEFAULT_INDEX_EVENT_DATE,
    ) -> None:
        super().__init__()
        self.plan = CasePlanner()
        self.gen_terms = SearchTermGenerator()
        self.categorize = PageCategorizer()
        self.extract = FactExtractor()
        self.cross_page = CrossPageLinker()
        self.section_map = SectionMapper()
        self.link = QuestionFactLinker()
        self.page_store = page_store
        self.max_hits = max_hits
        self.snippet_chars = snippet_chars
        self.max_deep_review = max_deep_review
        self.min_relevance = min_relevance
        self.case_id = case_id
        self.run_id = run_id
        self.house_rule_canonical = house_rule_canonical
        self.cross_page_task_id = cross_page_task_id
        self.index_event_date = index_event_date

    def forward(
        self,
        task_goal: str = "",
        question_text: str = "",
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
        run_id: str | None = None,
        house_rule_canonical: str | None = None,
        cross_page_task_id: str | None = None,
        index_event_date: str | None = None,
        letter: str | None = None,
        brief: str = "",
        questions_raw: str | None = None,
        today_iso: str = "",
        preferred_question_key: str | None = None,
    ) -> PipelineResult:
        budget = self.max_deep_review if max_deep_review is None else max_deep_review
        relevance = self.min_relevance if min_relevance is None else min_relevance
        resolved_case_id = case_id or self.case_id
        resolved_run_id = run_id or self.run_id
        house_rule = house_rule_canonical or self.house_rule_canonical
        task_id = cross_page_task_id or self.cross_page_task_id
        index_date = (
            index_event_date if index_event_date is not None else self.index_event_date
        )

        case_plan: CasePlan | None = None
        plan_pred: Any | None = None
        if letter is not None and str(letter).strip():
            raw_questions = questions_raw
            if raw_questions is None:
                if questions:
                    raw_questions = json.dumps(
                        [
                            {
                                "question_key": q.get("question_key"),
                                "text": q.get("text") or q.get("question_text"),
                            }
                            for q in questions
                            if isinstance(q, dict)
                        ]
                    )
                else:
                    raw_questions = "[]"
            plan_pred = self.plan(
                case_id=resolved_case_id,
                run_id=resolved_run_id,
                today_iso=today_iso or "",
                letter=letter,
                brief=brief or "",
                questions_raw=raw_questions,
            )
            case_plan = parse_case_plan(
                plan_pred, case_id=resolved_case_id, run_id=resolved_run_id
            )
            drivers = drivers_from_case_plan(
                case_plan,
                fallback_task_goal=task_goal,
                fallback_question_text=question_text,
                preferred_question_key=preferred_question_key or question_key,
            )
            task_goal = drivers.task_goal or task_goal
            question_text = drivers.question_text or question_text
            entities_json = drivers.entities_json
            date_windows_json = drivers.date_windows_json
            questions = drivers.questions
            question_key = drivers.question_key
            target_category = drivers.target_category or target_category
            section_hint = drivers.section_hint or section_hint
            allowed_fact_types = drivers.allowed_fact_types or allowed_fact_types
            if drivers.index_event_date:
                index_date = drivers.index_event_date
            if focus is None:
                focus = drivers.focus

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
        candidate_facts = normalize_facts_for_linking(
            facts_raw if isinstance(facts_raw, list) else []
        )
        notes = str(getattr(facts_pred, "extraction_notes", "") or "")
        needs_review = _as_bool(getattr(facts_pred, "needs_review", False))

        cross_pages_json = pages_for_candidate_facts(pages_json, candidate_facts)
        cross_pred = self.cross_page(
            case_id=resolved_case_id,
            task_id=task_id,
            house_rule_canonical=house_rule,
            candidate_facts_json=json.dumps(candidate_facts),
            pages_json=cross_pages_json,
        )
        merged_facts = _loads_obj(getattr(cross_pred, "merged_facts_json", None), [])
        unmerged_keys_raw = _loads_obj(
            getattr(cross_pred, "unmerged_fact_keys_json", None), []
        )
        page_link_requests = _loads_obj(
            getattr(cross_pred, "page_link_requests_json", None), []
        )
        merged_list = merged_facts if isinstance(merged_facts, list) else []
        unmerged_keys = (
            [str(k) for k in unmerged_keys_raw]
            if isinstance(unmerged_keys_raw, list)
            else []
        )
        # If the model omitted unmerged keys and produced no merges, keep all candidates.
        if not merged_list and not unmerged_keys:
            unmerged_keys = [
                str(f.get("local_fact_key") or f.get("fact_id"))
                for f in candidate_facts
            ]

        facts = apply_cross_page_merges(candidate_facts, merged_list, unmerged_keys)

        # Section map then question–fact link — both use the same post-merge facts.
        if facts:
            section_pred = self.section_map(
                case_id=resolved_case_id,
                index_event_date=index_date or "",
                facts_json=json.dumps(facts),
            )
            mappings = _loads_obj(getattr(section_pred, "mappings_json", None), [])
            summaries = _loads_obj(
                getattr(section_pred, "section_summaries_json", None), []
            )
            if not isinstance(mappings, list):
                mappings = []
            if not isinstance(summaries, list) or not summaries:
                summaries = empty_section_summaries()
        else:
            section_pred = None
            mappings = []
            summaries = empty_section_summaries()

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
            case_plan=case_plan,
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
            merged_facts=merged_list,
            unmerged_fact_keys=unmerged_keys,
            page_link_requests=page_link_requests
            if isinstance(page_link_requests, list)
            else [],
            mappings=mappings,
            section_summaries=summaries,
            links=links if isinstance(links, list) else [],
            question_coverage=coverage if isinstance(coverage, list) else [],
            raw={
                "plan": plan_pred,
                "terms": terms_pred,
                "categorize": cats_pred,
                "extract": facts_pred,
                "cross_page": cross_pred,
                "section_map": section_pred,
                "link": link_pred,
            },
        )
