"""DSPy signatures for the medicolegal plan → retrieve → extract → link spike.

Field shapes are a thin subset of docs/medicolegal-agent-prompts.md.
JSON strings are used for nested lists so adapters parse reliably in a spike.
"""

from __future__ import annotations

import dspy


class PlanCase(dspy.Signature):
    """Plan retrieval/analysis from insurer letter, brief, and required questions.

    Do NOT invent medical-record facts (pages are not provided). Normalize every
    supplied question; extract entities and date windows for retrieval.
    """

    case_id: str = dspy.InputField()
    run_id: str = dspy.InputField()
    today_iso: str = dspy.InputField(desc="Today's date YYYY-MM-DD")
    letter: str = dspy.InputField(desc="Insurer letter text")
    brief: str = dspy.InputField(desc="Case brief / instructions")
    questions_raw: str = dspy.InputField(
        desc="Required questions as JSON array or numbered list text"
    )
    entities_json: str = dspy.OutputField(
        desc=(
            "JSON object: claimant_names, dob, employers, insurers, providers, "
            "facilities, body_parts, other"
        )
    )
    date_windows_json: str = dspy.OutputField(
        desc="JSON list of {label, start, end, precision}"
    )
    constraints_json: str = dspy.OutputField(
        desc="JSON list of scope/constraint strings from letter/brief"
    )
    normalized_questions_json: str = dspy.OutputField(
        desc=(
            "JSON list of {question_key, ordinal, text, source, retrieval_intent, "
            "likely_sections, clarity, clarification_note}"
        )
    )
    section_priorities_json: str = dspy.OutputField(
        desc="JSON list of {section, priority (high|medium|low), reason}"
    )
    initial_tasks_json: str = dspy.OutputField(
        desc=(
            "JSON list of {task_key, task_type, priority, parent_question_key, goal} "
            "(obtain_pages|extract_facts|link_question|tag_section|needs_human)"
        )
    )
    planner_notes: str = dspy.OutputField()


class GenerateSearchTerms(dspy.Signature):
    """Invent high-recall then high-precision DB search queries. Do not read pages."""

    task_goal: str = dspy.InputField(desc="What pages or evidence to obtain")
    question_text: str = dspy.InputField(desc="Parent insurer question text")
    entities_json: str = dspy.InputField(desc="JSON object of entities for retrieval")
    date_windows_json: str = dspy.InputField(desc="JSON list of date windows")
    already_tried_json: str = dspy.InputField(
        desc="JSON list of already-tried query texts; avoid duplicates"
    )
    queries_json: str = dspy.OutputField(
        desc=(
            "JSON list of queries, each with query_id, text, mode "
            "(keyword|fts|vector|hybrid), category, priority (int), "
            "optional filters object"
        )
    )
    stop_conditions_json: str = dspy.OutputField(
        desc="JSON object with max_pages_to_collect and min_distinct_docs"
    )


class CategorizeHits(dspy.Signature):
    """Bucket search-hit snippets into DEEP_REVIEW | MAYBE | SKIP under a page budget."""

    task_goal: str = dspy.InputField()
    target_category: str = dspy.InputField(
        desc="Target retrieval category, e.g. medications"
    )
    max_deep_review: int = dspy.InputField(desc="Hard budget for DEEP_REVIEW pages")
    search_hits_json: str = dspy.InputField(
        desc="JSON array of hits with page_id, snippet, score, category (snippets only)"
    )
    selections_json: str = dspy.OutputField(
        desc=("JSON list of {page_id, bucket (DEEP_REVIEW|MAYBE|SKIP), rank, reason}")
    )
    stats_json: str = dspy.OutputField(
        desc="JSON object: deep_review_count, maybe_count, skip_count, coverage_risk"
    )


class ExtractFacts(dspy.Signature):
    """Extract atomic citable facts from selected pages only; every fact needs page_id + quote."""

    focus: str = dspy.InputField(desc="Extraction focus for this call")
    section_hint: str = dspy.InputField(
        desc="Report section: history|past_history|social|medications|treatment|employment"
    )
    allowed_fact_types: str = dspy.InputField(
        desc="Comma-separated or JSON list of allowed fact_type values"
    )
    pages_json: str = dspy.InputField(
        desc="JSON array of selected pages: page_id, doc_id, page_number, text"
    )
    facts_json: str = dspy.OutputField(
        desc=(
            "JSON list of facts with local_fact_key, fact_text, fact_type, confidence, "
            "canonical_page_id, page_number, evidence_quote, needs_cross_page"
        )
    )
    extraction_notes: str = dspy.OutputField()
    needs_review: bool = dspy.OutputField()


class LinkCrossPageFacts(dspy.Signature):
    """Merge facts that span multiple pages into one canonical fact representation.

    House rule: prefer the page where the claim is most complete; if equal, lowest
    page_number in the same document. Do not merge distinct claims.
    """

    case_id: str = dspy.InputField()
    task_id: str = dspy.InputField(desc="Orchestrator task id for this merge pass")
    house_rule_canonical: str = dspy.InputField(
        desc="Rule for choosing canonical_page_id among spanning pages"
    )
    candidate_facts_json: str = dspy.InputField(
        desc="JSON list of extracted facts (may set needs_cross_page=true)"
    )
    pages_json: str = dspy.InputField(
        desc="JSON list of only pages referenced by candidate facts"
    )
    merged_facts_json: str = dspy.OutputField(
        desc=(
            "JSON list of merged facts with local_fact_key, fact_text, fact_type, "
            "canonical_page_id, supporting_page_ids, confidence, "
            "source_local_fact_keys, optional evidence_quotes"
        )
    )
    unmerged_fact_keys_json: str = dspy.OutputField(
        desc="JSON list of local_fact_key values left unmerged"
    )
    page_link_requests_json: str = dspy.OutputField(
        desc=(
            "JSON list of {local_fact_key, page_ids, reason} for orchestrator "
            "page-link API (may be empty)"
        )
    )


class MapFactSections(dspy.Signature):
    """Map facts into report sections for drafting support.

    Sections: history | past_history | social | medications | treatment | employment.
    One primary_section required per fact; secondary_sections optional.
    """

    case_id: str = dspy.InputField()
    index_event_date: str = dspy.InputField(
        desc="Index event date (ISO) to distinguish history vs past_history"
    )
    facts_json: str = dspy.InputField(
        desc=(
            "JSON list of post-merge facts with fact_id, fact_text, fact_type, "
            "confidence, canonical_page_id, optional event_date"
        )
    )
    mappings_json: str = dspy.OutputField(
        desc=(
            "JSON list of {fact_id, primary_section, secondary_sections, "
            "confidence, reason}"
        )
    )
    section_summaries_json: str = dspy.OutputField(
        desc=(
            "JSON list of {section, fact_count, coverage "
            "(empty|sparse|adequate|rich), note} for all six sections"
        )
    )


class LinkQuestionsToFacts(dspy.Signature):
    """Link extracted facts to questions with relevance, role, and coverage gaps.

    Do not invent fact_ids. Prefer direct evidence. Roles: support|context|conflict.
    """

    case_id: str = dspy.InputField()
    questions_json: str = dspy.InputField(
        desc="JSON list of {question_key, text, ...} — only these questions"
    )
    facts_json: str = dspy.InputField(
        desc=(
            "JSON list of facts with fact_id (or local_fact_key), fact_text, fact_type, "
            "confidence, canonical_page_id, page_number"
        )
    )
    min_relevance: float = dspy.InputField(
        desc="Minimum relevance 0–1 to include a link (default 0.55)"
    )
    links_json: str = dspy.OutputField(
        desc=(
            "JSON list of {question_key, fact_id, relevance, role "
            "(support|context|conflict), reason}"
        )
    )
    question_coverage_json: str = dspy.OutputField(
        desc=(
            "JSON list of {question_key, status (covered|partial|uncovered), gap_note}"
        )
    )
