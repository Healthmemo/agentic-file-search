"""Tiny gold trainset for optimizing QuestionFactLinker (fixture-based)."""

from __future__ import annotations

import json
from typing import Any

import dspy

from ..fixtures import FIXTURE_CASE_ID, FIXTURE_MIN_RELEVANCE, FIXTURE_QUESTIONS

# Major knobs (not user input)
OPTIMIZE_STAGE = "question_fact_linker"
DEFAULT_MAX_LABELED_DEMOS = 2
DEFAULT_MAX_BOOTSTRAPPED_DEMOS = 2


def _example(
    *,
    case_id: str,
    questions: list[dict[str, Any]],
    facts: list[dict[str, Any]],
    links: list[dict[str, Any]],
    coverage: list[dict[str, Any]],
    min_relevance: float = FIXTURE_MIN_RELEVANCE,
) -> dspy.Example:
    return dspy.Example(
        case_id=case_id,
        questions_json=json.dumps(questions),
        facts_json=json.dumps(facts),
        min_relevance=min_relevance,
        links_json=json.dumps(links),
        question_coverage_json=json.dumps(coverage),
    ).with_inputs("case_id", "questions_json", "facts_json", "min_relevance")


def build_question_fact_trainset() -> list[dspy.Example]:
    """Two labeled examples aligned with the spike fixture narrative."""
    questions = [
        {"question_key": q["question_key"], "text": q["text"]}
        for q in FIXTURE_QUESTIONS
    ]

    med_facts = [
        {
            "fact_id": "F_med_1",
            "local_fact_key": "F_med_1",
            "fact_text": "As of 12 Jan 2023 GP review, claimantant was taking tramadol 50 mg BD.",
            "fact_type": "medication",
            "confidence": 0.92,
            "canonical_page_id": "page_a91",
            "page_number": 2,
        }
    ]
    hist_facts = [
        {
            "fact_id": "F_hist_merged_1",
            "local_fact_key": "F_hist_merged_1",
            "fact_text": (
                "Claimant slipped on oil in the warehouse on 14 March 2022 with "
                "immediate low back pain radiating to the left leg."
            ),
            "fact_type": "injury_mechanism",
            "confidence": 0.9,
            "canonical_page_id": "page_ed_2",
            "page_number": 2,
        }
    ]

    example_meds = _example(
        case_id=FIXTURE_CASE_ID,
        questions=questions,
        facts=med_facts,
        links=[
            {
                "question_key": "Q2",
                "fact_id": "F_med_1",
                "relevance": 0.91,
                "role": "support",
                "reason": "States tramadol dose for current medications question",
            }
        ],
        coverage=[
            {
                "question_key": "Q1",
                "status": "uncovered",
                "gap_note": "No injury facts in this example",
            },
            {
                "question_key": "Q2",
                "status": "partial",
                "gap_note": "Dose present; prescriber may still be missing",
            },
        ],
    )

    example_both = _example(
        case_id=FIXTURE_CASE_ID,
        questions=questions,
        facts=med_facts + hist_facts,
        links=[
            {
                "question_key": "Q1",
                "fact_id": "F_hist_merged_1",
                "relevance": 0.95,
                "role": "support",
                "reason": "Merged injury narrative answers history question",
            },
            {
                "question_key": "Q2",
                "fact_id": "F_med_1",
                "relevance": 0.91,
                "role": "support",
                "reason": "Medication fact answers Q2",
            },
        ],
        coverage=[
            {
                "question_key": "Q1",
                "status": "partial",
                "gap_note": "Mechanism covered; investigations not linked",
            },
            {
                "question_key": "Q2",
                "status": "partial",
                "gap_note": "Dose present; prescriber gap remains",
            },
        ],
    )

    return [example_meds, example_both]
