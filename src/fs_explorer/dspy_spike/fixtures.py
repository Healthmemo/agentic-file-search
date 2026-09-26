"""Tiny offline fixture: letter/brief/questions + medical page snippets."""

from __future__ import annotations

import json

from .retrieval import FixturePageStore, PageRecord

# Major spike fixture knobs (not user input)
FIXTURE_CASE_ID = "case_spike_acme_0142"
FIXTURE_RUN_ID = "run_spike_01"
FIXTURE_TODAY_ISO = "2024-06-01"
FIXTURE_QUESTION_KEY = "Q2"
FIXTURE_QUESTION_TEXT = "List current medications and who prescribed them."
FIXTURE_TASK_GOAL = (
    "Retrieve medication lists and prescription records after the index event"
)
FIXTURE_TARGET_CATEGORY = "medications"
FIXTURE_SECTION_HINT = "medications"
FIXTURE_ALLOWED_FACT_TYPES = "medication,injury_mechanism,presentation"
FIXTURE_MIN_RELEVANCE = 0.55
FIXTURE_INDEX_EVENT_DATE = "2022-03-14"
FIXTURE_HOUSE_RULE_CANONICAL = (
    "Prefer the page where the claim is most complete; if equal, "
    "prefer the lowest page_number within the same document."
)
FIXTURE_LETTER = """
Acme Workers Comp
Re: Claim WC-88921 — Jane Marie Doe (DOB 12 June 1984)

Please prepare a medicolegal report addressing only the questions listed below.
Index event: 14 March 2022 slip at Northside Logistics Pty Ltd (warehouse),
lumbar spine / left leg. Treating providers include Dr Patel and City Physio;
attended City General Hospital. Focus on capacity for pre-injury duties.
Do not expand into unrelated causation opinion beyond the evidence.
""".strip()
FIXTURE_BRIEF = """
Claimant: Jane Doe / J. Doe. Employer: Northside Logistics Pty Ltd.
Prioritise ED and GP notes near the index date before imaging.
Answer the supplied question list only.
""".strip()
FIXTURE_QUESTIONS = [
    {
        "question_key": "Q1",
        "ordinal": 1,
        "text": "What is the history of the claimed injury of 14 March 2022?",
        "likely_sections": ["history", "treatment"],
    },
    {
        "question_key": "Q2",
        "ordinal": 2,
        "text": FIXTURE_QUESTION_TEXT,
        "likely_sections": ["medications", "treatment"],
    },
]
FIXTURE_QUESTIONS_RAW = json.dumps(
    [{"question_key": q["question_key"], "text": q["text"]} for q in FIXTURE_QUESTIONS]
)
FIXTURE_ENTITIES = {
    "claimant_names": ["Jane Marie Doe", "J. Doe", "Jane Doe"],
    "dob": "1984-06-12",
    "employers": ["Northside Logistics Pty Ltd"],
    "insurers": ["Acme Workers Comp"],
    "providers": ["Dr Patel", "City Physio"],
    "facilities": ["City General Hospital"],
    "body_parts": ["lumbar spine", "left leg"],
    "other": ["claim number WC-88921"],
}
FIXTURE_DATE_WINDOWS = [
    {
        "label": "index_event",
        "start": "2022-03-14",
        "end": "2022-03-14",
        "precision": "day",
    },
    {
        "label": "post_injury_treatment",
        "start": "2022-03-14",
        "end": None,
        "precision": "day",
    },
    {
        "label": "pre_injury_history",
        "start": None,
        "end": "2022-03-13",
        "precision": "unknown",
    },
]
FIXTURE_CONSTRAINTS = [
    (
        "Address only the listed questions; do not expand into unrelated causation "
        "opinion beyond evidence."
    ),
    "Report focus: capacity for pre-injury duties.",
]
FIXTURE_NORMALIZED_QUESTIONS = [
    {
        "question_key": "Q1",
        "ordinal": 1,
        "text": "What is the history of the claimed injury of 14 March 2022?",
        "source": "supplied_list",
        "retrieval_intent": (
            "ED/GP/ambulance notes and employer incident reports describing "
            "mechanism and immediate symptoms around 2022-03-14"
        ),
        "likely_sections": ["history", "treatment"],
        "clarity": "clear",
        "clarification_note": None,
    },
    {
        "question_key": "Q2",
        "ordinal": 2,
        "text": FIXTURE_QUESTION_TEXT,
        "source": "supplied_list",
        "retrieval_intent": (
            "Medication lists, scripts, GP summaries after index event; "
            "drug names, dose, prescriber"
        ),
        "likely_sections": ["medications", "treatment"],
        "clarity": "clear",
        "clarification_note": None,
    },
]
FIXTURE_SECTION_PRIORITIES = [
    {
        "section": "history",
        "priority": "high",
        "reason": "Q1 centres on index injury narrative",
    },
    {"section": "medications", "priority": "high", "reason": "Q2 explicit"},
    {
        "section": "employment",
        "priority": "medium",
        "reason": "capacity for pre-injury duties mentioned in constraints",
    },
    {
        "section": "past_history",
        "priority": "medium",
        "reason": "likely needed for baseline",
    },
    {
        "section": "treatment",
        "priority": "high",
        "reason": "supports history and capacity",
    },
    {
        "section": "social",
        "priority": "low",
        "reason": "not directly questioned yet",
    },
]
FIXTURE_INITIAL_TASKS = [
    {
        "task_key": "T1",
        "task_type": "obtain_pages",
        "priority": 1,
        "parent_question_key": "Q1",
        "goal": (
            "Retrieve pages describing 14 Mar 2022 injury mechanism "
            "and acute presentation"
        ),
    },
    {
        "task_key": "T2",
        "task_type": "obtain_pages",
        "priority": 1,
        "parent_question_key": "Q2",
        "goal": "Retrieve medication lists and prescription records",
    },
    {
        "task_key": "T3",
        "task_type": "extract_facts",
        "priority": 2,
        "parent_question_key": "Q1",
        "goal": "Extract atomic history/treatment facts from selected pages",
    },
    {
        "task_key": "T4",
        "task_type": "link_question",
        "priority": 3,
        "parent_question_key": "Q1",
        "goal": "Link extracted facts to Q1",
    },
]
FIXTURE_PLANNER_NOTES = (
    "Strong index date. Prioritise ED and GP notes near 2022-03-14 before imaging."
)


def build_fixture_planner_prediction() -> dict[str, str]:
    """JSON-string fields matching PlanCase outputs (for FakePlanner / demos)."""
    return {
        "entities_json": json.dumps(FIXTURE_ENTITIES),
        "date_windows_json": json.dumps(FIXTURE_DATE_WINDOWS),
        "constraints_json": json.dumps(FIXTURE_CONSTRAINTS),
        "normalized_questions_json": json.dumps(FIXTURE_NORMALIZED_QUESTIONS),
        "section_priorities_json": json.dumps(FIXTURE_SECTION_PRIORITIES),
        "initial_tasks_json": json.dumps(FIXTURE_INITIAL_TASKS),
        "planner_notes": FIXTURE_PLANNER_NOTES,
    }


def build_fixture_pages() -> list[PageRecord]:
    """Synthetic pages — meds, multi-page ED injury span, billing, employment."""
    return [
        PageRecord(
            page_id="page_ed_1",
            doc_id="doc_ed_2022_03_14",
            path="hospital/ed-2022-03-14.pdf",
            page_number=1,
            text=(
                "ED triage 14 March 2022. Jane Doe. "
                "Slipped on oil in warehouse today. Continues on next page."
            ),
            category_hints=("history", "acute_injury"),
            page_date_hint="2022-03-14",
        ),
        PageRecord(
            page_id="page_ed_2",
            doc_id="doc_ed_2022_03_14",
            path="hospital/ed-2022-03-14.pdf",
            page_number=2,
            text=(
                "Immediate LBP radiating to left leg. "
                "Mechanism: slip on oil in warehouse with acute lumbar pain."
            ),
            category_hints=("history", "acute_injury"),
            page_date_hint="2022-03-14",
        ),
        PageRecord(
            page_id="page_a91",
            doc_id="doc_gp_2023_01",
            path="gp/2023-01-12.pdf",
            page_number=2,
            text=(
                "GP review 12 January 2023. Jane Doe attended for back pain follow-up. "
                "Medications: tramadol 50mg BD. Also taking paracetamol 1g PRN. "
                "Prescribed by Dr Patel. Plan: continue physio."
            ),
            category_hints=("medications", "treatment"),
            page_date_hint="2023-01-12",
        ),
        PageRecord(
            page_id="page_b02",
            doc_id="doc_discharge",
            path="hospital/discharge-2022-03-16.pdf",
            page_number=4,
            text=(
                "City General Hospital discharge summary. Discharge medications: "
                "ibuprofen 400mg TDS for 5 days; tramadol 50mg at night if needed. "
                "Follow up with GP Dr Patel."
            ),
            category_hints=("medications", "treatment"),
            page_date_hint="2022-03-16",
        ),
        PageRecord(
            page_id="page_c11",
            doc_id="doc_invoice",
            path="admin/invoice-physio.pdf",
            page_number=1,
            text=(
                "TAX INVOICE. Account payment due. Physiotherapy sessions x4. "
                "Amount owing $480. No clinical medication chart on this page."
            ),
            category_hints=("admin",),
            page_date_hint="2023-02-01",
        ),
        PageRecord(
            page_id="page_e10",
            doc_id="doc_employer_stmt",
            path="employer/statement.pdf",
            page_number=1,
            text=(
                "Employer statement: Jane Doe employed full-time as forklift operator "
                "at Northside Logistics Pty Ltd. Pre-injury duties include lifting."
            ),
            category_hints=("employment",),
            page_date_hint="2022-04-01",
        ),
    ]


def build_fixture_store() -> FixturePageStore:
    """In-memory page store loaded with the spike fixture corpus."""
    return FixturePageStore(build_fixture_pages())
