"""Tiny offline fixture: fake insurer question + a few medical page snippets."""

from __future__ import annotations

from .retrieval import FixturePageStore, PageRecord

# Major spike fixture knobs (not user input)
FIXTURE_CASE_ID = "case_spike_acme_0142"
FIXTURE_QUESTION_KEY = "Q2"
FIXTURE_QUESTION_TEXT = "List current medications and who prescribed them."
FIXTURE_TASK_GOAL = (
    "Retrieve medication lists and prescription records after the index event"
)
FIXTURE_TARGET_CATEGORY = "medications"
FIXTURE_SECTION_HINT = "medications"
FIXTURE_ALLOWED_FACT_TYPES = "medication,injury_mechanism,presentation"
FIXTURE_MIN_RELEVANCE = 0.55
FIXTURE_HOUSE_RULE_CANONICAL = (
    "Prefer the page where the claim is most complete; if equal, "
    "prefer the lowest page_number within the same document."
)
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
FIXTURE_ENTITIES = {
    "claimant_names": ["Jane Marie Doe", "Jane Doe"],
    "providers": ["Dr Patel"],
    "body_parts": ["lumbar spine"],
    "insurers": ["Acme Workers Comp"],
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
]


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
