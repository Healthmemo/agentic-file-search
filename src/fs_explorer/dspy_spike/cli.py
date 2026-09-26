"""CLI runner for the DSPy medicolegal retrieve→extract spike.

Usage
-----
uv run explore-dspy-spike
uv run python -m fs_explorer.dspy_spike
uv run python -m fs_explorer.dspy_spike.cli --provider ollama --page-store fixture
"""

from __future__ import annotations

import json
import sys
from typing import Annotated

from typer import Option, Typer

from .fixtures import (
    FIXTURE_ALLOWED_FACT_TYPES,
    FIXTURE_DATE_WINDOWS,
    FIXTURE_ENTITIES,
    FIXTURE_QUESTION_TEXT,
    FIXTURE_SECTION_HINT,
    FIXTURE_TARGET_CATEGORY,
    FIXTURE_TASK_GOAL,
)
from .lm import LMConfigError, build_lm, resolve_provider
from .modules import MedicolegalRetrieveExtract
from .store_factory import (
    PageStoreConfigError,
    build_page_store,
    resolve_page_store_kind,
)

app = Typer(
    add_completion=False,
    help="Run the minimal DSPy medicolegal retrieve→categorize→extract spike.",
)


@app.command()
def run(
    provider: Annotated[
        str | None,
        Option(help="LM provider: ollama (default) or bedrock"),
    ] = None,
    model: Annotated[
        str | None,
        Option(help="Override model id for the selected provider"),
    ] = None,
    page_store: Annotated[
        str | None,
        Option(help="Page store: fixture (default) or postgres"),
    ] = None,
    database_url: Annotated[
        str | None,
        Option(help="Postgres DSN override (else DATABASE_URL / DSPY_PG_*)"),
    ] = None,
    max_deep_review: Annotated[
        int,
        Option(help="Max pages sent to fact extraction"),
    ] = 4,
    task_goal: Annotated[
        str | None,
        Option(help="Override fixture task goal"),
    ] = None,
    question: Annotated[
        str | None,
        Option(help="Override fixture insurer question"),
    ] = None,
) -> None:
    """Run the fixture case through the DSPy pipeline."""
    try:
        resolved = resolve_provider(provider)
        build_lm(resolved, model=model, configure=True)
    except LMConfigError as exc:
        print(f"error: LM configuration failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    except Exception as exc:  # pragma: no cover - live provider failures
        print(
            f"error: failed to initialize LM ({type(exc).__name__}): {exc}",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc

    try:
        store_kind = resolve_page_store_kind(page_store)
        store = build_page_store(store_kind, database_url=database_url)
    except PageStoreConfigError as exc:
        print(f"error: page store configuration failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    except Exception as exc:
        print(
            f"error: failed to initialize page store ({type(exc).__name__}): {exc}",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc

    pipeline = MedicolegalRetrieveExtract(
        store,
        max_hits=40,
        snippet_chars=500,
        max_deep_review=max_deep_review,
    )

    goal = task_goal or FIXTURE_TASK_GOAL
    question_text = question or FIXTURE_QUESTION_TEXT
    page_count = (
        len(store.all_page_ids()) if hasattr(store, "all_page_ids") else "unknown"
    )

    print(f"provider={resolved}")
    print(f"page_store={store_kind}")
    print(f"task_goal={goal}")
    print(f"question={question_text}")
    print(f"pages={page_count}")
    print("---")

    try:
        result = pipeline(
            task_goal=goal,
            question_text=question_text,
            entities_json=json.dumps(FIXTURE_ENTITIES),
            date_windows_json=json.dumps(FIXTURE_DATE_WINDOWS),
            target_category=FIXTURE_TARGET_CATEGORY,
            section_hint=FIXTURE_SECTION_HINT,
            allowed_fact_types=FIXTURE_ALLOWED_FACT_TYPES,
            max_deep_review=max_deep_review,
        )
    except Exception as exc:  # pragma: no cover - live LM failures
        print(f"error: pipeline failed ({type(exc).__name__}): {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    finally:
        close = getattr(store, "close", None)
        if callable(close):
            close()

    summary = {
        "page_store": store_kind,
        "queries": result.queries,
        "stop_conditions": result.stop_conditions,
        "search_hit_page_ids": [h["page_id"] for h in result.search_hits],
        "selections": result.selections,
        "stats": result.stats,
        "selected_page_ids": result.selected_page_ids,
        "facts": result.facts,
        "extraction_notes": result.extraction_notes,
        "needs_review": result.needs_review,
    }
    print(json.dumps(summary, indent=2))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
