"""Metrics for the DSPy optimizer spike (CI-safe; no LM required)."""

from __future__ import annotations

import json
from typing import Any


def _loads_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def _link_edges(links: list[Any]) -> set[tuple[str, str]]:
    edges: set[tuple[str, str]] = set()
    for item in links:
        if not isinstance(item, dict):
            continue
        q = str(item.get("question_key") or "").strip()
        f = str(item.get("fact_id") or "").strip()
        if q and f:
            edges.add((q, f))
    return edges


def question_fact_link_f1(
    example: Any,
    pred: Any,
    trace: Any | None = None,
    **kwargs: Any,
) -> float:
    """Micro-F1 on ``(question_key, fact_id)`` edges vs gold ``links_json``.

    Used to score QuestionFactLinker predictions. Gold edges come from
    ``example.links_json`` (or ``example.gold_links_json``). Predicted edges
    come from ``pred.links_json``. Roles/relevance are ignored for this spike
    metric — only edge presence counts.
    """
    del trace, kwargs  # DSPy may pass extras
    gold_raw = getattr(example, "links_json", None)
    if gold_raw is None:
        gold_raw = getattr(example, "gold_links_json", None)
    gold = _link_edges(_loads_list(gold_raw))
    pred_links = _link_edges(_loads_list(getattr(pred, "links_json", None)))

    if not gold and not pred_links:
        return 1.0
    if not gold or not pred_links:
        return 0.0

    tp = len(gold & pred_links)
    precision = tp / len(pred_links)
    recall = tp / len(gold)
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def question_fact_link_pass(
    example: Any,
    pred: Any,
    trace: Any | None = None,
    **kwargs: Any,
) -> bool:
    """BootstrapFewShot gate: True when micro-F1 is effectively perfect (≥ 0.99)."""
    return question_fact_link_f1(example, pred, trace, **kwargs) >= 0.99


def search_recall_at_k(
    example: Any,
    pred: Any,
    k: int = 30,
    trace: Any | None = None,
    **kwargs: Any,
) -> float:
    """Recall@k of relevant page_ids (for a future search-term optimize stage).

    Expects ``example.gold_page_ids`` (list/set) and ``pred.hit_page_ids`` (list)
    after the orchestrator runs emitted queries against a PageStore. Included so
    the spike documents the integration-doc metric sketch; the default optimize
    stage is question–fact linking.
    """
    del trace, kwargs
    gold_raw = getattr(example, "gold_page_ids", None) or []
    hits_raw = getattr(pred, "hit_page_ids", None) or []
    gold = {str(x) for x in gold_raw}
    hits = [str(x) for x in hits_raw][:k]
    if not gold:
        return 1.0
    return len(gold & set(hits)) / float(len(gold))
