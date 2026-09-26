"""CI-safe tests for the DSPy optimizer spike (no live LM)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

dspy = pytest.importorskip("dspy")

from fs_explorer.dspy_spike.optimize.metrics import (
    question_fact_link_f1,
    question_fact_link_pass,
    search_recall_at_k,
)
from fs_explorer.dspy_spike.optimize.runner import (
    compile_question_fact_linker,
    default_artifact_path,
    load_optimized_question_fact_linker,
    save_optimized_program,
)
from fs_explorer.dspy_spike.optimize.trainset import (
    OPTIMIZE_STAGE,
    build_question_fact_trainset,
)


def test_question_fact_link_f1_perfect_and_partial() -> None:
    gold_links = json.dumps(
        [
            {"question_key": "Q1", "fact_id": "F_a"},
            {"question_key": "Q2", "fact_id": "F_b"},
        ]
    )
    example = dspy.Example(links_json=gold_links)
    perfect = dspy.Prediction(links_json=gold_links)
    assert question_fact_link_f1(example, perfect) == 1.0
    assert question_fact_link_pass(example, perfect) is True

    partial = dspy.Prediction(
        links_json=json.dumps([{"question_key": "Q1", "fact_id": "F_a"}])
    )
    score = question_fact_link_f1(example, partial)
    assert 0.0 < score < 1.0
    assert question_fact_link_pass(example, partial) is False

    empty = dspy.Prediction(links_json="[]")
    assert question_fact_link_f1(example, empty) == 0.0


def test_search_recall_at_k_helper() -> None:
    example = dspy.Example(gold_page_ids=["page_a91", "page_ed_2"])
    pred = dspy.Prediction(hit_page_ids=["page_a91", "page_c11", "page_ed_2"])
    assert search_recall_at_k(example, pred, k=30) == 1.0
    pred2 = dspy.Prediction(hit_page_ids=["page_a91"])
    assert search_recall_at_k(example, pred2, k=30) == 0.5


def test_trainset_is_small_and_typed() -> None:
    trainset = build_question_fact_trainset()
    assert OPTIMIZE_STAGE == "question_fact_linker"
    assert 1 <= len(trainset) <= 4
    for ex in trainset:
        assert "questions_json" in ex
        assert "facts_json" in ex
        assert "links_json" in ex
        assert set(ex.inputs()) >= {
            "case_id",
            "questions_json",
            "facts_json",
            "min_relevance",
        }


def test_labeled_compile_and_save_load_roundtrip(tmp_path: Path) -> None:
    """LabeledFewShot needs no LM — safe for CI."""
    compiled = compile_question_fact_linker(optimizer="labeled")
    out = tmp_path / "qfl.json"
    saved = save_optimized_program(compiled, out)
    assert saved.is_file()
    assert saved.stat().st_size > 100

    loaded = load_optimized_question_fact_linker(saved)
    demos = getattr(getattr(loaded, "predict", None), "demos", None)
    if demos is None:
        demos = getattr(
            getattr(getattr(loaded, "predict", None), "predict", None),
            "demos",
            [],
        )
    assert demos is not None
    assert len(demos) >= 1


def test_committed_artifact_loads() -> None:
    path = default_artifact_path()
    assert path.is_file(), f"missing committed artifact {path}"
    loaded = load_optimized_question_fact_linker(path)
    assert loaded is not None


def test_bootstrap_compile_is_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure bootstrap path is callable without a live teacher LM."""
    calls: list[str] = []

    class _FakeBootstrap:
        def __init__(self, **kwargs: object) -> None:
            calls.append("init")

        def compile(self, student: object, trainset: object) -> object:
            calls.append("compile")
            # Simulate returning a labeled compile instead of calling LM
            return compile_question_fact_linker(optimizer="labeled")

    monkeypatch.setattr(
        "fs_explorer.dspy_spike.optimize.runner.BootstrapFewShot",
        _FakeBootstrap,
    )
    compiled = compile_question_fact_linker(optimizer="bootstrap")
    assert calls == ["init", "compile"]
    assert compiled is not None
