"""Offline / live DSPy optimization for one spike stage (QuestionFactLinker).

Default offline path uses ``LabeledFewShot`` (no LM) to materialize demos from
the gold trainset. Live path uses ``BootstrapFewShot`` with Ollama (default) or
Bedrock via ``DSPY_LM_PROVIDER``.

MIPROv2 / GEPA are available in DSPy but intentionally not wired here — the
spike only proves one-stage BootstrapFewShot / labeled-demo compile. Use those
optimizers later when ~20–50 labeled examples exist (see integration doc).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Literal

import dspy
from dspy.teleprompt import BootstrapFewShot, LabeledFewShot

from ..lm import LMConfigError, build_lm, resolve_provider
from ..modules import QuestionFactLinker
from .metrics import question_fact_link_pass
from .trainset import (
    DEFAULT_MAX_BOOTSTRAPPED_DEMOS,
    DEFAULT_MAX_LABELED_DEMOS,
    OPTIMIZE_STAGE,
    build_question_fact_trainset,
)

OptimizerName = Literal["labeled", "bootstrap"]

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
DEFAULT_ARTIFACT_PATH = ARTIFACTS_DIR / "question_fact_linker.json"


def default_artifact_path() -> Path:
    return DEFAULT_ARTIFACT_PATH


def compile_question_fact_linker(
    *,
    optimizer: OptimizerName = "labeled",
    max_labeled_demos: int = DEFAULT_MAX_LABELED_DEMOS,
    max_bootstrapped_demos: int = DEFAULT_MAX_BOOTSTRAPPED_DEMOS,
    trainset: list[dspy.Example] | None = None,
) -> QuestionFactLinker:
    """Compile QuestionFactLinker with labeled demos or BootstrapFewShot."""
    data = trainset if trainset is not None else build_question_fact_trainset()
    student = QuestionFactLinker()

    if optimizer == "labeled":
        teleprompter: Any = LabeledFewShot(k=max_labeled_demos)
        compiled = teleprompter.compile(student, trainset=data)
    elif optimizer == "bootstrap":
        teleprompter = BootstrapFewShot(
            metric=question_fact_link_pass,
            max_bootstrapped_demos=max_bootstrapped_demos,
            max_labeled_demos=max_labeled_demos,
        )
        compiled = teleprompter.compile(student, trainset=data)
    else:
        raise ValueError(
            f"optimizer must be 'labeled' or 'bootstrap', got {optimizer!r}"
        )

    if not isinstance(compiled, QuestionFactLinker):
        # DSPy may return the same module instance
        return student if compiled is None else compiled  # type: ignore[return-value]
    return compiled


def save_optimized_program(
    program: QuestionFactLinker,
    path: Path | None = None,
) -> Path:
    """Persist a compiled QuestionFactLinker (JSON via DSPy ``save``)."""
    out = path or default_artifact_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    program.save(str(out))
    return out


def load_optimized_question_fact_linker(
    path: Path | None = None,
) -> QuestionFactLinker:
    """Load a previously saved QuestionFactLinker (no LM required)."""
    src = path or default_artifact_path()
    if not src.is_file():
        raise FileNotFoundError(f"Optimized program not found: {src}")
    module = QuestionFactLinker()
    module.load(str(src))
    return module


def run_optimize(
    *,
    optimizer: OptimizerName = "labeled",
    provider: str | None = None,
    model: str | None = None,
    output: Path | None = None,
    configure_lm: bool | None = None,
) -> Path:
    """Compile and save the question–fact linker program.

    Parameters
    ----------
    optimizer:
        ``labeled`` — offline, no LM (default for regenerating the committed artifact).
        ``bootstrap`` — BootstrapFewShot; requires a live LM.
    provider / model:
        Passed to ``build_lm`` when ``bootstrap`` (or when configure_lm is True).
    """
    use_lm = configure_lm if configure_lm is not None else (optimizer == "bootstrap")
    if use_lm:
        build_lm(provider, model=model, configure=True)

    compiled = compile_question_fact_linker(optimizer=optimizer)
    return save_optimized_program(compiled, output)


def _cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            f"DSPy spike optimizer ({OPTIMIZE_STAGE}). "
            "Default: labeled demos offline. Use --optimizer bootstrap for live LM."
        )
    )
    parser.add_argument(
        "--optimizer",
        choices=("labeled", "bootstrap"),
        default="labeled",
        help="labeled=no LM (default); bootstrap=BootstrapFewShot with live LM",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="LM provider for bootstrap: ollama (default) or bedrock",
    )
    parser.add_argument("--model", default=None, help="Override model id")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Artifact path (default: {DEFAULT_ARTIFACT_PATH})",
    )
    parser.add_argument(
        "--evaluate-artifact",
        action="store_true",
        help="Load artifact and score micro-F1 on the gold trainset (no compile)",
    )
    args = parser.parse_args(argv)

    try:
        if args.evaluate_artifact:
            from .metrics import question_fact_link_f1

            program = load_optimized_question_fact_linker(args.output)
            # Evaluating demos without calling the LM: score gold vs gold via predict demos
            # is not meaningful. Instead report demo count and trainset size.
            trainset = build_question_fact_trainset()
            demos = getattr(getattr(program, "predict", None), "demos", None)
            if demos is None:
                demos = getattr(
                    getattr(getattr(program, "predict", None), "predict", None),
                    "demos",
                    [],
                )
            print(f"stage={OPTIMIZE_STAGE}")
            print(f"artifact={args.output or default_artifact_path()}")
            print(f"trainset_size={len(trainset)}")
            print(f"demos={len(demos) if demos is not None else 0}")
            # Score metric helper on gold-as-pred sanity check
            scores = []
            for ex in trainset:
                fake_pred = dspy.Prediction(
                    links_json=ex.links_json,
                    question_coverage_json=ex.question_coverage_json,
                )
                scores.append(question_fact_link_f1(ex, fake_pred))
            print(f"gold_self_f1_mean={sum(scores) / len(scores):.3f}")
            return 0

        if args.optimizer == "bootstrap":
            try:
                resolved = resolve_provider(args.provider)
            except LMConfigError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 2
            print(f"provider={resolved} optimizer=bootstrap")
        else:
            print("optimizer=labeled (offline, no LM)")

        path = run_optimize(
            optimizer=args.optimizer,
            provider=args.provider,
            model=args.model,
            output=args.output,
        )
        print(f"stage={OPTIMIZE_STAGE}")
        print(f"saved={path}")
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI surface
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def main() -> None:
    raise SystemExit(_cli())


if __name__ == "__main__":
    main()
