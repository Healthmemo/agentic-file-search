"""DSPy optimizer spike: BootstrapFewShot / LabeledFewShot for one stage.

Default stage: QuestionFactLinker (micro-F1 on question–fact edges).
"""

from .metrics import (
    question_fact_link_f1,
    question_fact_link_pass,
    search_recall_at_k,
)
from .runner import (
    compile_question_fact_linker,
    default_artifact_path,
    load_optimized_question_fact_linker,
    run_optimize,
    save_optimized_program,
)
from .trainset import OPTIMIZE_STAGE, build_question_fact_trainset

__all__ = [
    "OPTIMIZE_STAGE",
    "build_question_fact_trainset",
    "compile_question_fact_linker",
    "default_artifact_path",
    "load_optimized_question_fact_linker",
    "question_fact_link_f1",
    "question_fact_link_pass",
    "run_optimize",
    "save_optimized_program",
    "search_recall_at_k",
]
