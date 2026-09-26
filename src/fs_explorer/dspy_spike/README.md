# DSPy spike — plan → retrieve → extract → link (+ optimizer)

Minimal medicolegal DSPy spike (not the full product).

**Pipeline:** `CasePlanner` → search → categorize → extract → cross-page → section map → question–fact link  

**Optimize:** QuestionFactLinker via `LabeledFewShot` (offline artifact) or `BootstrapFewShot` (live LM). Metric: micro-F1 on `(question_key, fact_id)` edges. MIPROv2/GEPA not wired (use later with larger gold).

Planner reads fixture letter/brief/questions and emits entities, date windows, normalized questions, section priorities, and light `initial_tasks` (no TaskManager persistence / answer-assist yet).

## Install / tests

```bash
uv sync --group dev
uv run pytest tests/test_dspy_spike.py tests/test_dspy_spike_postgres.py tests/test_dspy_spike_optimize.py -q
```

## Run pipeline

```bash
# Planner on by default (fixture letter/brief/questions)
DSPY_LM_PROVIDER=ollama uv run explore-dspy-spike

# Skip planner (compat path with fixture task_goal/entities)
uv run explore-dspy-spike --skip-planner

DSPY_PAGE_STORE=postgres uv run explore-dspy-spike --page-store postgres
```

## Run optimizer

```bash
# Offline — regenerates artifacts/question_fact_linker.json (no LM)
uv run explore-dspy-optimize --optimizer labeled
# or: uv run python -m fs_explorer.dspy_spike.optimize --optimizer labeled

# Live BootstrapFewShot
DSPY_LM_PROVIDER=ollama uv run explore-dspy-optimize --optimizer bootstrap
DSPY_LM_PROVIDER=bedrock uv run explore-dspy-optimize --optimizer bootstrap --provider bedrock

uv run explore-dspy-optimize --evaluate-artifact
```

Load: `from fs_explorer.dspy_spike import load_optimized_question_fact_linker`.

## Env

| Variable | Default | Notes |
|----------|---------|--------|
| `DSPY_LM_PROVIDER` | `ollama` | `ollama` \| `bedrock` |
| `DSPY_OLLAMA_*` / `DSPY_BEDROCK_*` / `AWS_*` | — | LM backends |
| `DSPY_PAGE_STORE` | `fixture` | `fixture` \| `postgres` |
| `DATABASE_URL` / `DSPY_PG_*` | compose defaults | Postgres |

## Layout

| Path | Role |
|------|------|
| `signatures.py` | `PlanCase`, search/categorize/extract/link signatures |
| `modules.py` | `CasePlanner` + orchestrator |
| `fixtures.py` | Letter/brief/questions + page corpus |
| `optimize/metrics.py` | `question_fact_link_f1` (+ search recall sketch) |
| `optimize/trainset.py` | Tiny gold examples |
| `optimize/runner.py` | Compile / save / load / CLI |
| `artifacts/question_fact_linker.json` | Committed labeled demos |

## Pull-in order

1–5 done (Postgres, Q–fact link, cross-page, section map, optimizers).  
6. **CasePlanner / PlanCase** — *done for spike* (light `initial_tasks` only).  

**Later / separate:** task manager persistence, answer-assist, MIPROv2/GEPA, FsExplorer rewrite.
