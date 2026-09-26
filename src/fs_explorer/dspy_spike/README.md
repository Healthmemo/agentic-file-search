# DSPy spike — retrieve → extract → cross-page → section map → Q–fact link

Minimal runnable spike (not the full medicolegal product).

- **Local default:** Ollama · **Production:** Amazon Bedrock
- **Retrieval:** fixture store default; Postgres via `DSPY_PAGE_STORE=postgres`
- **Pipeline order:** search → categorize → extract → **cross-page** → **section map** → **question–fact link**  
  (section map and question link both use post-merge facts; section map runs first)

## Install / run / tests

```bash
uv sync --group dev
uv run pytest tests/test_dspy_spike.py tests/test_dspy_spike_postgres.py -q

DSPY_LM_PROVIDER=ollama uv run explore-dspy-spike
DSPY_PAGE_STORE=postgres uv run explore-dspy-spike --page-store postgres
```

## Env (via `os.getenv`)

| Variable | Default | Notes |
|----------|---------|--------|
| `DSPY_LM_PROVIDER` | `ollama` | `ollama` \| `bedrock` |
| `DSPY_OLLAMA_*` / `DSPY_BEDROCK_*` / `AWS_*` | see code | LM backends |
| `DSPY_PAGE_STORE` | `fixture` | `fixture` \| `postgres` |
| `DATABASE_URL` / `DSPY_PG_*` | compose defaults | Postgres |

## Layout

| File | Role |
|------|------|
| `signatures.py` | Incl. `MapFactSections`, `LinkCrossPageFacts`, `LinkQuestionsToFacts` |
| `modules.py` | Orchestrator ends with section map then Q–fact link |
| `fixtures.py` | Multi-page ED span + meds + `FIXTURE_INDEX_EVENT_DATE` |
| `postgres_store.py` / `store_factory.py` | Postgres adapter |
| `cli.py` | Typer runner |

## Out of scope / pull-in order

**Still out:** planner, task manager, answer-assist, optimizers, FsExplorer rewrite.

1. Postgres PageStore — *done*
2. Question–fact linker — *done*
3. Cross-page linker — *done*
4. Section mapper — *done for spike*
5. DSPy optimizers ← next (last)
