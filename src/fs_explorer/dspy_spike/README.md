# DSPy spike — medicolegal retrieve → extract → cross-page → link

Minimal runnable spike (not the full medicolegal product).

- **Local default:** Ollama via DSPy / LiteLLM
- **Production path:** Amazon Bedrock via DSPy / LiteLLM
- **Retrieval:** `FixturePageStore` by default; optional Postgres via `DSPY_PAGE_STORE=postgres`
- **Pipeline:** search → snippets → categorize → extract → **cross-page link** → **question–fact link**

## Install

```bash
uv pip install -e ".[dspy]"
uv pip install -e ".[dspy,postgres]"
# or: uv sync --group dev
```

## Environment variables

| Variable | Default | Notes |
|----------|---------|--------|
| `DSPY_LM_PROVIDER` | `ollama` | `ollama` or `bedrock` |
| `DSPY_OLLAMA_MODEL` | `llama3.2` | |
| `DSPY_OLLAMA_BASE_URL` | `http://localhost:11434` | |
| `DSPY_BEDROCK_MODEL` | `amazon.nova-lite-v1:0` | |
| `DSPY_BEDROCK_REGION` | `us-east-1` | |
| `AWS_*` | — | Bedrock creds |
| `DSPY_PAGE_STORE` | `fixture` | `fixture` or `postgres` |
| `DATABASE_URL` / `DSPY_PG_*` | compose defaults | Postgres |

Keys via `os.getenv` — do not commit secrets.

## Run

```bash
DSPY_LM_PROVIDER=ollama uv run explore-dspy-spike
DSPY_LM_PROVIDER=bedrock uv run explore-dspy-spike --provider bedrock

docker compose -f docker/docker-compose.yml up -d
uv run python -m fs_explorer.dspy_spike.postgres_store --init-schema --seed-fixture
DSPY_PAGE_STORE=postgres uv run explore-dspy-spike --page-store postgres
```

## Tests

```bash
uv run pytest tests/test_dspy_spike.py tests/test_dspy_spike_postgres.py -q
# Optional: DSPY_PG_INTEGRATION=1 uv run pytest tests/test_dspy_spike_pg_integration.py -q
```

## Layout

| File | Role |
|------|------|
| `signatures.py` | Incl. `LinkCrossPageFacts`, `LinkQuestionsToFacts` |
| `modules.py` | Orchestrator: extract → cross-page → question–fact |
| `fixtures.py` | Multi-page ED span + meds/questions |
| `postgres_store.py` / `store_factory.py` | Postgres adapter + selection |
| `cli.py` | Typer runner |

## Out of scope / pull-in order

**Still out:** planner, task manager, answer-assist, section mapper, optimizers, FsExplorer rewrite.

1. Postgres `PageStore` — *done*
2. Question–fact linker — *done*
3. Cross-page linker — *done for spike*
4. Section mapper ← next
5. DSPy optimizers last
