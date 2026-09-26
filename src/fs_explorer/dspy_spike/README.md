# DSPy spike — medicolegal retrieve → extract → link

Minimal runnable spike (not the full medicolegal product).

- **Local default:** Ollama via DSPy / LiteLLM (`ollama_chat/...`)
- **Production path:** Amazon Bedrock via DSPy / LiteLLM (`bedrock/...`)
- **Retrieval:** `FixturePageStore` by default; optional Postgres via `DSPY_PAGE_STORE=postgres`
- **Pipeline:** search terms → snippets → categorize → extract → **question–fact link**

## Install

```bash
uv pip install -e ".[dspy]"
uv pip install -e ".[dspy,postgres]"   # for Postgres adapter
# or: uv sync --group dev
```

## Environment variables

| Variable | Default | Notes |
|----------|---------|--------|
| `DSPY_LM_PROVIDER` | `ollama` | `ollama` or `bedrock` |
| `DSPY_OLLAMA_MODEL` | `llama3.2` | Ollama model tag |
| `DSPY_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server |
| `DSPY_BEDROCK_MODEL` | `amazon.nova-lite-v1:0` | Bedrock model id |
| `DSPY_BEDROCK_REGION` | `us-east-1` | falls back to `AWS_REGION` / `AWS_DEFAULT_REGION` |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_PROFILE` | — | standard AWS creds for Bedrock |
| `DSPY_LM_TEMPERATURE` | unset | optional float |
| `DSPY_LM_MAX_TOKENS` | `2048` | max generation tokens |
| `DSPY_PAGE_STORE` | `fixture` | `fixture` or `postgres` |
| `DATABASE_URL` | unset | preferred Postgres DSN |
| `DSPY_PG_HOST` / `PORT` / `USER` / `PASSWORD` / `DATABASE` | compose defaults | used if `DATABASE_URL` unset |

Keys come from the environment via `os.getenv` — do not commit secrets.

## Run (live LM)

```bash
# Local Ollama + fixture store (default)
ollama pull llama3.2
DSPY_LM_PROVIDER=ollama uv run explore-dspy-spike

# Bedrock (AWS credentials required)
DSPY_LM_PROVIDER=bedrock uv run explore-dspy-spike --provider bedrock

# Postgres page store (docker compose + schema)
docker compose -f docker/docker-compose.yml up -d
uv run python -m fs_explorer.dspy_spike.postgres_store --init-schema --seed-fixture
DSPY_PAGE_STORE=postgres uv run explore-dspy-spike --page-store postgres
```

## Tests (mocked LM — no live Ollama/Bedrock/Postgres)

```bash
uv run pytest tests/test_dspy_spike.py tests/test_dspy_spike_postgres.py -q
# Optional: DSPY_PG_INTEGRATION=1 uv run pytest tests/test_dspy_spike_pg_integration.py -q
```

## Layout

| File | Role |
|------|------|
| `lm.py` | Ollama / Bedrock provider wiring |
| `signatures.py` | Signatures incl. `LinkQuestionsToFacts` |
| `modules.py` | Modules + orchestrator (ends with question–fact link) |
| `retrieval.py` | `PageStore` protocol + `FixturePageStore` |
| `postgres_store.py` | Postgres `PageStore` + schema init / seed helpers |
| `store_factory.py` | Fixture vs Postgres selection |
| `schema.sql` | Minimal spike pages schema |
| `fixtures.py` | Fake questions + page snippets |
| `cli.py` | Typer runner |

## Out of scope / suggested pull-in order

**Still out:** planner, task manager, answer-assist, cross-page linker, section mapper,
full ingest pipeline, optimizers, rewriting `FsExplorerWorkflow`.

**Pull-in order:**

1. Postgres `PageStore` / production `search_pages` — *done (minimal)*
2. Question–fact linker — *done for spike*
3. Cross-page linker ← next
4. Section mapper
5. DSPy optimizers last
