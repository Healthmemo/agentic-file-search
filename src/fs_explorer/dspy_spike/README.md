# DSPy spike — medicolegal retrieve → categorize → extract

Minimal runnable spike (not the full medicolegal product).

- **Local default:** Ollama via DSPy / LiteLLM (`ollama_chat/...`)
- **Production path:** Amazon Bedrock via DSPy / LiteLLM (`bedrock/...`)
- **Retrieval:** in-memory fixture page store (Postgres not required)

## Install

```bash
uv pip install -e ".[dspy]"
# or: uv pip install dspy
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

Keys come from the environment via `os.getenv` — do not commit secrets.

## Run (live LM)

```bash
# Local Ollama (start `ollama serve` and pull the model first)
ollama pull llama3.2
DSPY_LM_PROVIDER=ollama uv run explore-dspy-spike
# or
uv run python -m fs_explorer.dspy_spike --provider ollama

# Bedrock (AWS credentials required)
DSPY_LM_PROVIDER=bedrock \
DSPY_BEDROCK_MODEL=amazon.nova-lite-v1:0 \
DSPY_BEDROCK_REGION=us-east-1 \
uv run explore-dspy-spike --provider bedrock
```

## Tests (mocked LM — no live Ollama/Bedrock)

```bash
uv run pytest tests/test_dspy_spike.py -q
```

## Layout

| File | Role |
|------|------|
| `lm.py` | Ollama / Bedrock provider wiring |
| `signatures.py` | Thin DSPy signatures (search / categorize / extract) |
| `modules.py` | Modules + `MedicolegalRetrieveExtract` orchestrator |
| `retrieval.py` | `PageStore` protocol + `FixturePageStore` |
| `fixtures.py` | Fake insurer question + page snippets |
| `cli.py` | Typer runner |

## Out of scope

Planner, task manager, cross-page linker, question–fact linker, section mapper,
answer-assist, Postgres/`search_pages` production wiring, DSPy optimizers
(BootstrapFewShot / MIPROv2 / GEPA), and rewriting `FsExplorerWorkflow`.
