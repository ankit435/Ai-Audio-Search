# SETUP.md — Getting the Audio Search stack running

This is the practical "clone it and run it" guide. For design
rationale, real evaluation numbers, and known limitations, see
`SOLUTION.md`. For task-by-task completion status, see `PROGRESS.md`.

## 1. Prerequisites

| Requirement | Notes |
|---|---|
| Python >= 3.11 | `python3 --version` |
| Postgres 14+ with `pgvector` | via Docker (easiest) or native Homebrew (fallback if Docker Hub is blocked — see §2b) |
| `ffmpeg` on `PATH` | required by the `openai-whisper` transcription backend (`brew install ffmpeg` / `apt install ffmpeg`) |
| ~1-2 GB disk | for the Whisper model checkpoint + sentence-transformers model |

## 2. Database setup

### Option A — Docker (preferred, if not blocked by network policy)
```bash
docker compose up -d       # starts Postgres 16 + pgvector, auto-loads db/schema.sql
```

### Option B — Native Homebrew Postgres + pgvector built from source
Use this if Docker Hub image pulls are blocked (this was the case in
the original development environment):
```bash
brew install postgresql@14
brew services start postgresql@14

git clone --branch v0.8.6 https://github.com/pgvector/pgvector /tmp/pgvector-src
cd /tmp/pgvector-src
PG_CONFIG=$(brew --prefix postgresql@14)/bin/pg_config make
PG_CONFIG=$(brew --prefix postgresql@14)/bin/pg_config make install

createuser -s audio_search      # or use your existing superuser role
createdb -O audio_search audio_search
psql -d audio_search -c "ALTER USER audio_search WITH PASSWORD 'audio_search';"
psql -d audio_search -f db/schema.sql
```

Verify pgvector loaded correctly:
```bash
psql -d audio_search -c "CREATE EXTENSION IF NOT EXISTS vector; SELECT extversion FROM pg_extension WHERE extname='vector';"
```

## 3. Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -e ".[dev]"            # core app + pytest/pytest-asyncio/httpx
pip install pgvector                # python client for the vector type (asyncpg codec)
pip install -e ".[transcription]"  # openai-whisper + faster-whisper (real ASR)
# pip install -e ".[diarization]"  # pyannote.audio — optional, see §6
```

## 4. Configuration

```bash
cp .env.example .env
```
All settings are read once at the composition root
(`src/api/settings.py`) and are env vars prefixed `AUDIO_SEARCH_`:

| Var | Default | Purpose |
|---|---|---|
| `AUDIO_SEARCH_DATABASE_URL` | `******localhost:5432/audio_search` | asyncpg connection string |
| `AUDIO_SEARCH_EMBEDDING_PROVIDER` | `sentence-transformers` | `sentence-transformers` (local) or `openai-compatible` (hosted/self-hosted `/v1/embeddings` — NVIDIA NIM, OpenAI, vLLM, TEI, ...) |
| `AUDIO_SEARCH_EMBEDDING_MODEL_NAME` | `all-MiniLM-L6-v2` | model id/name for whichever provider is selected (384-dim default, matches `db/schema.sql`) |
| `AUDIO_SEARCH_EMBEDDING_BASE_URL` | *(unset)* | only used when `EMBEDDING_PROVIDER=openai-compatible`, e.g. `https://integrate.api.nvidia.com/v1` |
| `AUDIO_SEARCH_WHISPER_MODEL_SIZE` | `base` | `tiny`/`base`/`small`/... — or a local path / HF repo id when using `faster-whisper` |
| `AUDIO_SEARCH_TRANSCRIPTION_PRIMARY_BACKEND` | `faster-whisper` | tried first; falls back automatically on failure (see §6) |
| `AUDIO_SEARCH_TRANSCRIPTION_FALLBACK_BACKEND` | `openai-whisper` | used if the primary backend's model is unavailable |
| `AUDIO_SEARCH_ANSWER_MODEL` | `gpt-4o-mini` | chat-completions model name for `/search/answer` LLM mode |
| `AUDIO_SEARCH_ANSWER_BASE_URL` | *(unset)* | any OpenAI-compatible endpoint — NVIDIA NIM, self-hosted vLLM/Ollama, etc. Unset = hosted OpenAI |
| `AUDIO_SEARCH_LOG_LEVEL` | `INFO` | structured JSON logs (`src/infra/logging_config.py`) |
| `AUDIO_SEARCH_DEFAULT_TOP_K` | `10` | default result count |
| `AUDIO_SEARCH_RRF_K` | `60` | Reciprocal Rank Fusion constant |
| `HF_TOKEN` | *(unset)* | only needed for real `pyannote.audio` diarization (gated model) |
| `OPENAI_API_KEY` / `NVIDIA_API_KEY` | *(unset)* | key for whichever OpenAI-compatible embedding/answer provider is configured — safe to leave unset, everything falls back automatically (see §6) |
| `HF_HUB_OFFLINE` | *(unset)* | set to `1` once models are cached locally to skip slow online cache-validation checks (see §7 troubleshooting) |

### Using other model providers (NVIDIA NIM, other Hugging Face models, self-hosted servers)

Every ML integration point is a swappable adapter behind a small
`Protocol` port (`src/domain/ports.py`) chosen at the composition root
(`src/api/main.py`) — no application/domain code changes needed to
switch providers:

- **Embeddings via NVIDIA NIM** (or OpenAI, or any self-hosted
  OpenAI-compatible `/v1/embeddings` server — vLLM, TEI, Ollama, ...):
  ```bash
  AUDIO_SEARCH_EMBEDDING_PROVIDER=openai-compatible
  AUDIO_SEARCH_EMBEDDING_BASE_URL=https://integrate.api.nvidia.com/v1
  AUDIO_SEARCH_EMBEDDING_MODEL_NAME=nvidia/nv-embedqa-e5-v5
  NVIDIA_API_KEY=nvapi-...
  ```
  ⚠️ If the new model's output dimension differs from 384, update the
  `vector(384)` column in `db/schema.sql` to match before ingesting.
- **Any other Hugging Face `sentence-transformers`-compatible model**
  (default provider): just change `AUDIO_SEARCH_EMBEDDING_MODEL_NAME` to
  any HF repo id, e.g. `BAAI/bge-small-en-v1.5` — same dimension caveat
  applies.
- **A different Whisper checkpoint** (community/converted, local, or a
  different size): change `AUDIO_SEARCH_WHISPER_MODEL_SIZE` to a local
  path or HF repo id (works with the `faster-whisper` backend, which
  loads any CTranslate2-format checkpoint).
- **LLM answers via NVIDIA NIM** (or any OpenAI-compatible chat
  endpoint) instead of OpenAI:
  ```bash
  AUDIO_SEARCH_ANSWER_BASE_URL=https://integrate.api.nvidia.com/v1
  AUDIO_SEARCH_ANSWER_MODEL=meta/llama-3.1-8b-instruct
  NVIDIA_API_KEY=nvapi-...
  ```
- **A genuinely new provider shape** (not OpenAI-compatible, e.g. a
  gRPC-based service): implement the relevant port
  (`Embedder`/`Transcriber`/`Diarizer`/`AnswerGenerator`) as a new class
  under `src/infra/`, then register it in the matching factory in
  `src/api/main.py` (`_build_embedder`, `_build_answer_generator`, or
  `lifespan()`). Existing adapters (e.g.
  `src/infra/openai_compatible_embedder.py`) are the template to follow.

## 5. Run the tests

```bash
export AUDIO_SEARCH_DATABASE_URL="postgresql://audio_search:audio_search@localhost:5432/audio_search"
pytest -q
```
- Pure unit tests (chunking, fusion, search/ingest services) run with
  zero external dependencies via in-memory fakes (`tests/fakes.py`).
- Real Postgres/pgvector integration tests
  (`tests/test_postgres_integration.py`, `tests/test_ingestion_worker.py`,
  `tests/test_postgres_feedback_repository.py`) auto-skip if the DB
  isn't reachable — no manual flag needed.
- Golden-dataset evaluation tests
  (`tests/test_recall_at_k_golden.py`,
  `tests/test_speaker_accuracy_and_latency.py`) auto-skip unless the
  golden dataset has been ingested (§8).
- Expect **51 passed** with the DB up and the golden dataset ingested;
  fewer (with skips) otherwise — both are a healthy result.

## 6. Run the API

```bash
uvicorn src.api.main:app --reload --port 8000
```
Then open `http://127.0.0.1:8000/docs` (Swagger UI) or call directly:
```bash
curl "http://127.0.0.1:8000/search?q=your+query&top_k=5"
curl -X POST "http://127.0.0.1:8000/ingest" -H "Content-Type: application/json" \
     -d '{"audio_file_path": "/absolute/path/to/audio.wav"}'
```

**Routes**: `GET /search`, `POST /ingest`, `GET /search/stream` (SSE),
`POST /search/feedback`, `POST /search/answer`.

**You do not need `HF_TOKEN` or `OPENAI_API_KEY` to run this.** Real
diarization (`pyannote.audio`) and real LLM answers (`OpenAIAnswerGenerator`)
are wrapped in automatic fallback adapters
(`src/infra/resilient_diarizer.py`, `src/infra/resilient_answer_generator.py`):
if the real model/API is unavailable for any reason (no token, no
network, blocked proxy, expired key), the request still completes via
a working fallback (`SingleSpeakerDiarizer`, `ExtractiveAnswerGenerator`)
and a clear `*.primary_unavailable` structured log line is emitted with
the real reason — never a silent failure. Same pattern covers Whisper:
`ResilientTranscriber` tries `faster-whisper` first and falls back to
`openai-whisper` automatically.

Stop it with `Ctrl+C` (foreground) or, if started in the background,
find and stop the process:
```bash
ps aux | grep "uvicorn src.api.main"   # note the PID
kill <PID>
```

### Zero-dependency demo (no Docker/Postgres/model downloads needed)
If you just want to see the HTTP contracts work without any real
infra, `scripts/demo_server.py` runs the same FastAPI app wired to
in-memory fake adapters + a small seeded corpus:
```bash
python3 scripts/demo_server.py
curl "http://127.0.0.1:8000/search?q=quarterly+budget"
```

## 7. Troubleshooting

- **`SSL: WRONG_VERSION_NUMBER` downloading Whisper/pyannote/sentence-transformers models** —
  a TLS-inspecting corporate proxy blocking Hugging Face Hub. The
  `openai-whisper` backend downloads from `openaipublic.azureedge.net`
  instead, which is often not blocked — this is the default transcriber
  fallback already (§6). No action needed beyond having `ffmpeg`
  installed.
- **Slow startup / requests after a model is already cached locally** —
  `sentence-transformers`/`huggingface_hub` still attempt online
  cache-validation HEAD requests by default; if the network is blocked
  these retry with backoff (multi-minute delays). Fix: `export
  HF_HUB_OFFLINE=1` once the model is cached.
- **`pyannote.audio` diarization never actually runs** — requires a
  Hugging Face access token accepted for the gated model's license
  (`HF_TOKEN` env var). Without it, `ResilientDiarizer` transparently
  falls back to `SingleSpeakerDiarizer` — ingestion still works, just
  with single-speaker attribution instead of real diarization.
- **`ffmpeg: command not found`** — `openai-whisper` shells out to
  `ffmpeg` for audio decoding: `brew install ffmpeg` (macOS) or `apt
  install ffmpeg` (Debian/Ubuntu).
- **Postgres integration tests all skip** — the `pool` fixture pings
  `AUDIO_SEARCH_DATABASE_URL` and calls `pytest.skip(...)` if
  unreachable; check the DB is running and the URL/credentials match.

## 8. Golden dataset (optional — for real recall@k / accuracy numbers)

Regenerates and ingests the 5-file synthetic golden dataset used by
`tests/test_recall_at_k_golden.py` and
`tests/test_speaker_accuracy_and_latency.py` (macOS only — uses `say`):
```bash
python3 scripts/generate_golden_dataset.py     # writes data/golden/*.wav + *.turns.json
export HF_HUB_OFFLINE=1                        # once models are cached — see §7
python3 scripts/ingest_golden_dataset.py       # real Whisper + ScriptedDiarizer + real embeddings -> Postgres
pytest -q -s tests/test_recall_at_k_golden.py tests/test_speaker_accuracy_and_latency.py
python3 scripts/measure_speaker_accuracy_and_latency.py   # human-readable standalone report
```
This data is intentionally left in the database afterward (persistent
reference/eval fixture) — it is not cleaned up by the test suite the
way ad hoc integration-test rows are.
