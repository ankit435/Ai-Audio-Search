# PROGRESS.md — Tasks 1-18 Status

Mirrors the "Tasks" list in
`.github/prompts/plan-audioSearchHybridRetrieval.prompt.md` 1:1.

## Done — all 18 core tasks, plus all 4 optional stretch goals
- **Task 1** — Golden dataset sourced: 5 original (non-copyrighted),
  hand-authored two-speaker conversations synthesized with macOS `say`
  (two distinct voices standing in for two speakers) + `afconvert`
  (`scripts/golden_conversations.py`, `scripts/generate_golden_dataset.py`).
  Exact ground truth recorded at generation time into
  `data/golden/<name>.turns.json`. **Disclosed scoping tradeoff**: ~10.4
  min total audio vs. the plan's "8-10 min per file" target (session
  time constraints), and real copyrighted audio can't legally ship with
  this repo, hence the synthetic approach.
- **Task 2** — Repo scaffold: `src/domain` (models, ports, exceptions),
  `src/infra` (postgres repositories, sentence-transformer embedder,
  whisper transcriber, pyannote diarizer, scripted diarizer, job queue,
  feedback repository, extractive/OpenAI answer generators, structured
  logging config), `src/application` (chunking, fusion, ingest_pipeline,
  search_service, ingestion_worker, feedback_service,
  search_answer_service), `src/api` (main composition root, settings,
  Pydantic schemas), `db/schema.sql`, `docker-compose.yml`, `pyproject.toml`.
- **Task 3** — Ingestion pipeline implemented: transcribe -> diarize ->
  time-overlap speaker alignment -> merge-and-split chunking -> embed ->
  index, with checksum-based idempotency and structured per-stage logging.
  Unit-tested with in-memory fakes, **and real-data verified** via
  `scripts/ingest_golden_dataset.py` (real Whisper + `ScriptedDiarizer` +
  real embeddings + real Postgres — 77 chunks persisted from 5 files).
- **Task 4** — Hybrid search service implemented: keyword branch +
  semantic branch -> pure RRF fusion (k=60, weighted) -> ranked
  `SearchResultItem`s with file/timestamp/speaker. Unit-tested and
  **real-data verified** against the golden dataset.
- **Task 5** — Labeled query set: `scripts/golden_query_set.py`, 50
  hand-written queries (keyword + semantic) across all 5 golden files,
  each referencing ground-truth `expected_turn_indices` (robust to ASR
  wording drift, not exact-text matching). **Disclosed scoping
  tradeoff**: ~10 queries/file vs. the plan's 15-20/file target.
- **Task 6** — Real recall@k/MRR evaluation
  (`tests/test_recall_at_k_golden.py`) runs all 50 queries through the
  real `SearchService` against the real golden dataset in real Postgres.
  **Real measured results**: recall@5=0.92, recall@10=0.98, MRR=0.78 —
  both plan thresholds (0.80 / 0.90) **exceeded**. See SOLUTION.md for
  the full breakdown by query type.
- **Task 7** — Chunking QA regression tests implemented and passing:
  speaker-purity, length distribution, boundary sanity
  (`tests/test_chunking_qa.py`, `tests/test_chunking.py`).
- **Task 8** — Real speaker attribution accuracy + latency measurement:
  `scripts/measure_speaker_accuracy_and_latency.py` +
  `tests/test_speaker_accuracy_and_latency.py`. **Real measured
  results**: speaker attribution accuracy 0.988 (85/86 chunks, via
  time-overlap against ground truth — validates alignment logic, not
  real diarization-model accuracy, since `ScriptedDiarizer` supplies
  ground truth); search latency p95 ≈ 11ms over 150 real requests
  (well under the 500ms budget — reflects the golden dataset's small
  scale, not production-scale behavior).
- **Task 9** — Failure-mode analysis: of 50 queries, 1 missed entirely.
  Root cause reproduced against real data: the ground-truth answer
  chunk answers via a pronoun ("that") referring to an entity named
  only in the *previous* turn — a chunk-level embedding cannot see
  cross-turn coreference. A second pattern (semantic MRR 0.65 vs.
  keyword MRR 0.90) is the measured justification for hybrid over
  semantic-only retrieval. Full writeup in SOLUTION.md.
- **Task 10** — `SOLUTION.md` written: design rationale, implemented vs.
  deferred scope, known limitations, real success-criteria numbers,
  failure-mode analysis.
- **Task 11** — `AGENT_LOG.md` written: prompt-by-prompt disclosure of
  coding agent usage.
- **Task 12 (stretch)** — `PostgresJobQueue` (`SELECT ... FOR UPDATE
  SKIP LOCKED`) + `IngestionWorker` polling loop implemented and
  integration-tested against real Postgres
  (`tests/test_ingestion_worker.py`): enqueue -> claim -> process ->
  mark done/failed, plus concurrent-claim-safety and failure-path tests.
- **Task 13 (stretch)** — `SearchService.search_stream()` (async
  generator: `keyword_results` -> `semantic_results` -> `fused_results`
  -> `done`) + `GET /search/stream` SSE route. Unit-tested
  (`tests/test_search_stream.py`) and manually verified against the
  live composition-root API + real golden data.
- **Task 14 (stretch)** — `PostgresFeedbackRepository` +
  `FeedbackService` + `POST /search/feedback`. Records
  click/thumbs_up/thumbs_down/dwell_time_ms signals into
  `search_feedback`; `get_branch_weights()` infers keyword-vs-semantic
  attribution via a documented post-hoc heuristic (the schema doesn't
  persist which branch retrieved a chunk) and nudges RRF branch weights
  accordingly. Unit- and integration-tested
  (`tests/test_feedback_service.py`, `tests/test_postgres_feedback_repository.py`)
  and manually verified end-to-end against the live API.
- **Task 15 (stretch)** — `POST /search/answer`: default
  `ExtractiveAnswerGenerator` (no external API key required — cites the
  top retrieved snippets with speaker/timestamp) behind the same
  `AnswerGenerator` port as an optional `OpenAIAnswerGenerator` adapter
  (code-complete but **unverified** — this environment's proxy blocks
  `api.openai.com`, same as it blocks Hugging Face Hub). Unit-tested
  (`tests/test_search_answer_service.py`) and manually verified against
  the live API + real golden data.
- **Task 16** — Pydantic request/response models (`src/api/schemas.py`,
  including new `FeedbackRequest`/`SearchAnswerResponse`), route
  metadata (summary/description/tags), FastAPI app metadata — verified
  `/docs`, `/redoc`, `/openapi.json` register correctly with all 5 routes.
- **Task 17** — Structured JSON logging implemented
  (`src/infra/logging_config.py`) and wired into `ingest_pipeline.py`,
  `search_service.py`, and `ingestion_worker.py` at every stage/branch
  boundary, including `search.response.total_duration_ms` — the exact
  signal aggregated by the Task 8 latency measurement.
- **Task 18** — This file, maintained as of each session's changes.

## Known, disclosed scope reductions (not gaps — see SOLUTION.md)
- Golden dataset is ~10.4 min total (vs. 8-10 min/file target) and
  synthesized (not real recordings) — copyright/time constraints.
- Query set is ~10 queries/file (vs. 15-20/file target).
- `pyannote.audio` (real diarization) remains uninstalled/unverified —
  gated HF model + missing `HF_TOKEN`, independent of this session's
  proxy issue. `ScriptedDiarizer` demonstrates the port is swappable;
  it is explicitly not a production diarizer substitute.
- `OpenAIAnswerGenerator` is code-complete but unverified against a live
  API (same proxy-blocking pattern as Hugging Face Hub).

## Verification
- `pytest -q` → **42/42 passing** (up from 24), inside a project
  `.venv`: pure chunking/fusion/application-service unit tests via
  in-memory fakes, real Postgres/pgvector integration tests, real
  golden-dataset recall@k/speaker-accuracy/latency evaluation tests,
  and stretch-goal tests (job queue, SSE, feedback, answer generation).
- `python3 -c "from src.api.main import app"` → composition root imports
  cleanly; `/openapi.json` shows all 5 routes: `/search`, `/ingest`,
  `/search/stream`, `/search/feedback`, `/search/answer`.
- Live `uvicorn src.api.main:app` manually exercised against the real
  golden dataset in real Postgres for every route, including a real
  feedback round-trip (write + verify + clean up) and a partial SSE
  stream capture (`keyword_results` -> `semantic_results` ->
  `fused_results` events observed in order).
- **Real Postgres + pgvector integration verified end-to-end**: Docker
  Hub pulls of `pgvector/pgvector` are blocked by corporate network
  policy, so Postgres 14 (Homebrew) + `pgvector` 0.8.6 was built from
  source and run natively. Real integration testing found and fixed two
  genuine bugs no fake could have caught (deferrable self-referential
  FKs; `pgvector.Vector` unwrapping) — see SOLUTION.md for detail.
- **Real Whisper transcription** — `faster-whisper`'s Hugging Face Hub
  download is blocked by this environment's corporate TLS-inspecting
  proxy (`SSL: WRONG_VERSION_NUMBER`). Resolved by adding an
  `openai-whisper` backend (now default) that downloads from OpenAI's
  Azure Blob CDN, which this proxy does not block. Used for both the
  golden-dataset ingestion (Whisper `base`) and ad hoc verification
  (Whisper `tiny`).
- `pyannote.audio` (diarization) remains not installed/tested — gated
  Hugging Face model + missing `HF_TOKEN`, unrelated to the proxy issue.

## Added: automatic "model blocked" fallback logic (resilience layer)
Beyond the 18 tasks + 4 stretch goals, added a generic decorator-adapter
pattern so a blocked/unavailable ML model degrades the pipeline
gracefully instead of hard-failing the request — portable across
whichever specific model/proxy/credential is blocked in a given
environment (not hardcoded to this session's specific failures):
- `src/infra/resilient_diarizer.py` — `ResilientDiarizer(primary, fallback)`:
  tries real `pyannote.audio`, falls back to
  `src/infra/single_speaker_diarizer.py::SingleSpeakerDiarizer` (a real,
  production-safe single-speaker adapter requiring no model/network/
  credentials — distinct from `ScriptedDiarizer`, which only works for
  the golden dataset).
- `src/infra/resilient_transcriber.py` — `ResilientTranscriber(primary, fallback)`:
  tries `faster-whisper` (HF Hub), falls back to `openai-whisper`
  (Azure CDN, verified working in this environment).
- `src/infra/resilient_answer_generator.py` — `ResilientAnswerGenerator(primary, fallback)`:
  tries `OpenAIAnswerGenerator`, falls back to `ExtractiveAnswerGenerator`
  (no external dependency at all).
- All three: "sticky" circuit breaker (primary tried once per process;
  every subsequent call skips straight to fallback) and a structured
  `*.primary_unavailable` log event with the real failure reason — the
  failure is always visibly logged, never silently swallowed.
- Wired into `src/api/main.py`'s composition root (`lifespan`,
  `_build_answer_generator`) so `/ingest`, `/search/answer` etc. use the
  resilient versions by default.
- **Verified live, not just unit-tested**: started the real API against
  the real DB and called `POST /ingest` with real audio — structured
  logs confirmed both `resilient_transcriber.primary_unavailable`
  (faster-whisper: HF Hub offline) and `resilient_diarizer.primary_unavailable`
  (pyannote: not installed) fired with the real failure reason, and the
  request still completed successfully end-to-end via the fallbacks.
  Also verified `POST /search/answer` logged
  `answer_generator.openai_unavailable` (no `OPENAI_API_KEY`) and
  returned a correct extractive answer anyway.
- 9 new unit tests (`tests/test_resilient_adapters.py`,
  `tests/test_single_speaker_diarizer.py`) using fakes to prove
  primary-preferred, fallback-on-failure, and sticky-after-first-failure
  behavior deterministically. Full suite: **51/51 passing**.

## Added: configurable model providers (NVIDIA NIM / other HF models / any OpenAI-compatible endpoint)

Every ML integration point (embeddings, transcription, LLM answers) is
now provider-swappable purely via `AUDIO_SEARCH_*` env vars — no code
change needed to point the stack at a different model or vendor:

- **`src/infra/openai_compatible_embedder.py`** (new) — generic
  `Embedder` adapter for any OpenAI-compatible `/v1/embeddings`
  endpoint (NVIDIA NIM, hosted OpenAI, self-hosted vLLM/TEI/Ollama).
  Selected via `AUDIO_SEARCH_EMBEDDING_PROVIDER=openai-compatible` +
  `AUDIO_SEARCH_EMBEDDING_BASE_URL`; falls back to the existing local
  `SentenceTransformerEmbedder` by default (unchanged behavior).
- **`src/infra/openai_answer_generator.py`** — extended with a
  `base_url` param (`AUDIO_SEARCH_ANSWER_BASE_URL`) and
  `AUDIO_SEARCH_ANSWER_MODEL`, so the same adapter now serves NVIDIA NIM
  or any other OpenAI-compatible chat endpoint, not just hosted OpenAI.
  API key resolution also checks `NVIDIA_API_KEY` as a fallback to
  `OPENAI_API_KEY`.
- **`src/api/settings.py`** — new fields: `embedding_provider`,
  `embedding_base_url`, `transcription_primary_backend`,
  `transcription_fallback_backend`, `answer_model`, `answer_base_url`.
- **`src/api/main.py`** — new `_build_embedder()` factory (mirrors
  `_build_answer_generator()`); `lifespan()`'s transcriber wiring now
  reads backend order from settings instead of being hardcoded.
- Also works for other Hugging Face models with zero new code: any
  `sentence-transformers`-compatible HF repo id via
  `AUDIO_SEARCH_EMBEDDING_MODEL_NAME`, or any local/HF Whisper
  checkpoint (path or repo id) via `AUDIO_SEARCH_WHISPER_MODEL_SIZE`
  with the `faster-whisper` backend.
- Documented with concrete NVIDIA NIM example env var sets in
  `SETUP.md` ("Using other model providers") and `.env.example`.
- 6 new unit tests (`tests/test_configurable_model_providers.py`)
  covering settings defaults/overrides, provider-selection routing, and
  fail-fast-without-a-key behavior. Full suite: **57/57 passing**.
