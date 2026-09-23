# Audio Search — Hybrid Keyword + Semantic Retrieval

A hybrid (BM25-style keyword + vector semantic, fused via Reciprocal
Rank Fusion) search system over transcribed, diarized audio, built to
the layered SOLID architecture (`domain/` → `infra/` → `application/` →
`api/`) described in `.github/prompts/plan-audioSearchHybridRetrieval.prompt.md`.

## Where to look

| Doc | Purpose |
|---|---|
| **[SETUP.md](SETUP.md)** | Practical "clone it and run it" guide — Postgres/pgvector setup, Python env, config, running tests, running the API, troubleshooting. **Start here.** |
| **[SOLUTION.md](SOLUTION.md)** | Design rationale, architecture, real evaluation numbers (recall@k, MRR, speaker accuracy/latency), known limitations. |
| **[PROGRESS.md](PROGRESS.md)** | Task-by-task completion status against the original plan, including the resilience/fallback layer. |
| **[AGENT_LOG.md](AGENT_LOG.md)** | Chronological log of implementation decisions made while building this. |

## Quick start

```bash
docker compose up -d                 # Postgres 16 + pgvector
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,transcription]"
cp .env.example .env
pytest -q
uvicorn src.api.main:app --reload --port 8000
```
See `SETUP.md` for the full walkthrough, including a Docker-free
fallback path and a zero-dependency demo mode (`scripts/demo_server.py`).

## Highlights

- **Hybrid search**: keyword (Postgres full-text) + semantic
  (`sentence-transformers` + `pgvector`) fused with Reciprocal Rank
  Fusion, exposed via `GET /search`, `GET /search/stream` (SSE), and
  `POST /search/answer` (extractive/LLM answer generation).
- **Ingestion pipeline**: `POST /ingest` — real Whisper transcription,
  speaker diarization, turn alignment, chunking, embedding, indexing.
- **Resilience layer**: every ML-model integration point
  (transcription, diarization, answer generation) is wrapped in a
  sticky circuit-breaker fallback adapter
  (`src/infra/resilient_*.py`) — if a model is blocked or unavailable
  for any reason, the request still completes via a working fallback
  and a clear structured log line reports why.
- **51/51 tests passing**, including real Postgres/pgvector
  integration tests and a golden-dataset evaluation suite (recall@k,
  MRR, speaker attribution accuracy, latency).
