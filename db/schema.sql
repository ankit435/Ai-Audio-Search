-- Audio Search — Hybrid Retrieval: Postgres + pgvector schema
-- See .github/prompts/plan-audioSearchHybridRetrieval.prompt.md
--   "Data Model — Chunk" and "Production Considerations" / "Relevance Feedback Loop"

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto; -- gen_random_uuid()

-- Source audio files. One row per ingested file; checksum enables
-- idempotent re-ingestion (skip/replace on re-run of the same file).
CREATE TABLE IF NOT EXISTS audio_file (
    audio_file_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    file_name     TEXT NOT NULL,
    file_path     TEXT NOT NULL,
    checksum      TEXT NOT NULL UNIQUE,
    duration_s    NUMERIC,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Chunk: merge-and-split, speaker-bounded transcript segments.
-- embedding dimension (384) matches the default SentenceTransformer
-- model (e.g. all-MiniLM-L6-v2); adjust if a different model is chosen.
CREATE TABLE IF NOT EXISTS chunk (
    chunk_id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    audio_file_id  UUID NOT NULL REFERENCES audio_file(audio_file_id) ON DELETE CASCADE,
    speaker_id     TEXT NOT NULL,
    text           TEXT NOT NULL,
    start_time     NUMERIC NOT NULL,
    end_time       NUMERIC NOT NULL,
    embedding      vector(384),
    -- DEFERRABLE: a batch insert of a speaker-turn's chunks references
    -- sibling rows' chunk_id (prev/next) that don't exist yet within the
    -- same statement; deferring the FK check to COMMIT (see
    -- PostgresChunkRepository.save_chunks' explicit transaction) lets the
    -- whole batch land before constraints are validated.
    prev_chunk_id  UUID REFERENCES chunk(chunk_id) DEFERRABLE INITIALLY DEFERRED,
    next_chunk_id  UUID REFERENCES chunk(chunk_id) DEFERRABLE INITIALLY DEFERRED,
    token_count    INTEGER,
    char_count     INTEGER,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    tsv            tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    CONSTRAINT chunk_time_range_chk CHECK (end_time >= start_time)
);

CREATE INDEX IF NOT EXISTS chunk_tsv_gin_idx ON chunk USING GIN (tsv);
CREATE INDEX IF NOT EXISTS chunk_audio_file_id_idx ON chunk (audio_file_id);
-- ivfflat requires ANALYZE with data present; hnsw (pgvector >= 0.5) has no such requirement.
-- Choose one at index-build time once real embeddings are loaded, e.g.:
-- CREATE INDEX chunk_embedding_hnsw_idx ON chunk USING hnsw (embedding vector_cosine_ops);

-- ---------------------------------------------------------------------------
-- Stretch goal tables (optional; see plan "Production Considerations" and
-- "Relevance Feedback Loop"). Created up front so schema migrations aren't
-- needed if/when those features are implemented.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS ingestion_job (
    job_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    audio_file_path TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending'
                        CHECK (status IN ('pending', 'processing', 'done', 'failed')),
    attempts        INTEGER NOT NULL DEFAULT 0,
    last_error      TEXT,
    locked_by       TEXT,
    locked_at       TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS search_feedback (
    feedback_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_text  TEXT NOT NULL,
    chunk_id    UUID NOT NULL REFERENCES chunk(chunk_id) ON DELETE CASCADE,
    rank_shown  INTEGER NOT NULL,
    signal      TEXT NOT NULL CHECK (signal IN ('click', 'thumbs_up', 'thumbs_down', 'dwell_time_ms')),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
