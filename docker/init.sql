-- Bootstraps the local pgvector container for fs-explorer / DSPy spike.
-- Kept in sync with src/fs_explorer/dspy_spike/schema.sql (minimal pages tier).

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS corpora (
  id            TEXT PRIMARY KEY,
  root_path     TEXT NOT NULL UNIQUE,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS documents (
  id              TEXT PRIMARY KEY,
  corpus_id       TEXT NOT NULL REFERENCES corpora(id),
  relative_path   TEXT NOT NULL,
  absolute_path   TEXT NOT NULL DEFAULT '',
  content         TEXT,
  metadata        JSONB NOT NULL DEFAULT '{}'::jsonb,
  file_mtime      DOUBLE PRECISION NOT NULL DEFAULT 0,
  file_size       BIGINT NOT NULL DEFAULT 0,
  content_sha256  TEXT NOT NULL DEFAULT '',
  page_count      INT,
  extractor       TEXT,
  last_indexed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  is_deleted      BOOLEAN NOT NULL DEFAULT FALSE,
  UNIQUE (corpus_id, relative_path)
);

CREATE TABLE IF NOT EXISTS pages (
  id              TEXT PRIMARY KEY,
  doc_id          TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  corpus_id       TEXT NOT NULL REFERENCES corpora(id),
  page_number     INT NOT NULL CHECK (page_number >= 1),
  text            TEXT NOT NULL,
  text_sha256     TEXT NOT NULL DEFAULT '',
  char_count      INT NOT NULL DEFAULT 0,
  category_hints  TEXT[] NOT NULL DEFAULT '{}',
  page_date_hint  TEXT,
  UNIQUE (doc_id, page_number)
);

CREATE INDEX IF NOT EXISTS idx_spike_pages_corpus_fts
  ON pages USING GIN (to_tsvector('english', text));

CREATE INDEX IF NOT EXISTS idx_spike_pages_trgm
  ON pages USING GIN (text gin_trgm_ops);

CREATE INDEX IF NOT EXISTS idx_spike_documents_corpus
  ON documents (corpus_id)
  WHERE is_deleted = FALSE;
