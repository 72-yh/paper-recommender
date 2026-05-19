# Real OAI Index Proof Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a small real-data path that fetches arXiv OAI-PMH records, embeds active records locally, writes SQLite metadata plus a local vector index, and lets the existing UI/API query it.

**Architecture:** Add a focused OAI HTTP client, a `BAAI/bge-small-en-v1.5` sentence-transformers embedder for real indexing, a dependency-free hashing fallback for tests, and a build script that applies OAI records through the existing pipeline. The script writes a fresh exact local vector index from current active DB rows so deleted and updated records do not leave stale recommendation candidates.

**Tech Stack:** Python standard library HTTP, SQLite, NumPy, sentence-transformers, existing FastAPI API/UI, pytest, ruff.

---

### Task 1: OAI Client

**Files:**
- Create: `src/paper_recommender/oai_client.py`
- Test: `tests/test_oai_client.py`

- [ ] Write tests for ListRecords URL construction and resumption-token pagination.
- [ ] Implement the minimal client using `urllib.request`.
- [ ] Verify with `pytest tests/test_oai_client.py -v`.

### Task 2: Local Text Embedder

**Files:**
- Create: `src/paper_recommender/embedding.py`
- Test: `tests/test_embedding.py`

- [ ] Write tests for deterministic hashing vectors and sentence-transformers model loading.
- [ ] Implement `BAAI/bge-small-en-v1.5` as the default real embedder and keep hashing as a fallback.
- [ ] Verify with `pytest tests/test_embedding.py -v`.

### Task 3: Build Index Script

**Files:**
- Modify: `src/paper_recommender/storage.py`
- Create: `src/paper_recommender/index_builder.py`
- Create: `scripts/build_oai_index.py`
- Test: `tests/test_index_builder.py`

- [ ] Write tests that build an index from fake OAI XML and query recommendations.
- [ ] Add storage helpers for vector id assignment and active vector rows.
- [ ] Implement the builder and CLI.
- [ ] Add a `--reset` CLI option for replacing sample data with a clean real-data proof index.
- [ ] Verify with `pytest tests/test_index_builder.py -v`.

### Task 4: Final Verification

- [ ] Run `pytest -v`.
- [ ] Run `ruff check .`.
- [ ] Run the build script against a small real OAI `datestamp` window.
- [ ] Call `/api/recommend` against the generated index.

### Task 5: Compression Evaluation

**Files:**
- Create: `src/paper_recommender/compressed_vector_store.py`
- Create: `scripts/evaluate_compression.py`
- Test: `tests/test_compressed_vector_store.py`
- Test: `tests/test_evaluate_compression_script.py`

- [ ] Add a PCA + int8 scalar-quantized index for evaluation artifacts.
- [ ] Add `recall@k` comparison against the float32 exact baseline.
- [ ] Keep the API on the exact index until compressed recall is measured on a larger sample.
