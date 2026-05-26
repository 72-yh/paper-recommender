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

- [x] Write tests for ListRecords URL construction and resumption-token pagination.
- [x] Implement the minimal client using `urllib.request`.
- [x] Verify with `pytest tests/test_oai_client.py -v`.

### Task 2: Local Text Embedder

**Files:**
- Create: `src/paper_recommender/embedding.py`
- Test: `tests/test_embedding.py`

- [x] Write tests for deterministic hashing vectors and sentence-transformers model loading.
- [x] Implement `BAAI/bge-small-en-v1.5` as the default real embedder and keep hashing as a fallback.
- [x] Verify with `pytest tests/test_embedding.py -v`.

### Task 3: Build Index Script

**Files:**
- Modify: `src/paper_recommender/storage.py`
- Create: `src/paper_recommender/index_builder.py`
- Create: `scripts/build_oai_index.py`
- Test: `tests/test_index_builder.py`

- [x] Write tests that build an index from fake OAI XML and query recommendations.
- [x] Add storage helpers for vector id assignment and active vector rows.
- [x] Implement the builder and CLI.
- [x] Add a `--reset` CLI option for replacing sample data with a clean real-data proof index.
- [x] Verify with `pytest tests/test_index_builder.py -v`.

### Task 4: Final Verification

- [x] Run `pytest -v`.
- [x] Run `ruff check .`.
- [x] Run the build script against a small real OAI `datestamp` window.
- [x] Call `/api/recommend` against the generated index.

### Task 5: Compression Evaluation

**Files:**
- Create: `src/paper_recommender/compressed_vector_store.py`
- Create: `scripts/evaluate_compression.py`
- Test: `tests/test_compressed_vector_store.py`
- Test: `tests/test_evaluate_compression_script.py`

- [x] Add an int8 scalar-quantized index for evaluation artifacts.
- [x] Add `recall@k` comparison against the float32 exact baseline.
- [x] Keep the API on the exact index until compressed recall is measured on a larger sample.
- [x] Promote the 1M proof to the int8 serving path after local evaluation showed acceptable quality for the proof stage.

### Task 6: Container And Fly Deployment

**Files:**
- Create: `fly.toml`
- Create: `docs/deployment/fly-low-cost.md`
- Modify: `pyproject.toml`
- Test: `tests/test_deployment_config.py`

- [x] Add a Docker/Fly configuration that uses one small Machine and one mounted volume.
- [x] Keep the Docker image code-only and upload SQLite/vector artifacts to the Fly volume.
- [x] Use `fly deploy --app paper-recommender-72yh --ha=false --local-only` so no remote builder resource is created.
- [x] Package static UI assets so `/` works after deployment.
- [x] Deploy the 1M proof artifacts to `https://paper-recommender-72yh.fly.dev/`.
- [x] Verify `/api/status` reports 1,000,000 active and indexed papers with `index_kind=int8`.

### Task 7: Current Serving Performance Baseline

**Files:**
- Create: `docs/operations/current-state.md`
- Test: `tests/test_documentation.py`

- [x] Record the current deployed serving path as 1M int8 NumPy full-scan.
- [x] Record that FAISS is not currently deployed.
- [x] Record the cold-start recommendation behavior after Machine auto-start.
- [x] Record the warm recommendation behavior after the index has loaded.
- [x] Keep the Fly Machine stopped after verification to avoid unnecessary compute time.

### Task 8: ANN Serving Index Evaluation

**Status:** Started.

**Goal:** Replace the NumPy full-scan path only after measuring recall and latency against the current exact/int8 baseline.

- [x] Build a benchmark harness over the existing 1M proof artifacts.
- [x] Compare FAISS, USearch, or another local ANN index against the current NumPy full-scan path.
- [ ] Measure recall@10, recall@50, cold-load cost, warm latency, artifact size, and memory usage.
- [ ] Promote an ANN path only if it improves latency without an unacceptable quality drop.

**USearch local evaluation note:** Added `scripts/evaluate_ann.py` with an optional
`usearch` dependency. On a 50k slice of the 1M `int8_mmap` artifact, USearch f16
reported recall@10 `0.9980`, recall@50 `0.9886`, p95 search about `1.3ms`, and
an output artifact of `45,826,960` bytes. USearch i8 was faster and smaller at
`26,626,960` bytes for 50k vectors, but recall dropped to recall@10 `0.9130`
and recall@50 `0.9348`. The production path remains unchanged until memory
usage and full-corpus storage impact are acceptable.

### Task 9: Incremental int8_mmap Serving Sync

**Status:** Started.

**Goal:** Keep the daily OAI update path aligned with the currently deployed
`int8_mmap` serving artifact so future new/modified/deleted records do not need
a manual conversion step.

- [x] Add `--serving-index-kind int8_mmap` support to `scripts/sync_serving_index.py`.
- [x] Convert the rebuilt int8 `.npz` candidate into mmap `.npy` files and
  record the final mmap artifact path/size in the compression report.
- [x] Document the daily sync command for local build machines.
- [x] Verify with focused sync tests, full pytest, and ruff.
