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
- [x] Extend the Fly volume to 4GB, deploy the 3M artifacts, and verify
  `/api/status` reports 3,000,000 active and indexed papers with
  `index_kind=int8_mmap`.

### Task 7: Current Serving Performance Baseline

**Files:**
- Create: `docs/operations/current-state.md`
- Test: `tests/test_documentation.py`

- [x] Record the current deployed serving path as 1M int8 NumPy full-scan.
- [x] Record the updated deployed serving path as 3M int8 mmap NumPy full-scan.
- [x] Record that FAISS is not currently deployed.
- [x] Record the cold-start recommendation behavior after Machine auto-start.
- [x] Record the warm recommendation behavior after the index has loaded.
- [x] Keep the Fly Machine stopped after verification to avoid unnecessary compute time.

### Task 8: ANN Serving Index Evaluation

**Status:** Completed.

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

### Task 10: Controlled Full-Corpus Catch-Up

**Status:** Started.

**Goal:** Extend the local 1M proof toward the full corpus in measured chunks
without changing Fly resources or skipping OAI datestamps.

- [x] Add `--target-vector-count` support to `scripts/sync_serving_index.py` so
  catch-up jobs can stop at explicit corpus sizes.
- [x] Run a small real OAI catch-up from the saved `2016-01-27` cursor with CUDA
  embedding and `int8_mmap` output.
- [x] Run artifact preflight after the catch-up and record timing/counts.
- [x] Run the full 3M catch-up with CUDA embedding and `int8_mmap` output.
- [x] Run artifact preflight after the full 3M catch-up and record timing/counts.
- [x] Smoke test the local 3M API path.
- [x] Commit code/documentation changes separately from ignored local data.

**Small catch-up result:** The first real catch-up target was 1,000,050 vectors.
It processed 987 OAI records from `2016-01-27`, embedded 50 new records on CUDA,
rebuilt `data/vectors_1m_int8_mmap`, and reported recall@10 `0.9870` on a
100-query compression sample. Preflight passed with 1,000,050 active/indexed
papers and projected 3M artifact bytes `2421512213` under the 4GB review limit.

**3M catch-up result:** The full local catch-up target reached 3,000,000 vectors.
It processed 2,000,937 OAI records, embedded 1,999,950 new records on CUDA,
advanced the OAI cursor to `2026-04-23`, rebuilt `data/vectors_1m_int8_mmap`,
and reported recall@10 `0.9923` on 1,000 compression sample queries. Preflight
passed with 3,000,000 active/indexed papers, total artifact bytes `2469075200`,
category lookup rows `5172784`, and `max_volume_gb=4.0`.

**3M local API smoke:** FastAPI `TestClient` returned 200 for `/health`, 200 for
`/`, 3,000,000 active/indexed papers from `/api/status`, and 10 recommendations
for `0704.0004` with categories `cs.CL + cs.LG`.

### Task 11: 3M Fly Deployment Baseline

**Status:** Completed.

**Goal:** Put the completed 3M artifact on the reviewed low-cost Fly deployment
without adding managed services.

- [x] Extend `paper_recommender_data` from 2GB to 4GB after explicit review.
- [x] Upload the 3M SQLite database and `int8_mmap` artifact by chunked archive
  transfer after direct SFTP proved unreliable for the 1.28GB DB file.
- [x] Add `idx_papers_status_count` on `(active, vector_id)` so `/api/status`
  avoids slow full-table scans on the Fly volume.
- [x] Run deployment smoke with `--timeout-seconds 180`.
- [x] Record that 3M exact-scan recommendation latency on the current
  `shared-cpu-1x`, 1GB RAM runtime is about 60-70s and needs a performance
  follow-up before this is a polished UX.

**3M Fly smoke:** `scripts/smoke_deployment.py --timeout-seconds 180` returned
`indexed_papers=3000000`, `index_kind=int8_mmap`,
`last_oai_datestamp=2026-04-23`, and 3 results for `0704.0004`.

### Task 12: IVF int8_mmap Serving Index

**Status:** Started.

**Goal:** Reduce 3M recommendation latency without adding a managed vector
database or storing a second full vector copy.

- [x] Add `ivf_int8_mmap` serving support on top of the existing int8 mmap
  arrays.
- [x] Add `scripts/build_ivf_int8_index.py` to build `centroids.npy`,
  `cluster_ids.npy`, and clustered int8 mmap arrays.
- [x] Add `scripts/evaluate_ivf_int8_index.py` to compare IVF results against
  exact `int8_mmap` search.
- [x] Build 512 local IVF clusters on the 3M artifact.
- [x] Measure local recall and latency.
- [x] Upload/generate IVF cluster files on Fly and smoke test production.

**Local 3M clustered IVF result:** Building 512 clusters from a 100,000-vector
sample took 17.842s and wrote 1,194,791,304 bytes of IVF files, including the
clustered int8 mmap arrays. Preflight passed with total artifact bytes
`3702979208` under the reviewed 4GB volume. With `nprobe=32`, recall@10 was
0.9920 across 50 sampled queries against exact `int8_mmap`; clustered IVF p50
was 110.694ms and p95 was 128.606ms while exact p50 was 1,401.497ms. The
5-query serving benchmark with clustered `ivf_int8_mmap` reported unfiltered p50
122.280ms.

**Production clustered IVF result:** The small IVF files were uploaded by SFTP
and the large clustered mmap arrays were generated directly on the Fly volume to
avoid a slow 1.15GB SFTP upload. `fly deploy --ha=false --local-only` succeeded,
and deployment smoke returned `indexed_papers=3000000`,
`index_kind=ivf_int8_mmap`, `last_oai_datestamp=2026-04-23`, and 3 results for
`0704.0004`. Warm production timing on the current `shared-cpu-1x`, 1GB RAM
Machine was 0.572s for an unfiltered `0704.0004` recommendation and 8.405s for
the same query filtered to `cs.CL + cs.LG`. The Machine was stopped after
verification.

### Task 13: Filtered IVF Candidate Lookup

**Status:** Completed.

**Goal:** Reduce category-filtered recommendation latency without changing
storage artifacts or adding services.

- [x] Profile filtered recommendation phases on the local 3M artifact.
- [x] Add a regression test proving clustered slice filtering does not repeat
  `np.isin` over the full candidate ID list for each selected cluster.
- [x] Replace repeated `np.isin` with a one-time candidate ID lookup used by
  each clustered slice.
- [x] Deploy the code-only optimization to Fly with `--local-only`.
- [x] Smoke test production and stop the Fly Machine after verification.

**Filtered lookup result:** For `0704.0004` with categories `cs.CL + cs.LG`,
local profiling showed SQLite candidate lookup at 0.338s, candidate row mapping
at 0.084s, cluster selection at 0.006s, and clustered slice scoring at 0.131s
after the optimization. The local end-to-end filtered recommendation took
0.572s. Production timing on `shared-cpu-1x`, 1GB RAM was 1.880s for the
filtered request and 0.617s for the unfiltered request.
