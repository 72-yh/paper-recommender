# Current Operational State

Last updated: 2026-05-25

## Deployment

- App: `paper-recommender-72yh`
- URL: `https://paper-recommender-72yh.fly.dev/`
- Region: `sjc`
- Machine: `2872010f093248`, `shared-cpu-1x`, 1GB RAM
- Volume: `paper_recommender_data`, 2GB
- Public networking: shared IPv4 plus dedicated IPv6
- Idle policy: `min_machines_running = 0`, auto-start enabled, Machine stopped after verification when practical

No managed database, Redis, object store, GPU, extra Machine, extra region, or dedicated IPv4 is part of the current deployment.

## Data

- Current deployment artifact: 1M proof index, not the full current arXiv corpus
- SQLite database: `paper_recommender_1m.db`, about 210MB locally
- Vector index: `vectors_1m_int8.npz`, about 340MB locally
- Status endpoint after deployment: 1,000,000 active papers and 1,000,000 indexed papers
- Last OAI datestamp in the 1M proof: `2016-01-27`

The 1M proof is useful for serving and deployment validation, but newer arXiv IDs may be missing until the corpus is extended.

## Search Path

The current serving path is a 1M int8 NumPy full-scan index. FAISS is not currently deployed.

This means:

- Recommendation quality is still based on exact cosine comparison over the int8 vectors.
- Latency depends on loading and scanning the local `.npz` artifact.
- ANN work remains a future optimization, not a completed part of the MVP.

The next no-new-resource optimization is `int8_mmap`: convert
`vectors_1m_int8.npz` into a directory of `.npy` arrays and serve it with
`PAPER_RECOMMENDER_INDEX_KIND=int8_mmap`. This keeps int8 quantization and the
same full-scan ranking behavior, but stores precomputed row norms and allows the
large arrays to be memory-mapped instead of unpacked from one compressed file at
process start.

## Performance Evidence

- Cold start: after stopping the Fly Machine, two concurrent recommendation requests for `0704.0004` both returned HTTP 200 in about 22.6 seconds.
- Warm recommendation: after the index was already loaded, two concurrent recommendation requests both returned HTTP 200 in about 1.3 seconds.
- Smoke test: deployment smoke test returned `indexed_papers=1000000`, `index_kind=int8`, and `result_count=3` for `0704.0004`.
- Local artifact benchmark after adding `int8_mmap`: `.npz` load 3.079s and search 0.457s; mmap load 0.002s and search 0.544s on the local 1M artifact.

The cold-start number includes Machine auto-start and first index load. Warm recommendation is the more relevant number for repeated use after the process has loaded the index.

## Known Limits

- The deployed proof covers 1M papers up to OAI datestamp `2016-01-27`, not all roughly 3M current arXiv records.
- The web UI always requests 10 recommendations and does not expose a Top K control.
- First request after an idle stop can be slow.
- `int8_mmap` support is implemented for the next artifact conversion, but the
  currently deployed artifact is still the `.npz` int8 index until the converted
  directory is uploaded and the Fly env is switched.
- FAISS, USearch, or another ANN index still needs recall and latency evaluation before replacement.

## Next Step

Use this 1M int8 NumPy full-scan deployment as the baseline for Task 8. Evaluate FAISS, USearch, or another local ANN index against it with recall@10, recall@50, warm latency, cold-load cost, memory usage, and artifact size before changing the serving path.
