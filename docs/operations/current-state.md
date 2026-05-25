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
- SQLite database: `paper_recommender_1m.db`, about 411MB locally after `paper_categories` backfill
- Vector index: `vectors_1m_int8_mmap/`, 396,002,048 bytes after conversion
- Category lookup: `paper_categories`, a derived SQLite table for active indexed paper categories
- Status endpoint after deployment: 1,000,000 active papers and 1,000,000 indexed papers
- Last OAI datestamp in the 1M proof: `2016-01-27`

The 1M proof is useful for serving and deployment validation, but newer arXiv IDs may be missing until the corpus is extended.

## 3M Budget Path

The full-corpus target remains the same low-cost architecture: one small Fly Machine, local SQLite, local int8 mmap vectors, no managed database, and no managed vector service.

The current 2GB volume may be too tight for roughly 3M papers. The latest local preflight measured the 1M proof artifact at 807,158,528 bytes after category lookup backfill and projected a 3M corpus at 2,421,475,584 bytes. The expected next storage step is therefore a 4GB volume, not a new paid service. At the current Fly volume price of `$0.15/GB/month`, moving from 2GB to 4GB would add about `$0.30/month`.

Keeping `min_machines_running = 0` remains the main compute cost guardrail. If the 1GB `shared-cpu-1x` Machine were left running for a full month in the current region, the listed monthly compute price is still below the user budget, but the operational target is lower than always-on usage because expected traffic is about 100 users and the app can auto-stop.

## Search Path

The current serving path is a 1M int8 NumPy index loaded from the `int8_mmap` directory format. FAISS is not currently deployed.

This means:

- Recommendation quality is still based on exact cosine comparison over int8 vectors.
- Unfiltered recommendations scan the local mmap `.npy` artifact set.
- Category and date filters prefilter candidate `vector_id`s through indexed SQLite lookup tables, then score only that filtered vector subset.
- ANN work remains a future optimization, not a completed part of the MVP.

The `int8_mmap` format keeps int8 quantization and exact NumPy ranking
behavior, but stores precomputed row norms and allows the large arrays to be
memory-mapped instead of unpacked from one compressed file at process start.

## Performance Evidence

- Cold start: after stopping the Fly Machine, two concurrent recommendation requests for `0704.0004` both returned HTTP 200 in about 22.6 seconds.
- Warm recommendation: after the index was already loaded, two concurrent recommendation requests both returned HTTP 200 in about 1.3 seconds.
- Smoke test: deployment smoke test returned `indexed_papers=1000000`, `index_kind=int8`, and `result_count=3` for `0704.0004`.
- Local artifact benchmark after adding `int8_mmap`: `.npz` load 3.079s and search 0.457s; mmap load 0.002s and search 0.544s on the local 1M artifact.
- Production `int8_mmap` smoke test: returned `indexed_papers=1000000`, `index_kind=int8_mmap`, `index_bytes=396002048`, and `result_count=3` for `0704.0004`.
- Production warm recommendation after the switch: 0.856s for `0704.0004` with 3 returned results.
- Production filtered recommendation after candidate prefiltering: 2.433s for `0704.0004` with categories `cs.CL + cs.LG`, returning 10 results.
- Production category lookup backfill after deploying `paper_categories`: first `/api/categories` call rebuilt the lookup in 35.63s for the 1M proof database; the next `/api/categories` call returned in 0.502s.
- Production warm filtered recommendation after indexed category lookup: 0.56s for `0704.0004` with categories `cs.CL + cs.LG`, returning 10 results.
- Local 3M storage projection preflight: `target_indexed_papers=3000000`, `projected_total_artifact_bytes=2421475584`, `max_volume_gb=4.0`, category lookup rows `1558846`.

The cold-start number includes Machine auto-start and first index load. Warm recommendation is the more relevant number for repeated use after the process has loaded the index.

## Known Limits

- The deployed proof covers 1M papers up to OAI datestamp `2016-01-27`, not all roughly 3M current arXiv records.
- The web UI always requests 10 recommendations and does not expose a Top K control.
- The web UI exposes a searchable multi-select category filter. Production `/api/categories` returned 168 categories in 2.032s on first call after the deployment, and multi-category recommendation filtering returned HTTP 200.
- First request after an idle stop can be slow.
- FAISS, USearch, or another ANN index still needs recall and latency evaluation before replacement.

## Next Step

Use this 1M int8 NumPy full-scan deployment as the baseline for Task 8. Evaluate FAISS, USearch, or another local ANN index against it with recall@10, recall@50, warm latency, cold-load cost, memory usage, and artifact size before changing the serving path.
