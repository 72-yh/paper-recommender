# Current Operational State

Last updated: 2026-05-29

## Deployment

- App: `paper-recommender-72yh`
- URL: `https://paper-recommender-72yh.fly.dev/`
- Region: `sjc`
- Machine: `2872010f093248`, `shared-cpu-1x`, 1GB RAM
- Volume: `paper_recommender_data`, 4GB
- Public networking: shared IPv4 plus dedicated IPv6
- Idle policy: `min_machines_running = 0`, auto-start enabled, Machine stopped after verification when practical

No managed database, Redis, object store, GPU, extra Machine, extra region, or dedicated IPv4 is part of the current deployment.

## Data

- Current deployment artifact: 3,000,000 indexed papers after real OAI catch-up
  from `2016-01-27` to `2026-04-23`
- Current local artifact after daily update: 3,058,361 indexed papers up to OAI
  datestamp `2026-05-29`. This has not been uploaded to Fly yet.
- SQLite database: `paper_recommender_1m.db`, about 1.3GB on the Fly volume after
  the 3M catch-up, `paper_categories` lookup growth, and status count index
- Vector index: `vectors_1m_int8_mmap/`, about 2.38GB after adding clustered
  IVF mmap arrays for 3M serving
- Category lookup: `paper_categories`, a derived SQLite table for active indexed paper categories
- Status endpoint after deployment: 3,000,000 active papers and 3,000,000 indexed papers
- Last OAI datestamp in deployment: `2026-04-23`

## 3M Budget Path

The full-corpus target remains the same low-cost architecture: one small Fly Machine, local SQLite, local int8 mmap vectors, no managed database, and no managed vector service.

The completed 3M artifact set required moving the Fly volume from 2GB to 4GB.
The latest local preflight measured the 3M clustered IVF artifact set at
3,702,979,208 bytes including SQLite, base int8 mmap arrays, and clustered int8
mmap arrays. This remains within the reviewed 4GB volume, but leaves much less
headroom than the earlier exact-scan 2,469,075,200-byte artifact. Keep the next
growth step under explicit volume review.

Keeping `min_machines_running = 0` remains the main compute cost guardrail. If the 1GB `shared-cpu-1x` Machine were left running for a full month in the current region, the listed monthly compute price is still below the user budget, but the operational target is lower than always-on usage because expected traffic is about 100 users and the app can auto-stop.

## Search Path

The target serving path is a 3M `ivf_int8_mmap` index loaded from the same
`int8_mmap` directory format plus small IVF cluster files. FAISS and USearch are not currently deployed.

This means:

- Recommendation quality is still based on int8 cosine re-ranking over candidate vectors.
- Unfiltered recommendations search nearby IVF clusters instead of scanning all
  3M vectors.
- Category and date filters prefilter candidate `vector_id`s through indexed SQLite lookup tables, then score only that filtered vector subset.
- Category/date filters are intersected with nearby IVF clusters before int8
  re-ranking.

The `int8_mmap` format keeps int8 quantization and exact NumPy ranking
behavior, but stores precomputed row norms and allows the large arrays to be
memory-mapped instead of unpacked from one compressed file at process start.
The `ivf_int8_mmap` path keeps those same vectors and adds `centroids.npy`,
`cluster_ids.npy`, `cluster_offsets.npy`, `clustered_vector_ids.npy`,
`clustered_codes.npy`, and `clustered_row_norms.npy`. The clustered arrays
duplicate the compact int8 serving representation in cluster order so Fly reads
mostly contiguous slices instead of scattered mmap rows. It does not introduce a
managed vector database or float32 vector copy.

Daily OAI update orchestration now has a local wrapper:
`scripts/run_daily_update.py`. It runs OAI sync with `int8_mmap` output,
rebuilds local `ivf_int8_mmap` cluster files only when vectors changed, and runs
artifact preflight against the reviewed 3M/4GB budget. It intentionally stops
before any Fly upload or deploy step so production mutations stay explicit.

The lower-level sync script still accepts `--serving-index-kind int8_mmap` so
new/modified/deleted OAI records rebuild the same serving artifact format that
production already uses.

The same sync script accepts `--target-vector-count` for catch-up backfills.
Use it to grow from 1M in measured chunks while preserving OAI datestamp order.

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
- Local serving benchmark harness on the 1M `int8_mmap` artifact with 5 queries: load 0.0288s, unfiltered p50 532.645ms, unfiltered p95 788.231ms, filtered `cs.CL + cs.LG` p50 25.241ms, filtered p95 40.910ms.
- Local USearch f16 ANN evaluation on a 50k slice of the 1M `int8_mmap` artifact: build about 16.0s, load 0.048s, search p50 0.910ms, search p95 1.256ms, output 45,826,960 bytes, recall@10 0.9980 across 100 queries.
- Local USearch f16 recall@50 evaluation on the same 50k slice: build about 17.0s, load 0.044s, search p50 0.947ms, search p95 1.307ms, output 45,826,960 bytes, recall@50 0.9886 across 100 queries.
- Local USearch i8 ANN evaluation on the same 50k slice: output 26,626,960 bytes, search p95 under 0.7ms, recall@10 0.9130, and recall@50 0.9348 across 100 queries.
- Local controlled catch-up smoke run: starting from OAI datestamp `2016-01-27`,
  processed 987 records, embedded 50 new records with CUDA, rebuilt the
  `int8_mmap` serving artifact, and kept `last_datestamp=2016-01-27`.
- Local preflight after the 1,000,050 catch-up: active/indexed papers 1,000,050,
  `index_bytes=396021848`, `total_artifact_bytes=807211096`,
  `projected_total_artifact_bytes=2421512213`, and `max_volume_gb=4.0`.
- Local serving benchmark after the 1,000,050 catch-up with 5 queries: load
  0.0018s, unfiltered p50 461.968ms, unfiltered p95 526.376ms, filtered
  `cs.CL + cs.LG` p50 19.275ms, filtered p95 20.233ms.
- Local full catch-up run: processed 2,000,937 OAI records, embedded 1,999,950
  new records with CUDA, reached 3,000,000 active/indexed papers, advanced the
  OAI cursor to `2026-04-23`, rebuilt the `int8_mmap` serving artifact, and
  reported recall@10 0.9923 across 1,000 compression sample queries.
- Local preflight after the 3M catch-up: active/indexed papers 3,000,000,
  `db_bytes=1281073152`, `index_bytes=1188002048`,
  `total_artifact_bytes=2469075200`, category lookup rows 5,172,784, and
  `max_volume_gb=4.0`.
- Local serving benchmark after the 3M catch-up with 5 queries: load 0.0297s,
  unfiltered p50 1,564.969ms, unfiltered p95 1,780.531ms, filtered
  `cs.CL + cs.LG` p50 646.306ms, filtered p95 735.574ms.
- Local 3M API smoke test with FastAPI `TestClient`: `/health` returned 200,
  `/api/status` returned 3,000,000 active/indexed papers with
  `last_oai_datestamp=2026-04-23` and `index_kind=int8_mmap`, `/` returned 200,
  and `/api/recommend` for `0704.0004` with categories `cs.CL + cs.LG` returned
  10 results.
- Production 3M deployment: the Fly volume was extended to 4GB, the 3M SQLite
  database and `int8_mmap` index were uploaded by chunked archive transfer, and
  `scripts/smoke_deployment.py --timeout-seconds 180` passed with
  `indexed_papers=3000000`, `index_kind=int8_mmap`,
  `last_oai_datestamp=2026-04-23`, and 3 returned results for `0704.0004`.
- Production 3M status count fix: adding `idx_papers_status_count` on
  `(active, vector_id)` reduced remote status count queries from about 60-80s to
  about 0.13-0.16s each.
- Production 3M recommendation timing on `shared-cpu-1x`, 1GB RAM: a filtered
  recommendation for `0704.0004` with `cs.CL + cs.LG` returned 10 results in
  60.56s after the index was already loaded; an unfiltered recommendation
  returned 10 results in 69.07s, so current latency is about 60-70s. This confirms that 3M exact scan on the
  cheapest Fly runtime is correct but too slow for a polished user experience.
- Local 3M clustered IVF int8 build: `scripts/build_ivf_int8_index.py` built 512
  clusters from a 100,000-vector training sample in 17.842s and wrote
  1,194,791,304 bytes of IVF files, including the clustered int8 mmap arrays.
- Local preflight after clustered IVF: active/indexed papers 3,000,000,
  `db_bytes=1320185856`, `index_bytes=2382793352`,
  `total_artifact_bytes=3702979208`, category lookup rows 5,172,784, and
  `max_volume_gb=4.0`.
- Local 3M IVF int8 evaluation with `nprobe=32`: recall@10 0.9920 across 50
  sampled queries compared with exact `int8_mmap`; exact p50 1,401.497ms and
  exact p95 1,592.079ms; clustered IVF p50 110.694ms and p95 128.606ms.
- Local 3M serving benchmark with clustered `ivf_int8_mmap`, 5 queries: load
  0.0106s, unfiltered p50 122.280ms, unfiltered p95 201.648ms, filtered
  `cs.CL` p50 814.331ms, filtered p95 2,273.659ms.
- Production clustered IVF deployment: generated the large clustered mmap arrays
  directly on the Fly volume to avoid a slow 1.15GB SFTP upload, then deployed
  with `fly deploy --ha=false --local-only`.
- Production clustered IVF smoke: `scripts/smoke_deployment.py
  --timeout-seconds 180` returned `indexed_papers=3000000`,
  `index_kind=ivf_int8_mmap`, `last_oai_datestamp=2026-04-23`, and 3 returned
  results for `0704.0004`.
- Production clustered IVF timing on `shared-cpu-1x`, 1GB RAM: an unfiltered
  recommendation for `0704.0004` returned 10 results in 0.572s; the same query
  with categories `cs.CL + cs.LG` returned 10 results in 8.405s. Filtered
  category search remains the next optimization target.
- Production filtered IVF candidate lookup fix: replacing repeated per-cluster
  `np.isin` checks with a one-time dense candidate ID lookup reduced the local
  filtered cluster-slice phase from 7.265s to 0.131s for `0704.0004` with
  `cs.CL + cs.LG`. End-to-end local filtered recommendation time was 0.572s.
- Production timing after the filtered lookup fix: the same `0704.0004` query
  with categories `cs.CL + cs.LG` returned 10 results in 1.880s, while the
  unfiltered query returned 10 results in 0.617s.
- Local daily wrapper no-op check: running `scripts/run_daily_update.py` with
  `--target-vector-count 3000000` fetched no OAI records, rebuilt no serving or
  IVF artifacts, and passed preflight with 3,000,000 indexed papers,
  `index_kind=ivf_int8_mmap`, and `total_artifact_bytes=3702979208`.
- Local daily OAI smoke run: running the same wrapper with `--max-records 50`
  processed 50 unchanged OAI records at datestamp `2026-04-23`, embedded 0
  papers, deleted 0 papers, rebuilt no serving or IVF artifacts, and passed
  preflight with `total_artifact_bytes=3702983304`. The exact vector and serving
  artifact timestamps stayed unchanged, confirming unchanged daily records do
  not rewrite large vector artifacts.
- Local full daily update: running `scripts/run_daily_update.py` without
  `--max-records` processed 61,303 OAI records, embedded 59,426 papers, deleted
  0 papers, advanced the local cursor to `2026-05-29`, rebuilt `int8_mmap` and
  `ivf_int8_mmap`, and passed preflight with 3,058,361 indexed papers and
  `total_artifact_bytes=3775515858`. The compression report
  `daily-full-20260529` measured recall@10 0.9910 across 200 sampled queries.
- Local API smoke after the full daily update: FastAPI `TestClient` returned
  `/api/status` with 3,058,361 indexed papers, `last_oai_datestamp=2026-05-29`,
  and `index_kind=ivf_int8_mmap`; `/api/recommend` for `0704.0004` filtered to
  `cs.CL + cs.LG` returned HTTP 200 with 10 results.

The cold-start number includes Machine auto-start and first index load. Warm recommendation is the more relevant number for repeated use after the process has loaded the index.

## Known Limits

- The deployed app covers 3M papers up to OAI datestamp `2026-04-23` and now
  uses clustered `ivf_int8_mmap` in production. The local artifact is newer
  than production after the 2026-05-29 daily update.
- The web UI always requests 10 recommendations and does not expose a Top K control.
- The web UI exposes a searchable multi-select category filter. Production `/api/categories` returned 168 categories in 2.032s on first call after the deployment, and multi-category recommendation filtering returned HTTP 200.
- First request after an idle stop can be slow.
- USearch f16 is fast on a small local slice, but its artifact size projects to roughly 916MB per 1M vectors and about 2.75GB for 3M vectors before SQLite and the existing int8 mmap index. That likely breaks the current 4GB full-corpus storage target unless it replaces, rather than duplicates, another artifact and passes a stricter quality gate.
- USearch i8 is smaller, projecting to roughly 1.6GB for 3M vectors, but the measured recall drop is too large to promote as the default path without more tuning.
- IVF should be monitored on production traffic for insufficient-result cases
  under very narrow filters.
- Daily sync should run on a local build machine and upload reviewed artifacts;
  the Fly Machine should remain a low-cost file-serving API runtime.

## Next Step

Review whether to upload the 3,058,361-paper local artifact to Fly. It still
fits the reviewed 4GB volume, but only with limited headroom, so use the
explicit chunked or in-volume update procedure and check billing before and
after any Fly session.
