# Fly.io Low-Cost Deployment Runbook

This runbook keeps the Paper Recommender MVP within a small monthly budget. It
is intentionally conservative: one public API app, one small Machine, one small
volume, and no managed services.

## Cost Guardrails

- Use exactly one `shared-cpu-1x` Machine with 1GB RAM.
- Keep `min_machines_running = 0` and `auto_stop_machines = "stop"` so the
  Machine can stop when idle.
- Use one 4GB volume named `paper_recommender_data`.
- Do not run `fly ips allocate-v4`; the shared IPv4 and Anycast IPv6 addresses
  are enough for this MVP.
- Do not enable metrics-based autoscaling.
- Do not create Managed Postgres, Redis, Tigris, or other managed services.
- Do not create extra regions, extra Machines, GPUs, Kubernetes, or support
  plans.
- Do not use a remote builder for this small deployment; use local Docker with
  `--local-only` to avoid builder resources.
- Check Fly Dashboard > Billing before and after every deploy session.

As of 2026-05-25, Fly's public pricing makes this a low-dollar deployment, but
it is not a hard billing cap. Treat the Fly dashboard's current-month usage as
the source of truth.

## Local Inputs

The Docker image contains only app code. The serving artifacts stay outside the
image and are uploaded to the Fly volume:

- `data/paper_recommender_1m.db`
- `data/vectors_1m_int8_mmap/`

For the no-new-resource load optimization, convert the int8 index locally to
`data/vectors_1m_int8_mmap/` and upload that directory instead. The current
serving target is `PAPER_RECOMMENDER_INDEX_KIND=ivf_int8_mmap`, which reuses
those mmap arrays and adds IVF metadata plus clustered int8 mmap arrays:
`centroids.npy`, `cluster_ids.npy`, `cluster_offsets.npy`,
`clustered_vector_ids.npy`, `clustered_codes.npy`, and
`clustered_row_norms.npy`.

USearch and other ANN indexes are local evaluation candidates only. Do not
upload an ANN artifact or add `usearch` to the production image unless a larger
evaluation shows that it fits the reviewed volume budget and preserves recall.
The first local 50k USearch f16 candidate was fast but projected to about
2.75GB for 3M vectors by itself, before SQLite and the existing int8 mmap files.

The deployed clustered 3M artifact is about 3.70GB including SQLite, base int8
mmap arrays, and clustered IVF arrays, so the current reviewed size is 4GB with
limited headroom. A larger backfill requires an explicit volume-size review
before upload.

Before any larger upload, run:

```powershell
.\.venv\Scripts\python.exe scripts\preflight_artifacts.py `
  --db-path data\paper_recommender_1m.db `
  --index-path data\vectors_1m_int8_mmap `
  --index-kind ivf_int8_mmap `
  --min-indexed-papers 3000000 `
  --target-indexed-papers 3000000 `
  --max-volume-gb 4
```

This checks DB/index consistency, verifies the indexed category lookup table,
and estimates whether the target corpus fits the reviewed volume budget.

## Daily Local Sync

Run daily OAI updates on a local build machine, not on the small Fly Machine.
The sync resumes from `last_successful_oai_datestamp`, embeds only changed
records, and rebuilds the serving artifact only when vectors changed:

```powershell
.\.venv\Scripts\python.exe scripts\sync_serving_index.py `
  --db-path data\paper_recommender_1m.db `
  --exact-index-path data\vectors_1m.npz `
  --serving-index-path data\vectors_1m_int8_mmap `
  --serving-index-kind int8_mmap `
  --target-vector-count 1010000 `
  --device cuda `
  --embedding-batch-size 512 `
  --checkpoint-every-records 10000 `
  --label daily-int8-mmap
```

Increase `--target-vector-count` in controlled steps, for example 1.01M, 1.1M,
then larger windows after timing is known. After a successful local sync, rebuild
the IVF cluster files and run artifact preflight before uploading the updated
SQLite database and mmap directory to Fly. This keeps production cheap: the
deployed app serves files and does not need GPU, background workers, or a
managed vector database.

The first small catch-up smoke run used target 1,000,050, processed 987 OAI
records from `2016-01-27`, embedded 50 new records on CUDA, and passed preflight
with a 3M projection still below the reviewed 4GB artifact budget. The completed
local 3M catch-up later reached 3,000,000 indexed papers, advanced the OAI cursor
to `2026-04-23`, and passed preflight with 2,469,075,200 total artifact bytes
before adding clustered IVF arrays. The clustered IVF preflight later measured
3,702,979,208 total artifact bytes under the same 4GB review limit.

After the 3M catch-up, build IVF cluster files:

```powershell
.\.venv\Scripts\python.exe scripts\build_ivf_int8_index.py `
  --index-path data\vectors_1m_int8_mmap `
  --n-clusters 512 `
  --train-sample-size 100000 `
  --iterations 6
```

Then evaluate:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_ivf_int8_index.py `
  --index-path data\vectors_1m_int8_mmap `
  --top-k 10 `
  --sample-size 50 `
  --nprobe 32
```

## Safe Preparation

These commands should not create paid Fly resources:

```powershell
fly auth login
fly version
```

Install `flyctl` first if `fly` is not on `PATH`.

## Cost-Incurring Commands

Run these only after confirming the app name, region, and current Fly pricing.
Use `--ha=false`; plain `fly launch` defaults to creating spare Machines for
availability.

```powershell
fly apps create paper-recommender-72yh
fly volumes create paper_recommender_data --app paper-recommender-72yh --region sjc --size 4
fly deploy --app paper-recommender-72yh --ha=false --local-only
```

After deploy, confirm there is one Machine and one volume:

```powershell
fly status --app paper-recommender-72yh
fly machine list --app paper-recommender-72yh
fly volumes list --app paper-recommender-72yh
```

## Artifact Upload

Upload the artifacts into the mounted `/app/data` volume:

```powershell
fly sftp put data/paper_recommender_1m.db /app/data/paper_recommender_1m.db --app paper-recommender-72yh
```

Optional mmap int8 upload after local conversion:

```powershell
fly sftp shell --app paper-recommender-72yh
```

Then create `/app/data/vectors_1m_int8_mmap` and upload `vector_ids.npy`,
`codes.npy`, `scales.npy`, and `row_norms.npy` into that directory. Keep the
existing `.npz` file on the volume until the mmap path has passed preflight and
smoke tests.

If direct `put` does not work on the installed `flyctl`, use an interactive
session instead:

```powershell
fly sftp shell --app paper-recommender-72yh
```

Then run the same two `put` operations inside the SFTP shell.

Machine may auto-stop during long SFTP uploads. If an upload disconnects near
the end, verify the file size on the volume before retrying. During the 1M proof
deployment, the database upload completed, the vector upload needed a retry, and
a lightweight `/health` keepalive helped keep the Machine available during the
second upload.

For the 3M upload, direct SFTP of a 1.28GB SQLite file was not reliable enough.
The working chunked archive transfer path was:

1. Create a local compressed archive containing `paper_recommender_1m.db` and
   `vectors_1m_int8_mmap/`.
2. Split the archive into 64MB chunks.
3. Upload chunks to `/app/data/upload_3m_chunks/`.
4. Verify `wc -c /app/data/upload_3m_chunks/part_*` matches the local archive
   size.
5. Stop the keepalive, restart the Machine to release old file handles, remove
   the old 1M artifacts, then stream extract without creating another archive
   copy on the volume:

```sh
cat /app/data/upload_3m_chunks/part_* | tar -xzf - -C /app/data
```

6. Remove `/app/data/upload_3m_chunks/` after extraction.
7. Add the deployed SQLite status index if the uploaded database does not
   already have it:

```sql
CREATE INDEX IF NOT EXISTS idx_papers_status_count ON papers(active, vector_id);
```

For an IVF update, upload only the new `centroids.npy` and `cluster_ids.npy`
files into `/app/data/vectors_1m_int8_mmap/` if only centroids changed. If the
base mmap arrays changed, regenerate or upload all clustered IVF files:
`cluster_offsets.npy`, `clustered_vector_ids.npy`, `clustered_codes.npy`, and
`clustered_row_norms.npy`. For large full-corpus updates, generating the
clustered mmap arrays directly on the Fly volume can avoid a very slow SFTP
upload of `clustered_codes.npy`.

## Verification

Use the deployment smoke test against the Fly URL:

```powershell
.\.venv\Scripts\python.exe scripts\smoke_deployment.py `
  --base-url https://paper-recommender-72yh.fly.dev `
  --query-url https://arxiv.org/abs/0704.0004 `
  --expected-index-kind ivf_int8_mmap `
  --min-indexed-papers 3000000 `
  --timeout-seconds 180
```

Then check billing again:

```powershell
fly dashboard --app paper-recommender-72yh
```

The first recommendation requests can load the index from the mounted volume.
The 3M exact `int8_mmap` baseline was correct but slow on the current 1GB
`shared-cpu-1x` Machine: measured filtered and unfiltered requests were about
60-70s. The deployed clustered `ivf_int8_mmap` path reduced an unfiltered
`0704.0004` production recommendation to 0.572s. The same query filtered to
`cs.CL + cs.LG` took 8.405s, so category-filtered latency remains the next
serving optimization target.

## Cleanup

If the bill estimate looks wrong, stop before experimenting further. To remove
the deployment, delete the app and confirm no volumes or managed services remain
in the Fly dashboard:

```powershell
fly apps destroy paper-recommender-72yh
```
