# paper-recommender

## Large Local Index Builds

Use separate output files for large proofs so smaller smoke-test data is not
overwritten.

Recommended runtime split:

- Use the local NVIDIA GPU machine for large initial backfills and any large
  re-embedding jobs.
- Use a low-cost CPU server for the recommendation API. Serving uses the stored
  index and does not need a GPU.
- Run daily incremental OAI updates on CPU first. If a backlog grows or a large
  re-embedding is needed, run the same build command on the local GPU machine.

Before GPU builds, install a CUDA-enabled PyTorch wheel in the virtual
environment and verify `torch.cuda.is_available()` returns `True`. The project
exposes `--device auto|cpu|cuda`; use `--device cuda` for large local builds and
`--device cpu` for CPU-only update jobs.

Initial 1M build:

```powershell
.\.venv\Scripts\python.exe scripts\build_oai_index.py `
  --target-vector-count 1000000 `
  --device cuda `
  --request-delay-seconds 3 `
  --fetch-retries 10 `
  --fetch-retry-delay-seconds 120 `
  --embedding-batch-size 1024 `
  --checkpoint-every-records 25000 `
  --db-path data/paper_recommender_1m.db `
  --index-path data/vectors_1m.npz `
  --reset
```

Resume an interrupted 1M build:

```powershell
.\.venv\Scripts\python.exe scripts\build_oai_index.py `
  --target-vector-count 1000000 `
  --resume `
  --device cuda `
  --request-delay-seconds 3 `
  --fetch-retries 10 `
  --fetch-retry-delay-seconds 120 `
  --embedding-batch-size 1024 `
  --checkpoint-every-records 25000 `
  --db-path data/paper_recommender_1m.db `
  --index-path data/vectors_1m.npz
```

Build a full-dimension int8 index after the exact 1M index is complete:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_compression.py `
  --method int8 `
  --input data/vectors_1m.npz `
  --output data/vectors_1m_int8.npz `
  --top-k 10 `
  --sample-size 10000 `
  --label bge-small-1m-int8-r10
```

Convert the compressed `.npz` int8 index to mmap-ready `.npy` files when
optimizing cold-start load behavior on a low-memory server:

```powershell
.\.venv\Scripts\python.exe scripts\convert_int8_mmap.py `
  --input data/vectors_1m_int8.npz `
  --output data/vectors_1m_int8_mmap `
  --overwrite
```

Build the IVF cluster files on top of the existing `int8_mmap` directory when
optimizing 3M serving latency without adding a managed vector service. This
keeps int8 quantization and writes clustered mmap arrays for contiguous cluster
reads:

```powershell
.\.venv\Scripts\python.exe scripts\build_ivf_int8_index.py `
  --index-path data\vectors_1m_int8_mmap `
  --n-clusters 512 `
  --train-sample-size 100000 `
  --iterations 6
```

Then evaluate recall and latency against exact `int8_mmap` search:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_ivf_int8_index.py `
  --index-path data\vectors_1m_int8_mmap `
  --top-k 10 `
  --sample-size 50 `
  --nprobe 32
```

Serve the 1M int8 index locally:

```powershell
$env:PAPER_RECOMMENDER_DB_PATH='data/paper_recommender_1m.db'
$env:PAPER_RECOMMENDER_INDEX_PATH='data/vectors_1m_int8.npz'
$env:PAPER_RECOMMENDER_INDEX_KIND='int8'
.\.venv\Scripts\python.exe -m uvicorn paper_recommender.app:app --host 127.0.0.1 --port 8000
```

## Container Deployment

The container image packages only the API code. Keep generated artifacts outside
the image and mount them read-only at runtime:

- `data/paper_recommender_1m.db`
- `data/vectors_1m_int8.npz`

Compose project name is fixed to `paper_recommender` so generated local image,
container, and network names do not inherit the repository directory name.

Run locally with Docker Compose:

```powershell
docker compose up --build
```

Run the full local container verification flow:

```powershell
.\.venv\Scripts\python.exe scripts\verify_container_deployment.py
```

This runs artifact preflight, `docker compose config`, `docker compose build`,
`docker compose up -d`, `/health` polling, and the deployment smoke test.

For the low-cost Fly.io deployment path, follow
`docs/deployment/fly-low-cost.md` before running any Fly commands that create
apps, volumes, Machines, IPs, or managed services.

Validate the mounted artifacts before starting or shipping the container:

```powershell
.\.venv\Scripts\python.exe scripts\preflight_artifacts.py `
  --db-path data/paper_recommender_1m.db `
  --index-path data/vectors_1m_int8_mmap `
  --index-kind ivf_int8_mmap `
  --min-indexed-papers 3000000
```

The preflight check verifies the DB and index files, index format, row counts,
and vector IDs.

Smoke-test a running deployment:

```powershell
.\.venv\Scripts\python.exe scripts\smoke_deployment.py `
  --base-url http://127.0.0.1:8000 `
  --query-url https://arxiv.org/abs/0704.0004 `
  --expected-index-kind ivf_int8_mmap `
  --min-indexed-papers 3000000 `
  --timeout-seconds 180
```

The smoke test checks `/health`, `/api/status`, and one recommendation request.

Docker healthcheck support is built into both the image and Compose file. It
uses Python's standard library inside the container to call `/health`, so no
extra `curl` package is installed.

The compose file mounts `./data` to `/app/data` and configures:

```text
PAPER_RECOMMENDER_DB_PATH=/app/data/paper_recommender_1m.db
PAPER_RECOMMENDER_INDEX_PATH=/app/data/vectors_1m_int8.npz
PAPER_RECOMMENDER_INDEX_KIND=int8
```

The app also supports the mmap int8 directory format:

```text
PAPER_RECOMMENDER_INDEX_PATH=/app/data/vectors_1m_int8_mmap
PAPER_RECOMMENDER_INDEX_KIND=int8_mmap
```

After IVF cluster files are present in the same directory, use:

```text
PAPER_RECOMMENDER_INDEX_PATH=/app/data/vectors_1m_int8_mmap
PAPER_RECOMMENDER_INDEX_KIND=ivf_int8_mmap
```

For a low-cost server, copy or attach the DB plus the selected vector artifact to
the same paths, then run the same image. Do not bake the DB or vector index into
the image; the current deployed 3M `ivf_int8_mmap` artifact keeps the base
`int8_mmap` vectors and adds clustered int8 mmap files for faster reads on the
small Fly volume.

Daily serving-index sync after a full current backfill:

```powershell
.\.venv\Scripts\python.exe scripts\sync_serving_index.py `
  --device cpu `
  --request-delay-seconds 3 `
  --fetch-retries 10 `
  --fetch-retry-delay-seconds 120 `
  --embedding-batch-size 128 `
  --checkpoint-every-records 10000 `
  --db-path data/paper_recommender_1m.db `
  --exact-index-path data/vectors_1m.npz `
  --serving-index-path data/vectors_1m_int8_mmap `
  --serving-index-kind int8_mmap `
  --top-k 10 `
  --sample-size 1000 `
  --label daily-int8-mmap
```

Use this command only after the exact index has been backfilled to the current
OAI datestamp. The deployed 3M artifact currently stops at OAI datestamp
`2026-04-23`, so a daily sync resumes from that cursor.
