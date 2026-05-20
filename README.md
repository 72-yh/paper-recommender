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
  --embedding-batch-size 512 `
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
  --embedding-batch-size 512 `
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

Serve the 1M int8 index locally:

```powershell
$env:PAPER_RECOMMENDER_DB_PATH='data/paper_recommender_1m.db'
$env:PAPER_RECOMMENDER_INDEX_PATH='data/vectors_1m_int8.npz'
$env:PAPER_RECOMMENDER_INDEX_KIND='int8'
.\.venv\Scripts\python.exe -m uvicorn paper_recommender.app:app --host 127.0.0.1 --port 8000
```
