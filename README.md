# paper-recommender

## Large Local Index Builds

Use separate output files for large proofs so smaller smoke-test data is not
overwritten.

Initial 1M build:

```powershell
.\.venv\Scripts\python.exe scripts\build_oai_index.py `
  --target-vector-count 1000000 `
  --request-delay-seconds 3 `
  --checkpoint-every-batches 10 `
  --db-path data/paper_recommender_1m.db `
  --index-path data/vectors_1m.npz `
  --reset
```

Resume an interrupted 1M build:

```powershell
.\.venv\Scripts\python.exe scripts\build_oai_index.py `
  --target-vector-count 1000000 `
  --resume `
  --request-delay-seconds 3 `
  --checkpoint-every-batches 10 `
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
