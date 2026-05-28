from pathlib import Path


DESIGN_DOC = Path("docs/superpowers/specs/2026-05-18-paper-recommender-design.md")
REAL_OAI_PLAN = Path("docs/superpowers/plans/2026-05-19-real-oai-index-proof.md")
FLY_RUNBOOK = Path("docs/deployment/fly-low-cost.md")
CURRENT_STATE = Path("docs/operations/current-state.md")


def test_design_doc_reflects_current_mvp_decisions() -> None:
    design = DESIGN_DOC.read_text(encoding="utf-8")

    assert "The UI does not expose a Top K control" in design
    assert "always requests 10 recommendations" in design
    assert "multi-select category filter" in design
    assert "OR semantics" in design
    assert "Fly.io low-cost deployment" in design
    assert "FAISS and USearch are not deployed in the current MVP" in design
    assert "ivf_int8_mmap" in design
    assert "NumPy full-scan" in design
    assert "score only that filtered vector subset" in design
    assert "USearch has a local" in design
    assert "USearch i8 is smaller but currently loses too much recall" in design
    assert "scripts/run_daily_update.py" in design
    assert "daily wrapper should not call Fly commands automatically" in design
    assert "do not rewrite the large exact vector index" in design
    assert "sync command should use `--serving-index-kind int8_mmap`" in design
    assert "Use `--target-vector-count` during catch-up backfills" in design
    assert "paper_categories" in design
    assert "4GB volume" in design
    assert "3,000,000 indexed papers" in design
    assert "idx_papers_status_count" in design
    assert "clustered int8 mmap" in design
    assert "0.572s" in design
    assert "1.880s" in design


def test_real_oai_plan_tracks_current_followup_tasks() -> None:
    plan = REAL_OAI_PLAN.read_text(encoding="utf-8")

    assert "Task 6: Container And Fly Deployment" in plan
    assert "Task 7: Current Serving Performance Baseline" in plan
    assert "Task 8: ANN Serving Index Evaluation" in plan
    assert "Task 9: Incremental int8_mmap Serving Sync" in plan
    assert "Task 10: Controlled Full-Corpus Catch-Up" in plan
    assert "- [x] Build a benchmark harness over the existing 1M proof artifacts." in plan
    assert "--serving-index-kind int8_mmap" in plan
    assert "--target-vector-count" in plan
    assert "Small catch-up result" in plan
    assert "3M catch-up result" in plan
    assert "3M local API smoke" in plan
    assert "Task 11: 3M Fly Deployment Baseline" in plan
    assert "Task 12: IVF int8_mmap Serving Index" in plan
    assert "3M Fly smoke" in plan
    assert "Production clustered IVF result" in plan
    assert "Task 13: Filtered IVF Candidate Lookup" in plan
    assert "Task 14: Daily Local Update Orchestration" in plan
    assert "scripts/run_daily_update.py" in plan
    assert "production mutations remain manual and reviewed" in plan
    assert "Daily smoke result" in plan
    assert "`--max-records 50` processed 50 unchanged records" in plan
    assert "3702979208" in plan
    assert "0.572s" in plan
    assert "1.880s" in plan
    assert "1,000,050" in plan
    assert "3,000,000" in plan
    assert "2469075200" in plan
    assert "scripts/evaluate_ann.py" in plan
    assert "recall@10 `0.9980`" in plan


def test_fly_runbook_documents_current_operational_lessons() -> None:
    runbook = FLY_RUNBOOK.read_text(encoding="utf-8")

    assert "--local-only" in runbook
    assert "Do not use a remote builder" in runbook
    assert "--target-indexed-papers 3000000" in runbook
    assert "--max-volume-gb 4" in runbook
    assert "USearch and other ANN indexes are local evaluation candidates only" in runbook
    assert "Daily Local Sync" in runbook
    assert "scripts\\run_daily_update.py" in runbook
    assert "--target-vector-count" in runbook
    assert "does not" in runbook
    assert "`fly sftp`, `fly deploy`" in runbook
    assert "first small catch-up smoke run used target 1,000,050" in runbook
    assert "local 3M catch-up later reached 3,000,000 indexed papers" in runbook
    assert "2,469,075,200 total artifact bytes" in runbook
    assert "Run daily OAI updates on a local build machine" in runbook
    assert "Machine may auto-stop during long SFTP uploads" in runbook
    assert "chunked archive transfer" in runbook
    assert "--timeout-seconds 180" in runbook
    assert "idx_papers_status_count" in runbook
    assert "ivf_int8_mmap" in runbook
    assert "build_ivf_int8_index.py" in runbook
    assert "clustered_codes.npy" in runbook
    assert "3,702,979,208 total artifact bytes" in runbook
    assert "0.572s" in runbook
    assert "1.880s" in runbook
    assert "first recommendation requests can load the index" in runbook


def test_current_state_doc_records_faiss_and_latency_status() -> None:
    current_state = CURRENT_STATE.read_text(encoding="utf-8")

    assert "FAISS and USearch are not currently deployed" in current_state
    assert "Local serving benchmark harness" in current_state
    assert "Local USearch f16 ANN evaluation" in current_state
    assert "recall@10 0.9980" in current_state
    assert "USearch i8 is smaller" in current_state
    assert "Daily OAI update orchestration now has a local wrapper" in current_state
    assert "scripts/run_daily_update.py" in current_state
    assert "intentionally stops" in current_state
    assert "Fly upload or deploy step" in current_state
    assert "--serving-index-kind int8_mmap" in current_state
    assert "accepts `--target-vector-count`" in current_state
    assert "Local controlled catch-up smoke run" in current_state
    assert "1,000,050" in current_state
    assert "projected_total_artifact_bytes=2421512213" in current_state
    assert "Local full catch-up run" in current_state
    assert "3,000,000" in current_state
    assert "total_artifact_bytes=2469075200" in current_state
    assert "Local serving benchmark after the 3M catch-up" in current_state
    assert "Local 3M API smoke test" in current_state
    assert "`last_oai_datestamp=2026-04-23`" in current_state
    assert "int8_mmap" in current_state
    assert "prefilter candidate `vector_id`s" in current_state
    assert "3M Budget Path" in current_state
    assert "Production 3M deployment" in current_state
    assert "idx_papers_status_count" in current_state
    assert "Local 3M IVF int8 evaluation" in current_state
    assert "recall@10 0.9920" in current_state
    assert "ivf_int8_mmap" in current_state
    assert "Production clustered IVF deployment" in current_state
    assert "clustered_codes.npy" in current_state
    assert "total_artifact_bytes=3702979208" in current_state
    assert "0.572s" in current_state
    assert "8.405s" in current_state
    assert "1.880s" in current_state
    assert "filtered IVF candidate lookup fix" in current_state
    assert "Local daily wrapper no-op check" in current_state
    assert "Local daily OAI smoke run" in current_state
    assert "processed 50 unchanged OAI records" in current_state
    assert "do" in current_state
    assert "not rewrite large vector artifacts" in current_state
    assert "Cold start" in current_state
    assert "Warm recommendation" in current_state
