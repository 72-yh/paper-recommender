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
    assert "FAISS is not implemented in the current MVP" in design
    assert "NumPy full-scan" in design
    assert "score only that filtered vector subset" in design
    assert "paper_categories" in design
    assert "4GB volume" in design


def test_real_oai_plan_tracks_current_followup_tasks() -> None:
    plan = REAL_OAI_PLAN.read_text(encoding="utf-8")

    assert "Task 6: Container And Fly Deployment" in plan
    assert "Task 7: Current Serving Performance Baseline" in plan
    assert "Task 8: ANN Serving Index Evaluation" in plan


def test_fly_runbook_documents_current_operational_lessons() -> None:
    runbook = FLY_RUNBOOK.read_text(encoding="utf-8")

    assert "--local-only" in runbook
    assert "Do not use a remote builder" in runbook
    assert "Machine may auto-stop during long SFTP uploads" in runbook
    assert "first recommendation requests can load the index" in runbook


def test_current_state_doc_records_faiss_and_latency_status() -> None:
    current_state = CURRENT_STATE.read_text(encoding="utf-8")

    assert "FAISS is not currently deployed" in current_state
    assert "1M int8 NumPy full-scan" in current_state
    assert "int8_mmap" in current_state
    assert "prefilter candidate `vector_id`s" in current_state
    assert "3M Budget Path" in current_state
    assert "$0.30/month" in current_state
    assert "Cold start" in current_state
    assert "Warm recommendation" in current_state
