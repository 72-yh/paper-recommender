import subprocess
import sys
from pathlib import Path

from paper_recommender.index_builder import IndexBuildSummary
from scripts.build_ivf_int8_index import IvfBuildReport
from scripts.preflight_artifacts import ArtifactPreflightSummary
from scripts.run_daily_update import run_daily_update
from scripts.sync_serving_index import ServingSyncSummary


def _index_summary(*, embedded: int = 0, deleted: int = 0) -> IndexBuildSummary:
    return IndexBuildSummary(
        batches_seen=1,
        records_seen=3,
        inserted=embedded,
        updated=0,
        unchanged=3 - embedded - deleted,
        deleted=deleted,
        embedded=embedded,
        checkpoints_written=1,
        last_datestamp="2026-05-28",
    )


def _serving_summary(*, rebuilt: bool) -> ServingSyncSummary:
    return ServingSyncSummary(
        update=_index_summary(embedded=1 if rebuilt else 0),
        serving_index_kind="int8_mmap",
        serving_index_path=Path("data/vectors_1m_int8_mmap"),
        rebuilt_serving_index=rebuilt,
        compression=None,
    )


def _ivf_report(index_path: Path) -> IvfBuildReport:
    return IvfBuildReport(
        index_path=index_path,
        indexed_vectors=3_000_000,
        dimensions=384,
        n_clusters=512,
        train_sample_size=100_000,
        iterations=6,
        build_seconds=17.0,
        output_bytes=1_194_791_304,
    )


def _preflight_summary(db_path: Path, index_path: Path) -> ArtifactPreflightSummary:
    return ArtifactPreflightSummary(
        db_path=db_path,
        index_path=index_path,
        index_kind="ivf_int8_mmap",
        active_papers=3_000_000,
        indexed_papers=3_000_000,
        index_vectors=3_000_000,
        dimensions=384,
        db_bytes=1,
        index_bytes=2,
        total_artifact_bytes=3,
        last_oai_datestamp="2026-05-28",
        vector_ids_checked=True,
        category_lookup_checked=True,
        category_lookup_rows=5,
        target_indexed_papers=3_000_000,
        projected_total_artifact_bytes=3,
        max_volume_gb=4.0,
    )


def test_run_daily_update_rebuilds_ivf_and_preflights_after_vector_changes(tmp_path) -> None:
    calls: list[str] = []
    db_path = tmp_path / "papers.db"
    exact_index_path = tmp_path / "vectors.npz"
    serving_index_path = tmp_path / "vectors_int8_mmap"

    def sync(**kwargs):
        calls.append("sync")
        assert kwargs["serving_index_kind"] == "int8_mmap"
        assert kwargs["db_path"] == db_path
        assert kwargs["exact_index_path"] == exact_index_path
        assert kwargs["serving_index_path"] == serving_index_path
        return _serving_summary(rebuilt=True)

    def build_ivf(**kwargs):
        calls.append("build_ivf")
        assert kwargs["index_path"] == serving_index_path
        assert kwargs["n_clusters"] == 512
        return _ivf_report(serving_index_path)

    def preflight(**kwargs):
        calls.append("preflight")
        assert kwargs["db_path"] == db_path
        assert kwargs["index_path"] == serving_index_path
        assert kwargs["index_kind"] == "ivf_int8_mmap"
        assert kwargs["target_indexed_papers"] == 3_000_000
        assert kwargs["max_volume_gb"] == 4.0
        return _preflight_summary(db_path, serving_index_path)

    summary = run_daily_update(
        db_path=db_path,
        exact_index_path=exact_index_path,
        serving_index_path=serving_index_path,
        sync=sync,
        build_ivf=build_ivf,
        preflight=preflight,
    )

    assert calls == ["sync", "build_ivf", "preflight"]
    assert summary.rebuilt_ivf is True
    assert summary.preflight is not None


def test_run_daily_update_skips_ivf_when_no_vectors_changed_but_still_preflights(
    tmp_path,
) -> None:
    calls: list[str] = []

    def sync(**_kwargs):
        calls.append("sync")
        return _serving_summary(rebuilt=False)

    def build_ivf(**_kwargs):
        raise AssertionError("IVF should not rebuild without vector changes")

    def preflight(**kwargs):
        calls.append("preflight")
        return _preflight_summary(kwargs["db_path"], kwargs["index_path"])

    summary = run_daily_update(
        db_path=tmp_path / "papers.db",
        exact_index_path=tmp_path / "vectors.npz",
        serving_index_path=tmp_path / "vectors_int8_mmap",
        sync=sync,
        build_ivf=build_ivf,
        preflight=preflight,
    )

    assert calls == ["sync", "preflight"]
    assert summary.rebuilt_ivf is False


def test_run_daily_update_force_rebuilds_ivf_without_vector_changes(tmp_path) -> None:
    calls: list[str] = []

    def sync(**_kwargs):
        calls.append("sync")
        return _serving_summary(rebuilt=False)

    def build_ivf(**kwargs):
        calls.append("build_ivf")
        return _ivf_report(kwargs["index_path"])

    def preflight(**kwargs):
        calls.append("preflight")
        return _preflight_summary(kwargs["db_path"], kwargs["index_path"])

    summary = run_daily_update(
        db_path=tmp_path / "papers.db",
        exact_index_path=tmp_path / "vectors.npz",
        serving_index_path=tmp_path / "vectors_int8_mmap",
        force_ivf_rebuild=True,
        sync=sync,
        build_ivf=build_ivf,
        preflight=preflight,
    )

    assert calls == ["sync", "build_ivf", "preflight"]
    assert summary.rebuilt_ivf is True


def test_run_daily_update_cli_help_loads() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/run_daily_update.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Run daily OAI sync" in result.stdout
    assert "--force-ivf-rebuild" in result.stdout
