from __future__ import annotations

import argparse
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from itertools import zip_longest

import numpy as np

from paper_recommender.storage import connect_db, get_pipeline_state, list_active_category_counts


class ArtifactPreflightError(Exception):
    """Raised when local deployment artifacts are not ready to serve."""


@dataclass(frozen=True)
class ArtifactPreflightSummary:
    db_path: Path
    index_path: Path
    index_kind: str
    active_papers: int
    indexed_papers: int
    index_vectors: int
    dimensions: int
    db_bytes: int
    index_bytes: int
    total_artifact_bytes: int
    last_oai_datestamp: str | None
    vector_ids_checked: bool
    category_lookup_checked: bool
    category_lookup_rows: int | None
    target_indexed_papers: int | None
    projected_total_artifact_bytes: int | None
    max_volume_gb: float | None


@dataclass(frozen=True)
class _DbInfo:
    active_papers: int
    indexed_papers: int
    last_oai_datestamp: str | None
    vector_ids: np.ndarray | None
    category_lookup_rows: int | None


@dataclass(frozen=True)
class _IndexInfo:
    vectors: int
    dimensions: int
    bytes: int
    vector_ids: np.ndarray


def preflight_artifacts(
    *,
    db_path: str | Path,
    index_path: str | Path,
    index_kind: str = "int8",
    min_indexed_papers: int = 1,
    check_vector_ids: bool = True,
    check_category_lookup: bool = True,
    target_indexed_papers: int | None = None,
    max_volume_gb: float | None = None,
) -> ArtifactPreflightSummary:
    db_path = Path(db_path)
    index_path = Path(index_path)
    _require_file(db_path, "DB")
    _require_index_path(index_path, index_kind)

    db_info = _inspect_db(
        db_path,
        include_vector_ids=check_vector_ids,
        check_category_lookup=check_category_lookup,
    )
    index_info = _inspect_index(index_path, index_kind=index_kind)
    db_bytes = db_path.stat().st_size
    total_artifact_bytes = db_bytes + index_info.bytes

    if db_info.indexed_papers < min_indexed_papers:
        raise ArtifactPreflightError(
            f"Indexed paper count {db_info.indexed_papers} is below minimum {min_indexed_papers}"
        )
    if db_info.indexed_papers != index_info.vectors:
        raise ArtifactPreflightError(
            "DB/index count mismatch: "
            f"db_indexed_papers={db_info.indexed_papers} index_vectors={index_info.vectors}"
        )
    if check_vector_ids:
        _check_vector_ids(db_info, index_info)
    projected_total_artifact_bytes = _project_total_artifact_bytes(
        total_artifact_bytes=total_artifact_bytes,
        indexed_papers=db_info.indexed_papers,
        target_indexed_papers=target_indexed_papers,
    )
    if projected_total_artifact_bytes is not None and max_volume_gb is not None:
        max_volume_bytes = int(max_volume_gb * 1024 * 1024 * 1024)
        if projected_total_artifact_bytes > max_volume_bytes:
            raise ArtifactPreflightError(
                "Projected artifact size exceeds volume budget: "
                f"projected_bytes={projected_total_artifact_bytes} "
                f"max_volume_gb={max_volume_gb}"
            )

    return ArtifactPreflightSummary(
        db_path=db_path,
        index_path=index_path,
        index_kind=index_kind,
        active_papers=db_info.active_papers,
        indexed_papers=db_info.indexed_papers,
        index_vectors=index_info.vectors,
        dimensions=index_info.dimensions,
        db_bytes=db_bytes,
        index_bytes=index_info.bytes,
        total_artifact_bytes=total_artifact_bytes,
        last_oai_datestamp=db_info.last_oai_datestamp,
        vector_ids_checked=check_vector_ids,
        category_lookup_checked=check_category_lookup,
        category_lookup_rows=db_info.category_lookup_rows,
        target_indexed_papers=target_indexed_papers,
        projected_total_artifact_bytes=projected_total_artifact_bytes,
        max_volume_gb=max_volume_gb,
    )


def _require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise ArtifactPreflightError(f"{label} file does not exist: {path}")
    if not path.is_file():
        raise ArtifactPreflightError(f"{label} path is not a file: {path}")


def _require_index_path(path: Path, index_kind: str) -> None:
    if index_kind in {"int8_mmap", "ivf_int8_mmap"}:
        if not path.exists():
            raise ArtifactPreflightError(f"Index directory does not exist: {path}")
        if not path.is_dir():
            raise ArtifactPreflightError(f"Index path is not a directory: {path}")
        return
    _require_file(path, "Index")


def _inspect_db(
    path: Path,
    *,
    include_vector_ids: bool,
    check_category_lookup: bool,
) -> _DbInfo:
    conn = connect_db(path)
    try:
        active_papers = _count_papers(conn, "active = 1")
        indexed_papers = _count_papers(conn, "active = 1 AND vector_id IS NOT NULL")
        last_oai_datestamp = get_pipeline_state(conn, "last_successful_oai_datestamp")
        if last_oai_datestamp is None:
            last_oai_datestamp = _max_oai_datestamp(conn)
        vector_ids = _db_vector_ids(conn, indexed_papers) if include_vector_ids else None
        category_lookup_rows = (
            _check_category_lookup(conn) if check_category_lookup else None
        )
    except sqlite3.Error as exc:
        raise ArtifactPreflightError(f"Could not inspect DB {path}: {exc}") from exc
    finally:
        conn.close()
    return _DbInfo(
        active_papers=active_papers,
        indexed_papers=indexed_papers,
        last_oai_datestamp=last_oai_datestamp,
        vector_ids=vector_ids,
        category_lookup_rows=category_lookup_rows,
    )


def _inspect_index(path: Path, *, index_kind: str) -> _IndexInfo:
    if index_kind in {"int8_mmap", "ivf_int8_mmap"}:
        return _inspect_int8_mmap_index(path, require_ivf=index_kind == "ivf_int8_mmap")
    try:
        with np.load(path) as data:
            if index_kind == "exact":
                _require_arrays(data, path, ("vector_ids", "vectors"))
                vector_ids = np.asarray(data["vector_ids"], dtype=np.int64)
                matrix = data["vectors"]
                _require_matrix(matrix, "vectors", path)
            elif index_kind == "int8":
                _require_arrays(data, path, ("vector_ids", "codes", "scales"))
                vector_ids = np.asarray(data["vector_ids"], dtype=np.int64)
                matrix = data["codes"]
                scales = data["scales"]
                _require_matrix(matrix, "codes", path)
                if matrix.dtype != np.int8:
                    raise ArtifactPreflightError(f"Index codes must be int8 in {path}")
                if scales.ndim != 1 or scales.shape[0] != matrix.shape[1]:
                    raise ArtifactPreflightError(f"Index scales shape does not match codes in {path}")
            else:
                raise ArtifactPreflightError(f"Unsupported index kind: {index_kind}")

            if vector_ids.ndim != 1:
                raise ArtifactPreflightError(f"Index vector_ids must be one-dimensional in {path}")
            if vector_ids.shape[0] != matrix.shape[0]:
                raise ArtifactPreflightError(
                    f"Index vector_id count does not match vector rows in {path}"
                )
            return _IndexInfo(
                vectors=int(matrix.shape[0]),
                dimensions=int(matrix.shape[1]),
                bytes=path.stat().st_size,
                vector_ids=vector_ids,
            )
    except OSError as exc:
        raise ArtifactPreflightError(f"Could not read index {path}: {exc}") from exc
    except ValueError as exc:
        raise ArtifactPreflightError(f"Could not read index {path}: {exc}") from exc


def _inspect_int8_mmap_index(path: Path, *, require_ivf: bool = False) -> _IndexInfo:
    try:
        vector_ids = np.load(path / "vector_ids.npy", mmap_mode="r")
        matrix = np.load(path / "codes.npy", mmap_mode="r")
        scales = np.load(path / "scales.npy", mmap_mode="r")
        row_norms = np.load(path / "row_norms.npy", mmap_mode="r")
        _require_matrix(matrix, "codes", path)
        if vector_ids.ndim != 1:
            raise ArtifactPreflightError(f"Index vector_ids must be one-dimensional in {path}")
        if vector_ids.shape[0] != matrix.shape[0]:
            raise ArtifactPreflightError(
                f"Index vector_id count does not match vector rows in {path}"
            )
        if matrix.dtype != np.int8:
            raise ArtifactPreflightError(f"Index codes must be int8 in {path}")
        if scales.ndim != 1 or scales.shape[0] != matrix.shape[1]:
            raise ArtifactPreflightError(f"Index scales shape does not match codes in {path}")
        if row_norms.ndim != 1 or row_norms.shape[0] != matrix.shape[0]:
            raise ArtifactPreflightError(f"Index row_norms shape does not match codes in {path}")
        if require_ivf:
            cluster_ids = np.load(path / "cluster_ids.npy", mmap_mode="r")
            centroids = np.load(path / "centroids.npy", mmap_mode="r")
            cluster_offsets = np.load(path / "cluster_offsets.npy", mmap_mode="r")
            clustered_vector_ids = np.load(path / "clustered_vector_ids.npy", mmap_mode="r")
            clustered_codes = np.load(path / "clustered_codes.npy", mmap_mode="r")
            clustered_row_norms = np.load(path / "clustered_row_norms.npy", mmap_mode="r")
            if cluster_ids.ndim != 1 or cluster_ids.shape[0] != matrix.shape[0]:
                raise ArtifactPreflightError(
                    f"Index cluster_ids shape does not match codes in {path}"
                )
            _require_matrix(centroids, "centroids", path)
            if centroids.shape[1] != matrix.shape[1]:
                raise ArtifactPreflightError(
                    f"Index centroids shape does not match codes in {path}"
                )
            if np.max(cluster_ids, initial=0) >= centroids.shape[0]:
                raise ArtifactPreflightError(f"Index cluster_ids exceed centroid count in {path}")
            if cluster_offsets.ndim != 1 or cluster_offsets.shape[0] != centroids.shape[0] + 1:
                raise ArtifactPreflightError(
                    f"Index cluster_offsets length does not match centroids in {path}"
                )
            if cluster_offsets[0] != 0 or cluster_offsets[-1] != matrix.shape[0]:
                raise ArtifactPreflightError(f"Index cluster_offsets do not span rows in {path}")
            if np.any(np.diff(cluster_offsets) < 0):
                raise ArtifactPreflightError(f"Index cluster_offsets must be non-decreasing in {path}")
            if clustered_vector_ids.ndim != 1 or clustered_vector_ids.shape[0] != matrix.shape[0]:
                raise ArtifactPreflightError(
                    f"Index clustered_vector_ids shape does not match codes in {path}"
                )
            if clustered_codes.shape != matrix.shape or clustered_codes.dtype != np.int8:
                raise ArtifactPreflightError(
                    f"Index clustered_codes shape or dtype does not match codes in {path}"
                )
            if clustered_row_norms.ndim != 1 or clustered_row_norms.shape[0] != matrix.shape[0]:
                raise ArtifactPreflightError(
                    f"Index clustered_row_norms shape does not match codes in {path}"
                )
            if not np.array_equal(np.sort(clustered_vector_ids), np.sort(vector_ids)):
                raise ArtifactPreflightError(
                    f"Index clustered_vector_ids do not match vector_ids in {path}"
                )
        return _IndexInfo(
            vectors=int(matrix.shape[0]),
            dimensions=int(matrix.shape[1]),
            bytes=_directory_bytes(path),
            vector_ids=np.asarray(vector_ids, dtype=np.int64),
        )
    except FileNotFoundError as exc:
        raise ArtifactPreflightError(f"Index {path} is missing required mmap arrays") from exc
    except OSError as exc:
        raise ArtifactPreflightError(f"Could not read index {path}: {exc}") from exc
    except ValueError as exc:
        raise ArtifactPreflightError(f"Could not read index {path}: {exc}") from exc


def _require_arrays(data, path: Path, names: tuple[str, ...]) -> None:
    missing = [name for name in names if name not in data.files]
    if missing:
        raise ArtifactPreflightError(
            f"Index {path} is missing required arrays for this index kind: {', '.join(missing)}"
        )


def _require_matrix(matrix: np.ndarray, name: str, path: Path) -> None:
    if matrix.ndim != 2:
        raise ArtifactPreflightError(f"Index {name} must be two-dimensional in {path}")


def _check_vector_ids(db_info: _DbInfo, index_info: _IndexInfo) -> None:
    if db_info.vector_ids is None:
        return
    sorted_index_ids = np.sort(index_info.vector_ids)
    if not np.array_equal(db_info.vector_ids, sorted_index_ids):
        raise ArtifactPreflightError("Vector IDs do not match between DB and index")


def _count_papers(conn, where_clause: str) -> int:
    row = conn.execute(f"SELECT COUNT(*) AS value FROM papers WHERE {where_clause}").fetchone()
    return int(row["value"])


def _max_oai_datestamp(conn) -> str | None:
    row = conn.execute("SELECT MAX(oai_datestamp) AS value FROM papers").fetchone()
    return None if row is None else row["value"]


def _db_vector_ids(conn, count: int) -> np.ndarray:
    rows = conn.execute(
        """
        SELECT vector_id
        FROM papers
        WHERE active = 1 AND vector_id IS NOT NULL
        ORDER BY vector_id
        """
    )
    return np.fromiter((int(row["vector_id"]) for row in rows), dtype=np.int64, count=count)


def _check_category_lookup(conn) -> int:
    list_active_category_counts(conn)
    expected_rows = _expected_category_rows(conn)
    actual_rows = _actual_category_rows(conn)
    missing = object()
    count = 0
    for expected, actual in zip_longest(expected_rows, actual_rows, fillvalue=missing):
        if expected != actual:
            raise ArtifactPreflightError(
                "Category lookup mismatch: "
                f"expected_row={None if expected is missing else expected} "
                f"actual_row={None if actual is missing else actual}"
            )
        count += 1
    return count


def _expected_category_rows(conn):
    rows = conn.execute(
        """
        SELECT arxiv_id, vector_id, published_date, categories
        FROM papers
        WHERE active = 1 AND vector_id IS NOT NULL
        ORDER BY arxiv_id
        """
    )
    for row in rows:
        categories = sorted(dict.fromkeys(part for part in row["categories"].split(" ") if part))
        for category in categories:
            yield (row["arxiv_id"], category, int(row["vector_id"]), row["published_date"])


def _actual_category_rows(conn):
    rows = conn.execute(
        """
        SELECT arxiv_id, category, vector_id, published_date
        FROM paper_categories
        ORDER BY arxiv_id, category
        """
    )
    for row in rows:
        yield (row["arxiv_id"], row["category"], int(row["vector_id"]), row["published_date"])


def _project_total_artifact_bytes(
    *,
    total_artifact_bytes: int,
    indexed_papers: int,
    target_indexed_papers: int | None,
) -> int | None:
    if target_indexed_papers is None:
        return None
    if target_indexed_papers <= indexed_papers:
        return total_artifact_bytes
    if indexed_papers <= 0:
        raise ArtifactPreflightError("Cannot project artifact size without indexed papers")
    return int(np.ceil(total_artifact_bytes * (target_indexed_papers / indexed_papers)))


def _directory_bytes(path: Path) -> int:
    return sum(child.stat().st_size for child in path.iterdir() if child.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate Paper Recommender deployment artifacts."
    )
    parser.add_argument("--db-path", type=Path, default=Path("data/paper_recommender_1m.db"))
    parser.add_argument("--index-path", type=Path, default=Path("data/vectors_1m_int8.npz"))
    parser.add_argument(
        "--index-kind",
        choices=("exact", "int8", "int8_mmap", "ivf_int8_mmap"),
        default="int8",
    )
    parser.add_argument("--min-indexed-papers", type=int, default=1)
    parser.add_argument("--skip-vector-id-check", action="store_true")
    parser.add_argument("--skip-category-lookup-check", action="store_true")
    parser.add_argument("--target-indexed-papers", type=int)
    parser.add_argument("--max-volume-gb", type=float)
    args = parser.parse_args()

    try:
        summary = preflight_artifacts(
            db_path=args.db_path,
            index_path=args.index_path,
            index_kind=args.index_kind,
            min_indexed_papers=args.min_indexed_papers,
            check_vector_ids=not args.skip_vector_id_check,
            check_category_lookup=not args.skip_category_lookup_check,
            target_indexed_papers=args.target_indexed_papers,
            max_volume_gb=args.max_volume_gb,
        )
    except ArtifactPreflightError as exc:
        print(f"Artifact preflight failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(
        "Artifact preflight ok: "
        f"db_path={summary.db_path} "
        f"index_path={summary.index_path} "
        f"index_kind={summary.index_kind} "
        f"active_papers={summary.active_papers} "
        f"indexed_papers={summary.indexed_papers} "
        f"index_vectors={summary.index_vectors} "
        f"dimensions={summary.dimensions} "
        f"db_bytes={summary.db_bytes} "
        f"index_bytes={summary.index_bytes} "
        f"total_artifact_bytes={summary.total_artifact_bytes} "
        f"last_oai_datestamp={summary.last_oai_datestamp} "
        f"vector_ids_checked={summary.vector_ids_checked} "
        f"category_lookup_checked={summary.category_lookup_checked} "
        f"category_lookup_rows={summary.category_lookup_rows} "
        f"target_indexed_papers={summary.target_indexed_papers} "
        f"projected_total_artifact_bytes={summary.projected_total_artifact_bytes} "
        f"max_volume_gb={summary.max_volume_gb}"
    )


if __name__ == "__main__":
    main()
