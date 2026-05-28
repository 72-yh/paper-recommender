from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from paper_recommender.compressed_vector_store import MmapInt8VectorIndex


@dataclass(frozen=True)
class IvfBuildReport:
    index_path: Path
    indexed_vectors: int
    dimensions: int
    n_clusters: int
    train_sample_size: int
    iterations: int
    build_seconds: float
    output_bytes: int


def build_ivf_int8_index(
    *,
    index_path: str | Path,
    n_clusters: int,
    train_sample_size: int,
    iterations: int,
    assignment_batch_size: int = 65_536,
    seed: int = 13,
) -> IvfBuildReport:
    if n_clusters <= 0:
        raise ValueError("n_clusters must be positive")
    if n_clusters > np.iinfo(np.uint16).max:
        raise ValueError("n_clusters must fit in uint16")
    if train_sample_size <= 0:
        raise ValueError("train_sample_size must be positive")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if assignment_batch_size <= 0:
        raise ValueError("assignment_batch_size must be positive")

    start = time.perf_counter()
    index_path = Path(index_path)
    index = MmapInt8VectorIndex.load(index_path)
    if len(index.vector_ids) == 0:
        raise ValueError("index must contain vectors")
    if n_clusters > len(index.vector_ids):
        raise ValueError("n_clusters cannot exceed indexed vector count")

    rng = np.random.default_rng(seed)
    sample_rows = _sample_rows(len(index.vector_ids), train_sample_size, rng)
    training_vectors = _decode_normalized_rows(index, sample_rows)
    centroids = _train_centroids(
        training_vectors,
        n_clusters=n_clusters,
        iterations=iterations,
        rng=rng,
    )
    cluster_ids = _assign_clusters(
        index,
        centroids,
        assignment_batch_size=assignment_batch_size,
    )

    np.save(index_path / "centroids.npy", centroids.astype(np.float32))
    np.save(index_path / "cluster_ids.npy", cluster_ids.astype(np.uint16))
    _write_clustered_arrays(
        index,
        cluster_ids,
        n_clusters=n_clusters,
        index_path=index_path,
        batch_size=assignment_batch_size,
    )
    output_bytes = sum(
        (index_path / filename).stat().st_size
        for filename in (
            "centroids.npy",
            "cluster_ids.npy",
            "cluster_offsets.npy",
            "clustered_vector_ids.npy",
            "clustered_codes.npy",
            "clustered_row_norms.npy",
        )
    )

    return IvfBuildReport(
        index_path=index_path,
        indexed_vectors=len(index.vector_ids),
        dimensions=int(index.codes.shape[1]),
        n_clusters=n_clusters,
        train_sample_size=len(training_vectors),
        iterations=iterations,
        build_seconds=time.perf_counter() - start,
        output_bytes=output_bytes,
    )


def _sample_rows(total_rows: int, sample_size: int, rng: np.random.Generator) -> np.ndarray:
    limit = min(total_rows, sample_size)
    if limit == total_rows:
        return np.arange(total_rows, dtype=np.int64)
    return np.sort(rng.choice(total_rows, size=limit, replace=False).astype(np.int64))


def _decode_normalized_rows(index: MmapInt8VectorIndex, rows: np.ndarray) -> np.ndarray:
    vectors = np.asarray(index.codes[rows], dtype=np.float32) * index.scales
    return _normalize_matrix(vectors)


def _train_centroids(
    vectors: np.ndarray,
    *,
    n_clusters: int,
    iterations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    initial_rows = rng.choice(len(vectors), size=n_clusters, replace=False)
    centroids = vectors[initial_rows].astype(np.float32, copy=True)
    centroids = _normalize_matrix(centroids)

    for _ in range(iterations):
        assignments = _nearest_centroids(vectors, centroids)
        sums = np.zeros_like(centroids, dtype=np.float32)
        np.add.at(sums, assignments, vectors)
        counts = np.bincount(assignments, minlength=n_clusters).astype(np.float32)
        empty = counts == 0
        counts[empty] = 1.0
        centroids = sums / counts[:, np.newaxis]
        if np.any(empty):
            replacement_rows = rng.choice(len(vectors), size=int(empty.sum()), replace=False)
            centroids[empty] = vectors[replacement_rows]
        centroids = _normalize_matrix(centroids)

    return centroids.astype(np.float32)


def _assign_clusters(
    index: MmapInt8VectorIndex,
    centroids: np.ndarray,
    *,
    assignment_batch_size: int,
) -> np.ndarray:
    cluster_ids = np.empty(len(index.vector_ids), dtype=np.uint16)
    for start in range(0, len(index.vector_ids), assignment_batch_size):
        end = min(len(index.vector_ids), start + assignment_batch_size)
        rows = np.arange(start, end, dtype=np.int64)
        vectors = _decode_normalized_rows(index, rows)
        cluster_ids[start:end] = _nearest_centroids(vectors, centroids).astype(np.uint16)
    return cluster_ids


def _nearest_centroids(vectors: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    return np.argmax(vectors @ centroids.T, axis=1).astype(np.int64)


def _write_clustered_arrays(
    index: MmapInt8VectorIndex,
    cluster_ids: np.ndarray,
    *,
    n_clusters: int,
    index_path: Path,
    batch_size: int,
) -> None:
    order = np.argsort(cluster_ids, kind="stable").astype(np.int64)
    counts = np.bincount(cluster_ids, minlength=n_clusters).astype(np.int64)
    offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
    np.save(index_path / "cluster_offsets.npy", offsets)
    _write_reordered_array(
        index_path / "clustered_vector_ids.npy",
        index.vector_ids,
        order,
        batch_size=batch_size,
        dtype=np.int64,
    )
    _write_reordered_array(
        index_path / "clustered_codes.npy",
        index.codes,
        order,
        batch_size=batch_size,
        dtype=np.int8,
    )
    _write_reordered_array(
        index_path / "clustered_row_norms.npy",
        index._row_norms,
        order,
        batch_size=batch_size,
        dtype=np.float32,
    )


def _write_reordered_array(
    path: Path,
    source: np.ndarray,
    order: np.ndarray,
    *,
    batch_size: int,
    dtype,
) -> None:
    target = np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=dtype,
        shape=source.shape,
    )
    for start in range(0, len(order), batch_size):
        end = min(len(order), start + batch_size)
        target[start:end] = source[order[start:end]]
    del target


def _normalize_matrix(vectors: np.ndarray) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float32)
    if len(matrix) == 0:
        return matrix.copy()
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms != 0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build IVF cluster files for an int8_mmap index."
    )
    parser.add_argument("--index-path", type=Path, required=True)
    parser.add_argument("--n-clusters", type=int, default=1024)
    parser.add_argument("--train-sample-size", type=int, default=100_000)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--assignment-batch-size", type=int, default=65_536)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    report = build_ivf_int8_index(
        index_path=args.index_path,
        n_clusters=args.n_clusters,
        train_sample_size=args.train_sample_size,
        iterations=args.iterations,
        assignment_batch_size=args.assignment_batch_size,
        seed=args.seed,
    )
    print(_format_report(report))


def _format_report(report: IvfBuildReport) -> str:
    return (
        "IVF build ok: "
        f"index_path={report.index_path} "
        f"indexed_vectors={report.indexed_vectors} "
        f"dimensions={report.dimensions} "
        f"n_clusters={report.n_clusters} "
        f"train_sample_size={report.train_sample_size} "
        f"iterations={report.iterations} "
        f"build_seconds={report.build_seconds:.3f} "
        f"output_bytes={report.output_bytes}"
    )


if __name__ == "__main__":
    main()
