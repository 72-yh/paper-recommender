from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path

from paper_recommender.compressed_vector_store import Int8VectorIndex


@dataclass(frozen=True)
class Int8MmapConversionSummary:
    input_path: Path
    output_path: Path
    vectors: int
    dimensions: int
    output_bytes: int


def convert_int8_index_to_mmap(
    *,
    input_path: str | Path,
    output_path: str | Path,
    overwrite: bool = False,
) -> Int8MmapConversionSummary:
    input_path = Path(input_path)
    output_path = Path(output_path)
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"Output path already exists: {output_path}")
        if output_path.is_dir():
            shutil.rmtree(output_path)
        else:
            output_path.unlink()

    index = Int8VectorIndex.load(input_path)
    index.save_mmap(output_path)
    return Int8MmapConversionSummary(
        input_path=input_path,
        output_path=output_path,
        vectors=int(index.codes.shape[0]),
        dimensions=int(index.codes.shape[1]) if index.codes.ndim == 2 else 0,
        output_bytes=_directory_bytes(output_path),
    )


def _directory_bytes(path: Path) -> int:
    return sum(child.stat().st_size for child in path.iterdir() if child.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a compressed int8 NPZ index to mmap NPY files."
    )
    parser.add_argument("--input", type=Path, default=Path("data/vectors_1m_int8.npz"))
    parser.add_argument("--output", type=Path, default=Path("data/vectors_1m_int8_mmap"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    summary = convert_int8_index_to_mmap(
        input_path=args.input,
        output_path=args.output,
        overwrite=args.overwrite,
    )
    print(
        "Converted int8 mmap index: "
        f"input={summary.input_path} "
        f"output={summary.output_path} "
        f"vectors={summary.vectors} "
        f"dimensions={summary.dimensions} "
        f"output_bytes={summary.output_bytes}"
    )


if __name__ == "__main__":
    main()
