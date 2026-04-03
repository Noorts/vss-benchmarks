"""
Repack shuffle_train parquet files so that every output file (except
possibly the last) contains exactly a multiple of ROW_GROUP_SIZE rows,
while staying under MAX_FILE_SIZE_GB.  The original files are left
untouched.

The number of row-groups per file is determined automatically from the
first input file's bytes-per-row ratio.

Usage:
  python repack_train.py <input_dir> <output_dir>

Example:
  python repack_train.py \
    /path/to/cohere_large_10m \
    /path/to/cohere_large_10m_repacked
"""

import argparse
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROW_GROUP_SIZE = 122_880
MAX_FILE_SIZE_GB = 5
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_GB * 2**30


def iter_input_files(input_dir: Path) -> list[Path]:
    """Return sorted shuffle_train parquet paths."""
    files = sorted(input_dir.glob("shuffle_train-*-of-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No shuffle_train files found in {input_dir}")
    return files


def compute_rows_per_file(input_files: list[Path]) -> int:
    """Pick the largest multiple of ROW_GROUP_SIZE that fits under the size limit.

    Uses a 10% safety margin because compressed size is not perfectly
    proportional to row count.
    """
    # Average bytes-per-row across all input files for a better estimate.
    total_bytes = sum(os.path.getsize(f) for f in input_files)
    total_rows = sum(pq.read_metadata(f).num_rows for f in input_files)
    bytes_per_row = total_bytes / total_rows

    safe_limit = MAX_FILE_SIZE_BYTES * 0.9
    max_rows = int(safe_limit / bytes_per_row)
    num_groups = max_rows // ROW_GROUP_SIZE
    if num_groups < 1:
        num_groups = 1
    return num_groups * ROW_GROUP_SIZE


def repack(input_dir: Path, output_dir: Path) -> None:
    input_files = iter_input_files(input_dir)

    total_rows = 0
    for f in input_files:
        n = pq.read_metadata(f).num_rows
        total_rows += n
        print(f"  {f.name}: {n:,} rows")
    print(f"Total: {total_rows:,} rows")

    rows_per_file = compute_rows_per_file(input_files)
    num_full = total_rows // rows_per_file
    remainder = total_rows % rows_per_file
    num_files = num_full + (1 if remainder else 0)
    print(f"Output: {num_files} files, {rows_per_file:,} rows each "
          f"({rows_per_file // ROW_GROUP_SIZE} row-groups per file)" +
          (f", last file: {remainder:,} rows" if remainder else ""))

    output_dir.mkdir(parents=True, exist_ok=True)
    padding = max(2, len(str(num_files)))

    # Stream through input files and write fixed-size output chunks.
    buf: pa.Table | None = None
    file_idx = 0
    for input_path in input_files:
        print(f"Reading {input_path.name} …")
        table = pq.read_table(input_path)
        buf = pa.concat_tables([buf, table]) if buf is not None else table

        while buf.num_rows >= rows_per_file:
            chunk = buf.slice(0, rows_per_file)
            buf = buf.slice(rows_per_file)

            out_name = (
                f"shuffle_train-{str(file_idx).zfill(padding)}"
                f"-of-{num_files}.parquet"
            )
            out_path = output_dir / out_name
            pq.write_table(chunk, out_path, row_group_size=ROW_GROUP_SIZE)
            size = os.path.getsize(out_path)
            print(f"  {out_name}: {chunk.num_rows:,} rows ({size / (1024**3):.2f} GB)")
            if size > MAX_FILE_SIZE_BYTES:
                print(f"  WARNING: file exceeds {MAX_FILE_SIZE_GB} GB limit!")
            file_idx += 1

    # Write the remainder (if any).
    if buf is not None and buf.num_rows > 0:
        out_name = (
            f"shuffle_train-{str(file_idx).zfill(padding)}"
            f"-of-{num_files}.parquet"
        )
        out_path = output_dir / out_name
        pq.write_table(buf, out_path, row_group_size=ROW_GROUP_SIZE)
        size = os.path.getsize(out_path)
        print(f"  {out_name}: {buf.num_rows:,} rows ({size / (1024**3):.2f} GB)")
        file_idx += 1

    print(f"Wrote {file_idx} files to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repack shuffle_train parquet files into fixed-size chunks."
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Directory containing the original shuffle_train-*-of-*.parquet files.",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Directory to write the repacked files into.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repack(Path(args.input_dir), Path(args.output_dir))
    print("Done.")


if __name__ == "__main__":
    main()
