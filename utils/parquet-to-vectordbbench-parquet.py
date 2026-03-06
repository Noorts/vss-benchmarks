"""
Read an arxiv-for-fanns-train.parquet file (http://arxiv.org/abs/2507.21989) (which may contain extra attribute
columns) and write VectorDBBench-compatible train parquet file(s) with only
the 'id' and 'emb' columns.  Large files are automatically split into
multiple files that stay under MAX_FILE_SIZE_GB.
"""

import argparse
import os

import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

ROW_GROUP_SIZE = 122_880
MAX_FILE_SIZE_GB = 4
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_GB * 2**30

VECTORDBBENCH_SCHEMA = pa.schema(
    [
        ("id", pa.int64()),
        ("emb", pa.list_(pa.float32())),
    ]
)


def convert_parquet(input_file: str, output_file: str) -> None:
    input_path = Path(input_file)
    output_path = Path(output_file)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Reading {input_path} …")
    source = pq.read_table(input_path, columns=["id", "emb"])

    source = source.cast(VECTORDBBENCH_SCHEMA)
    total_rows = source.num_rows
    print(f"  {total_rows:,} rows, columns: {source.column_names}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Writing {output_path} …")
    pq.write_table(source, output_path, row_group_size=ROW_GROUP_SIZE)

    file_size = os.path.getsize(output_path)
    print(f"  Wrote {output_path} ({file_size / (1024**3):.2f} GB)")

    if file_size > MAX_FILE_SIZE_BYTES:
        print(f"File exceeds {MAX_FILE_SIZE_GB} GB limit, splitting …")
        split_parquet_file(output_path, source, total_rows, file_size)
    else:
        print("File is within size limit, no split needed.")


def split_parquet_file(
    parquet_path: Path,
    table: pa.Table,
    total_rows: int,
    file_size: int,
) -> None:
    """Split a parquet table into multiple files that stay under MAX_FILE_SIZE_GB."""
    bytes_per_row = file_size / total_rows
    max_rows_per_file = int(MAX_FILE_SIZE_BYTES / bytes_per_row)

    rows_per_file = (max_rows_per_file // ROW_GROUP_SIZE) * ROW_GROUP_SIZE
    if rows_per_file < ROW_GROUP_SIZE:
        rows_per_file = ROW_GROUP_SIZE

    num_files = (total_rows + rows_per_file - 1) // rows_per_file
    print(f"Splitting into {num_files} files, ~{rows_per_file:,} rows each")

    base_dir = parquet_path.parent
    base_name = parquet_path.stem
    extension = parquet_path.suffix
    padding = max(2, len(str(num_files)))

    for file_idx in range(num_files):
        start = file_idx * rows_per_file
        end = min(start + rows_per_file, total_rows)
        chunk = table.slice(start, end - start)

        out_name = (
            f"{base_name}-{str(file_idx).zfill(padding)}-of-{num_files}{extension}"
        )
        out_path = base_dir / out_name

        pq.write_table(chunk, out_path, row_group_size=ROW_GROUP_SIZE)
        chunk_size = os.path.getsize(out_path)
        print(f"  {out_name}: {end - start:,} rows ({chunk_size / (1024**3):.2f} GB)")

    os.remove(parquet_path)
    print(f"Removed original {parquet_path.name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert an arxiv-for-fanns parquet file to VectorDBBench-compatible "
            "train parquet file(s) with only 'id' and 'emb' columns."
        )
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default="files/arxiv-for-fanns-train.parquet",
        help="Path to the source parquet file.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="files/train.parquet",
        help="Path to the output parquet file (will be split if >4 GB).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_parquet(args.input_file, args.output_file)
    print("Done.")


if __name__ == "__main__":
    main()
