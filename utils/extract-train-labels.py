"""
Extract per-vector label columns from arxiv-for-fanns-train.parquet (http://arxiv.org/abs/2507.21989) into
three separate parquet files (one per filter type), each with an 'id' index
column and the corresponding attribute column:

  - em_labels.parquet   : id (int), number_of_sub_categories (int)
  - r_labels.parquet    : id (int), update_date (int)
  - emis_labels.parquet : id (int), main_categories (list<str>)
"""

import argparse
import os

import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

ROW_GROUP_SIZE = 122_880

SCHEMA_EM = pa.schema(
    [
        ("id", pa.int64()),
        ("number_of_sub_categories", pa.int64()),
    ]
)

SCHEMA_R = pa.schema(
    [
        ("id", pa.int64()),
        ("update_date", pa.int64()),
    ]
)

SCHEMA_EMIS = pa.schema(
    [
        ("id", pa.int64()),
        ("main_categories", pa.list_(pa.utf8())),
    ]
)


def extract_labels(input_file: str, output_dir: str) -> None:
    input_path = Path(input_file)
    output = Path(output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output.mkdir(parents=True, exist_ok=True)

    print(f"Reading {input_path} …")
    source = pq.read_table(
        input_path,
        columns=["id", "number_of_sub_categories", "update_date", "main_categories"],
    )
    print(f"  {source.num_rows:,} rows")

    ids = source.column("id")

    # EM
    em_table = pa.table(
        {
            "id": ids,
            "number_of_sub_categories": source.column("number_of_sub_categories"),
        },
        schema=SCHEMA_EM,
    )
    write_and_report(em_table, output / "em_labels.parquet")

    # R
    r_table = pa.table(
        {"id": ids, "update_date": source.column("update_date")},
        schema=SCHEMA_R,
    )
    write_and_report(r_table, output / "r_labels.parquet")

    # EMIS
    emis_table = pa.table(
        {"id": ids, "main_categories": source.column("main_categories")},
        schema=SCHEMA_EMIS,
    )
    write_and_report(emis_table, output / "emis_labels.parquet")


def write_and_report(table: pa.Table, path: Path) -> None:
    pq.write_table(table, path, row_group_size=ROW_GROUP_SIZE)
    size = os.path.getsize(path)
    print(f"  Wrote {path.name}: {table.num_rows:,} rows ({size / 1024:.1f} KB)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract per-vector label columns from arxiv-for-fanns-train.parquet.",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default="files/arxiv-for-fanns-train.parquet",
        help="Path to the source train parquet file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="files/arxiv-for-fanns-1024-euclidean",
        help="Directory for the output label parquet files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    extract_labels(args.input_file, args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
