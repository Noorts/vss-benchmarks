"""
Read the arxiv-for-fanns (http://arxiv.org/abs/2507.21989) query attribute JSONL files and convert them into
VectorDBBench-compatible parquet files:

  - test_attrs_em.parquet   : label (int)              from em_query_attributes.jsonl
  - test_attrs_r.parquet    : range_start/range_end (int) from r_query_attributes.jsonl
  - test_attrs_emis.parquet : label (str)              from emis_query_attributes.jsonl
"""

import argparse
import json
import os

import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

ROW_GROUP_SIZE = 122_880

SCHEMA_EM = pa.schema([("label", pa.int64())])
SCHEMA_R = pa.schema([("range_start", pa.int64()), ("range_end", pa.int64())])
SCHEMA_EMIS = pa.schema([("label", pa.utf8())])


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def extract_attrs(em_file: str, r_file: str, emis_file: str, output_dir: str) -> None:
    em_path = Path(em_file)
    r_path = Path(r_file)
    emis_path = Path(emis_file)
    output = Path(output_dir)

    for p in (em_path, r_path, emis_path):
        if not p.exists():
            raise FileNotFoundError(f"Input file not found: {p}")

    output.mkdir(parents=True, exist_ok=True)

    # EM: {"label": int} → label (int)
    print(f"Reading {em_path.name} …")
    em_attrs = load_jsonl(em_path)
    em_table = pa.table(
        {"label": [a["label"] for a in em_attrs]},
        schema=SCHEMA_EM,
    )
    write_and_report(em_table, output / "test_attrs_em.parquet")

    # R: {"range_start": int, "range_end": int} → range_start, range_end (int)
    print(f"Reading {r_path.name} …")
    r_attrs = load_jsonl(r_path)
    r_table = pa.table(
        {
            "range_start": [a["range_start"] for a in r_attrs],
            "range_end": [a["range_end"] for a in r_attrs],
        },
        schema=SCHEMA_R,
    )
    write_and_report(r_table, output / "test_attrs_r.parquet")

    # EMIS: {"label": str} → label (str)
    print(f"Reading {emis_path.name} …")
    emis_attrs = load_jsonl(emis_path)
    emis_table = pa.table(
        {"label": [a["label"] for a in emis_attrs]},
        schema=SCHEMA_EMIS,
    )
    write_and_report(emis_table, output / "test_attrs_emis.parquet")


def write_and_report(table: pa.Table, path: Path) -> None:
    pq.write_table(table, path, row_group_size=ROW_GROUP_SIZE)
    size = os.path.getsize(path)
    print(f"  Wrote {path.name}: {table.num_rows:,} rows ({size / 1024:.1f} KB)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert arxiv-for-fanns query attribute JSONL files to parquet.",
    )
    parser.add_argument(
        "--em-file",
        type=str,
        default="arxiv-for-fanns-large-em_query_attributes.jsonl",
        help="Path to the EM query attributes JSONL file.",
    )
    parser.add_argument(
        "--r-file",
        type=str,
        default="arxiv-for-fanns-large-r_query_attributes.jsonl",
        help="Path to the R query attributes JSONL file.",
    )
    parser.add_argument(
        "--emis-file",
        type=str,
        default="arxiv-for-fanns-large-emis_query_attributes.jsonl",
        help="Path to the EMIS query attributes JSONL file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="files/arxiv-for-fanns-1024-euclidean",
        help="Directory for the output attribute parquet files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    extract_attrs(args.em_file, args.r_file, args.emis_file, args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
