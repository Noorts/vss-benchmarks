"""
Remap neighbor files for the sorted Cohere6M dataset using id_mapping.json
produced by sort_cohere_by_id.py.

Each neighbor ID in neighbors_id lists is replaced via old_to_new[old_id].
The row id (test query id) is left unchanged (test data is not reordered).

Usage:
    python remap_cohere_neighbors.py \
        --input-dir  /path/to/c_cohere6m_medium_6m \
        --output-dir /path/to/c_cohere6msorted_medium_6m
"""

from __future__ import annotations

import argparse
import json
import pathlib

import polars as pl


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remap neighbor files using id_mapping.json"
    )
    parser.add_argument(
        "--input-dir",
        type=pathlib.Path,
        required=True,
        help="Directory containing the shuffled dataset's neighbor files",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        required=True,
        help="Directory containing sorted train files and id_mapping.json",
    )
    args = parser.parse_args()

    input_dir: pathlib.Path = args.input_dir
    output_dir: pathlib.Path = args.output_dir

    # --- Load mapping ---
    mapping_path = output_dir / "id_mapping.json"
    with open(mapping_path) as f:
        old_to_new: dict[int, int] = {
            int(k): v for k, v in json.load(f)["old_to_new"].items()
        }
    print(f"Loaded mapping with {len(old_to_new):,} entries")

    # Build remap array for fast lookup
    max_id = max(old_to_new.keys())
    remap_array = [0] * (max_id + 1)
    for old_id, new_id in old_to_new.items():
        remap_array[old_id] = new_id

    # --- Remap neighbor files ---
    neighbor_files = ["neighbors.parquet", "neighbors_r.parquet"]
    for name in neighbor_files:
        src = input_dir / name
        if not src.exists():
            print(f"Skipping {name} (not found)")
            continue

        df = pl.read_parquet(src)
        remapped_lists = [
            [remap_array[old_id] for old_id in row]
            for row in df["neighbors_id"].to_list()
        ]
        df = df.with_columns(
            pl.Series("neighbors_id", remapped_lists, dtype=pl.List(pl.Int64))
        )

        out = output_dir / name
        df.write_parquet(out)
        print(f"Remapped {name}: {len(df):,} rows → {out}")

    print("Done.")


if __name__ == "__main__":
    main()
