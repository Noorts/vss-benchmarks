#!/usr/bin/env python3
"""Remap label and neighbor files using an id_mapping.json produced by sort_arxiv_by_date.py.

Reads the mapping from the output directory, then for each file:

  Labels (em_labels, r_labels, emis_labels):
    - Row at old position k gets its id replaced with old_to_new[k]
    - Rows are then sorted by new id so row position matches id again

  Neighbors (neighbors, neighbors_em, neighbors_r, neighbors_emis):
    - Each id in neighbors_id lists is replaced via old_to_new[old_id]
    - The row id (test query id) is left unchanged (test data is not reordered)

  Test files (test.parquet, test_attrs_*.parquet):
    - Copied as-is (test data is not reordered)

Usage:
    python scripts/remap_arxiv_files.py \
        --input-dir  /path/to/c_arxivforfanns_medium_1m \
        --output-dir /path/to/c_arxivforfannsupdatedatesorted_medium_1m
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil

import polars as pl


def main() -> None:
    parser = argparse.ArgumentParser(description="Remap labels and neighbors using id_mapping.json")
    parser.add_argument(
        "--input-dir",
        type=pathlib.Path,
        required=True,
        help="Directory containing the original dataset files",
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
        old_to_new: dict[int, int] = {int(k): v for k, v in json.load(f)["old_to_new"].items()}
    print(f"Loaded mapping with {len(old_to_new):,} entries")

    # Build a polars Series for vectorized remapping of neighbor lists
    max_id = max(old_to_new.keys())
    remap_array = [0] * (max_id + 1)
    for old_id, new_id in old_to_new.items():
        remap_array[old_id] = new_id
    remap_series = pl.Series("remap", remap_array, dtype=pl.Int64)

    # --- Remap label files ---
    label_files = ["em_labels.parquet", "r_labels.parquet", "emis_labels.parquet"]
    for name in label_files:
        src = input_dir / name
        if not src.exists():
            print(f"Skipping {name} (not found)")
            continue

        df = pl.read_parquet(src)
        # Replace old ids with new ids, then sort by new id
        new_ids = [old_to_new[old_id] for old_id in df["id"].to_list()]
        df = df.with_columns(pl.Series("id", new_ids)).sort("id")

        out = output_dir / name
        df.write_parquet(out)
        print(f"Remapped {name}: {len(df):,} rows")

    # --- Remap neighbor files ---
    neighbor_files = [
        "neighbors.parquet",
        "neighbors_em.parquet",
        "neighbors_r.parquet",
        "neighbors_emis.parquet",
    ]
    for name in neighbor_files:
        src = input_dir / name
        if not src.exists():
            print(f"Skipping {name} (not found)")
            continue

        df = pl.read_parquet(src)
        # Remap each id inside the neighbors_id lists
        remapped_lists = [
            [remap_array[old_id] for old_id in row]
            for row in df["neighbors_id"].to_list()
        ]
        df = df.with_columns(pl.Series("neighbors_id", remapped_lists, dtype=pl.List(pl.Int64)))

        out = output_dir / name
        df.write_parquet(out)
        print(f"Remapped {name}: {len(df):,} rows")

    # --- Copy test files as-is ---
    test_files = [
        "test.parquet",
        "test_attrs_em.parquet",
        "test_attrs_r.parquet",
        "test_attrs_emis.parquet",
    ]
    for name in test_files:
        src = input_dir / name
        if not src.exists():
            print(f"Skipping {name} (not found)")
            continue

        shutil.copy2(src, output_dir / name)
        print(f"Copied {name}")

    print("\nDone! All files written to", output_dir)


if __name__ == "__main__":
    main()
