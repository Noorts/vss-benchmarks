#!/usr/bin/env python3
"""Randomly shuffle ArxivForFANNs train data and produce an old->new ID mapping.

Reads the original train-*.parquet files, shuffles all rows randomly (with a
fixed seed for reproducibility), reassigns sequential IDs 0..N-1, and writes:

  1. Shuffled train parquet files (same split sizes as originals)
  2. id_mapping.json  — { "old_to_new": { old_id: new_id, ... } }

The mapping file can then be used by remap_arxiv_files.py to remap labels and
ground-truth neighbor files without recomputation.

Usage:
    python utils/shuffle_arxiv.py \
        --input-dir  /path/to/c_arxivforfanns_medium_1m \
        --output-dir /path/to/c_arxivforfannsrandom_medium_1m
"""

from __future__ import annotations

import argparse
import json
import pathlib

import polars as pl


def main() -> None:
    parser = argparse.ArgumentParser(description="Randomly shuffle Arxiv train data")
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
        help="Directory to write shuffled train files and id_mapping.json",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible shuffling (default: 42)",
    )
    args = parser.parse_args()

    input_dir: pathlib.Path = args.input_dir
    output_dir: pathlib.Path = args.output_dir
    seed: int = args.seed
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Read all train files and record original split sizes ---
    train_files = sorted(input_dir.glob("train-*-of-*.parquet"))
    if not train_files:
        raise FileNotFoundError(f"No train-*-of-*.parquet files found in {input_dir}")

    split_sizes: list[int] = []
    train_dfs: list[pl.DataFrame] = []
    for f in train_files:
        df = pl.read_parquet(f)
        split_sizes.append(len(df))
        train_dfs.append(df)
        print(f"Read {f.name}: {len(df):,} rows")

    train = pl.concat(train_dfs)
    del train_dfs
    print(f"Total train rows: {len(train):,}")

    # --- 2. Shuffle rows randomly ---
    train = train.sample(fraction=1.0, shuffle=True, seed=seed)

    # Build mapping: old_id -> new_id
    old_ids = train["id"].to_list()
    old_to_new = {old_id: new_id for new_id, old_id in enumerate(old_ids)}

    # Replace id column with new sequential IDs
    train = train.with_columns(pl.Series("id", list(range(len(train)))))

    print(f"Shuffled {len(train):,} rows (seed={seed})")

    # --- 3. Write shuffled train files (same split sizes) ---
    offset = 0
    n_splits = len(split_sizes)
    for i, size in enumerate(split_sizes):
        chunk = train.slice(offset, size)
        out_name = f"train-{i:02d}-of-{n_splits}.parquet"
        out_path = output_dir / out_name
        chunk.write_parquet(out_path)
        print(f"Wrote {out_path.name}: {len(chunk):,} rows (ids {offset}..{offset + size - 1})")
        offset += size

    # --- 4. Write ID mapping ---
    mapping_path = output_dir / "id_mapping.json"
    with open(mapping_path, "w") as f:
        json.dump({"old_to_new": old_to_new}, f)
    print(f"Wrote {mapping_path} ({len(old_to_new):,} entries)")

    # Quick sanity check
    print("\nSanity check:")
    print(f"  old id 0 -> new id {old_to_new[0]}")
    print(f"  old id {len(old_to_new)-1} -> new id {old_to_new[len(old_to_new)-1]}")
    # Verify the mapping is a bijection
    assert len(set(old_to_new.values())) == len(old_to_new), "Mapping is not a bijection!"
    print("  Mapping is a valid bijection ✓")


if __name__ == "__main__":
    main()
