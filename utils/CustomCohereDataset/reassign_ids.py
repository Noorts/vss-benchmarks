"""
Rewrite shuffle_train parquet files with sequential IDs (0..N-1),
preserving the original row order.  The original sampled IDs are saved
as `original_id` in a separate r_labels.parquet file.

Usage:
  python reassign_ids.py <dataset_dir>

Overwrites the train files in-place and writes r_labels.parquet.
"""

import argparse
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROW_GROUP_SIZE = 122_880


def reassign(dataset_dir: Path) -> None:
    train_files = sorted(dataset_dir.glob("shuffle_train-*-of-*.parquet"))
    if not train_files:
        raise FileNotFoundError(f"No shuffle_train files found in {dataset_dir}")

    # Read all files, record split sizes.
    split_sizes: list[int] = []
    tables: list[pa.Table] = []
    for f in train_files:
        t = pq.read_table(f)
        split_sizes.append(t.num_rows)
        tables.append(t)
        print(f"Read {f.name}: {t.num_rows:,} rows")

    combined = pa.concat_tables(tables)
    del tables
    total = combined.num_rows
    print(f"Total: {total:,} rows")

    original_ids = combined.column("id")
    new_ids = pa.array(range(total), type=pa.int64())

    # Write r_labels.parquet: id (sequential), original_id.
    r_labels = pa.table({"id": new_ids, "original_id": original_ids})
    r_labels_path = dataset_dir / "r_labels.parquet"
    pq.write_table(r_labels, r_labels_path, row_group_size=ROW_GROUP_SIZE)
    print(f"Wrote {r_labels_path.name}: {r_labels.num_rows:,} rows "
          f"({os.path.getsize(r_labels_path) / 1024:.0f} KB)")

    # Replace id column with sequential IDs.
    combined = combined.drop("id").append_column("id", new_ids)
    # Reorder columns to (id, emb).
    combined = combined.select(["id", "emb"])

    # Write back in original split sizes.
    offset = 0
    n_splits = len(split_sizes)
    for i, size in enumerate(split_sizes):
        chunk = combined.slice(offset, size)
        out_path = train_files[i]
        pq.write_table(chunk, out_path, row_group_size=ROW_GROUP_SIZE)
        print(f"Wrote {out_path.name}: {chunk.num_rows:,} rows "
              f"(ids {offset}..{offset + size - 1})")
        offset += size

    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reassign sequential IDs to shuffle_train parquet files."
    )
    parser.add_argument("dataset_dir", type=str, help="Directory containing shuffle_train files.")
    args = parser.parse_args()
    reassign(Path(args.dataset_dir))


if __name__ == "__main__":
    main()
