"""
Remove test set rows from train.parquet based on the original_id column in test.parquet.

Usage:
  python filter_train.py <dataset_dir>

Reads train.parquet and test.parquet from the dataset directory.
Overwrites train.parquet with the filtered version.
"""

import sys
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq

dataset_dir = Path(sys.argv[1])
train_path = dataset_dir / "train.parquet"
test_path = dataset_dir / "test.parquet"

print(f"Loading {test_path}...")
test_table = pq.read_table(test_path, columns=["original_id"])
exclude_ids = set(test_table.column("original_id").to_pylist())
print(f"  {len(exclude_ids)} ids to exclude")

print(f"Loading {train_path}...")
train_table = pq.read_table(train_path)
original_count = train_table.num_rows
print(f"  {original_count:,} rows")

mask = pc.invert(pc.is_in(train_table.column("id"), value_set=test_table.column("original_id")))
filtered = train_table.filter(mask)
print(f"  {filtered.num_rows:,} rows after filtering ({original_count - filtered.num_rows} removed)")

pq.write_table(filtered, train_path)
print(f"Wrote {train_path}")
