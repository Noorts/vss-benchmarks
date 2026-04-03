"""
Sort Cohere6M train data by original_id and produce an old->new ID mapping.

Memory-efficient: reads one input file at a time, uses numpy index arrays
for the sort, and writes output file by file.

Usage:
    python sort_cohere_by_id.py \
        --input-dir  /path/to/cohere_medium_6m \
        --output-dir /path/to/cohere_sorted_medium_6m
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROW_GROUP_SIZE = 122_880


def main() -> None:
    parser = argparse.ArgumentParser(description="Sort Cohere train data by original_id")
    parser.add_argument("--input-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()

    input_dir: pathlib.Path = args.input_dir
    output_dir: pathlib.Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Compute sort order from r_labels (small) ---
    print("Reading r_labels.parquet ...")
    r_labels = pq.read_table(input_dir / "r_labels.parquet")
    original_ids = r_labels.column("original_id").to_numpy()
    total = len(original_ids)
    print(f"  {total:,} rows")

    # sort_order[i] = old_id that should go to position i in sorted output
    sort_order = np.argsort(original_ids, kind="stable")
    sorted_original_ids = original_ids[sort_order]

    # old->new mapping
    inv = np.empty_like(sort_order)
    inv[sort_order] = np.arange(total)
    old_to_new = {int(old): int(inv[old]) for old in range(total)}

    print("Computed sort order")

    # --- 2. Get input file info ---
    train_files = sorted(input_dir.glob("shuffle_train-*-of-*.parquet"))
    if not train_files:
        raise FileNotFoundError(f"No shuffle_train files found in {input_dir}")

    split_sizes = []
    file_offsets = [0]  # cumulative start offset for each file
    for f in train_files:
        n = pq.read_metadata(f).num_rows
        split_sizes.append(n)
        file_offsets.append(file_offsets[-1] + n)
        print(f"  {f.name}: {n:,} rows")

    # --- 3. Write sorted output file by file ---
    # For each output chunk, determine which old_ids we need, grouped by source file.
    n_splits = len(split_sizes)
    out_offset = 0

    for out_idx, size in enumerate(split_sizes):
        chunk_sort = sort_order[out_offset:out_offset + size]

        # Group needed indices by source file
        source_file_idx = np.searchsorted(file_offsets[1:], chunk_sort, side="right")
        local_indices = chunk_sort - np.array(file_offsets)[source_file_idx]

        # Collect embeddings from each source file needed for this chunk.
        # We read source files one at a time and pick the needed rows.
        needed_by_file: dict[int, list[int]] = {}
        position_by_file: dict[int, list[int]] = {}
        for pos_in_chunk, (fid, lid) in enumerate(zip(source_file_idx, local_indices)):
            fid = int(fid)
            needed_by_file.setdefault(fid, []).append(int(lid))
            position_by_file.setdefault(fid, []).append(pos_in_chunk)

        # Allocate output arrays
        embs_out = [None] * size

        for fid in sorted(needed_by_file.keys()):
            local_idxs = needed_by_file[fid]
            positions = position_by_file[fid]

            source_table = pq.read_table(train_files[fid], columns=["emb"])
            emb_col = source_table.column("emb")

            # Use direct indexing instead of take — pyarrow's take on
            # list arrays can return wrong results for large indices.
            for i, pos in enumerate(positions):
                embs_out[pos] = emb_col[local_idxs[i]].as_py()

            del source_table, emb_col

        chunk = pa.table({
            "id": pa.array(range(out_offset, out_offset + size), type=pa.int64()),
            "emb": pa.array(embs_out, type=pa.list_(pa.float64())),
        })
        del embs_out

        out_name = f"train-{out_idx:02d}-of-{n_splits}.parquet"
        out_path = output_dir / out_name
        pq.write_table(chunk, out_path, row_group_size=ROW_GROUP_SIZE)
        print(f"Wrote {out_path.name}: {chunk.num_rows:,} rows (ids {out_offset}..{out_offset + size - 1})")
        del chunk
        out_offset += size

    # --- 4. Write r_labels for sorted dataset ---
    new_r_labels = pa.table({
        "id": pa.array(range(total), type=pa.int64()),
        "original_id": pa.array(sorted_original_ids.tolist(), type=pa.int64()),
    })
    r_labels_path = output_dir / "r_labels.parquet"
    pq.write_table(new_r_labels, r_labels_path, row_group_size=ROW_GROUP_SIZE)
    print(f"Wrote {r_labels_path.name}: {new_r_labels.num_rows:,} rows")

    # --- 5. Write ID mapping ---
    mapping_path = output_dir / "id_mapping.json"
    with open(mapping_path, "w") as f:
        json.dump({"old_to_new": old_to_new}, f)
    print(f"Wrote {mapping_path} ({len(old_to_new):,} entries)")

    # Sanity checks
    print(f"\nSanity check:")
    is_sorted = all(sorted_original_ids[i] <= sorted_original_ids[i+1] for i in range(min(1000, total-1)))
    print(f"  original_id is sorted (first 1000): {is_sorted}")
    assert len(set(old_to_new.values())) == len(old_to_new), "Mapping is not a bijection!"
    print("  Mapping is a valid bijection")


if __name__ == "__main__":
    main()
