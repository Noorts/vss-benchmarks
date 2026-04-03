"""
Generate test queries with range filters for Cohere6M filtered search benchmark.

For each of 21 scalar_int_rates, generates 500 range filters that match exactly
that selectivity percentage of rows. Query vectors are randomly sampled (with
replacement) from the Cohere10M test queries (1,000 vectors).

All 10,500 queries are shuffled before writing.

Outputs (written to --output-dir):
  - test.parquet:       id (0..10499), emb
  - test_attrs_r.parquet: range_start, range_end
  - test_queries_selectivity.json

Usage:
  python generate_cohere_test_queries.py \
      --labels /path/to/cohere_medium_6m/r_labels.parquet \
      --source-test /path/to/cohere_large_10m/test.parquet \
      --output-dir /path/to/output
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROW_GROUP_SIZE = 122_880

SCALAR_INT_RATES = [
    0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5,
    0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999,
]
QUERIES_PER_RATE = 500
SEED = 42


def generate_range_filters(
    sorted_original_ids: np.ndarray,
    total: int,
    rate: float,
    count: int,
    rng: random.Random,
) -> list[dict]:
    """Generate `count` range filters each matching exactly `target` rows.

    Since original_ids are non-contiguous (sampled from 0..9,999,999),
    we pick a contiguous window of `target` elements in the sorted original_id
    array and use the first/last values as range_start/range_end.
    """
    target = round(rate * total)
    if target < 1:
        target = 1
    if target > total:
        target = total

    max_start_idx = total - target
    filters = []
    for _ in range(count):
        start_idx = rng.randint(0, max_start_idx)
        end_idx = start_idx + target - 1
        range_start = int(sorted_original_ids[start_idx])
        range_end = int(sorted_original_ids[end_idx])

        # Verify: count how many original_ids fall in [range_start, range_end]
        actual = int(np.searchsorted(sorted_original_ids, range_end, side="right")
                     - np.searchsorted(sorted_original_ids, range_start, side="left"))
        assert actual == target, (
            f"Expected {target} rows in [{range_start}, {range_end}], got {actual}"
        )

        filters.append({
            "range_start": range_start,
            "range_end": range_end,
            "target_count": target,
            "selectivity": rate,
        })

    return filters


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Cohere6M test queries with range filters")
    parser.add_argument("--labels", type=str, required=True, help="Path to r_labels.parquet")
    parser.add_argument("--source-test", type=str, required=True, help="Path to Cohere10M test.parquet")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(SEED)
    np_rng = np.random.RandomState(SEED)

    # --- 1. Load original_ids and sort them ---
    print("Loading r_labels.parquet ...")
    r_labels = pq.read_table(args.labels)
    original_ids = r_labels.column("original_id").to_numpy()
    total = len(original_ids)
    sorted_oids = np.sort(original_ids)
    print(f"  {total:,} rows, original_id range: {sorted_oids[0]} - {sorted_oids[-1]}")

    # --- 2. Generate range filters for each rate ---
    print("Generating range filters ...")
    all_queries = []
    for rate in SCALAR_INT_RATES:
        filters = generate_range_filters(sorted_oids, total, rate, QUERIES_PER_RATE, rng)
        all_queries.extend(filters)
        target = round(rate * total)
        print(f"  rate={rate:.3f}: {QUERIES_PER_RATE} filters, target_count={target:,}")

    total_queries = len(all_queries)
    print(f"Total queries: {total_queries:,}")

    # --- 3. Sample query vectors ---
    print("Loading source test vectors ...")
    source_test = pq.read_table(args.source_test)
    source_embs = source_test.column("emb")
    n_source = len(source_embs)
    print(f"  {n_source} source test vectors")

    query_indices = np_rng.randint(0, n_source, size=total_queries)

    # --- 4. Shuffle everything together ---
    print("Shuffling queries ...")
    shuffle_perm = list(range(total_queries))
    rng.shuffle(shuffle_perm)

    shuffled_queries = [all_queries[i] for i in shuffle_perm]
    shuffled_emb_indices = query_indices[shuffle_perm]

    # --- 5. Write test.parquet ---
    test_ids = pa.array(range(total_queries), type=pa.int64())
    test_embs = source_embs.take(pa.array(shuffled_emb_indices, type=pa.int32()))

    test_table = pa.table({"id": test_ids, "emb": test_embs})
    test_path = output_dir / "test.parquet"
    pq.write_table(test_table, test_path, row_group_size=ROW_GROUP_SIZE)
    print(f"Wrote {test_path.name}: {test_table.num_rows:,} rows")

    # --- 6. Write test_attrs_r.parquet ---
    range_starts = pa.array([q["range_start"] for q in shuffled_queries], type=pa.int64())
    range_ends = pa.array([q["range_end"] for q in shuffled_queries], type=pa.int64())
    attrs_table = pa.table({"range_start": range_starts, "range_end": range_ends})
    attrs_path = output_dir / "test_attrs_r.parquet"
    pq.write_table(attrs_table, attrs_path, row_group_size=ROW_GROUP_SIZE)
    print(f"Wrote {attrs_path.name}: {attrs_table.num_rows:,} rows")

    # --- 7. Write selectivity JSON ---
    selectivity_data = {
        "total_rows": total,
        "num_queries": total_queries,
        "queries": [
            {
                "range_start": q["range_start"],
                "range_end": q["range_end"],
                "count": q["target_count"],
                "selectivity": q["selectivity"],
            }
            for q in shuffled_queries
        ],
    }
    sel_path = output_dir / "test_queries_selectivity.json"
    with open(sel_path, "w") as f:
        json.dump(selectivity_data, f, indent=2)
    print(f"Wrote {sel_path.name}")

    print("Done.")


if __name__ == "__main__":
    main()
