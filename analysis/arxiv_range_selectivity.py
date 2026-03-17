"""Compute the selectivity (number of rows passing the filter) for each of the 10,000
range queries in test_attrs_r.parquet, for both the original and sorted datasets.

Outputs a JSON file with per-query counts.
"""

import json
import duckdb
import polars as pl

DATASET_BASE = "/Users/user/vectordb_bench/dataset"
ORIG_DIR = f"{DATASET_BASE}/c_arxivforfanns/c_arxivforfanns_medium_1m"
SORTED_DIR = f"{DATASET_BASE}/c_arxivforfannssortedbyupdatedate/c_arxivforfannssortedbyupdatedate_medium_1m"

OUTPUT_FILE = "arxiv_range_selectivity.json"


def load_train_labels(con: duckdb.DuckDBPyConnection, directory: str, table_name: str):
    """Load r_labels.parquet into a DuckDB table with a single 'label' column."""
    r_labels_path = f"{directory}/r_labels.parquet"
    con.execute(f"CREATE TABLE {table_name} AS SELECT update_date AS label FROM read_parquet('{r_labels_path}')")
    count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    print(f"  Loaded {count} rows into {table_name}")
    return count


def compute_selectivities(con: duckdb.DuckDBPyConnection, table_name: str, test_attrs: list[dict]) -> list[int]:
    """Run a COUNT(*) query for each range query and return the list of counts."""
    counts = []
    for attrs in test_attrs:
        result = con.execute(
            f"SELECT COUNT(*) FROM {table_name} WHERE label BETWEEN {attrs['range_start']} AND {attrs['range_end']}"
        ).fetchone()[0]
        counts.append(result)
    return counts


if __name__ == "__main__":
    # Load test attributes (same for both datasets — queries don't change)
    test_attrs_path = f"{ORIG_DIR}/test_attrs_r.parquet"
    test_attrs = pl.read_parquet(test_attrs_path).to_dicts()
    print(f"Loaded {len(test_attrs)} test queries")

    con = duckdb.connect()

    print("Loading original dataset labels...")
    total_orig = load_train_labels(con, ORIG_DIR, "orig")

    print("Loading sorted dataset labels...")
    total_sorted = load_train_labels(con, SORTED_DIR, "sorted")

    print("Computing selectivities for original dataset...")
    counts_orig = compute_selectivities(con, "orig", test_attrs)

    print("Computing selectivities for sorted dataset...")
    counts_sorted = compute_selectivities(con, "sorted", test_attrs)

    con.close()

    # Sanity check: counts should be identical (same data, just reordered)
    assert counts_orig == counts_sorted, "Selectivities differ between original and sorted — data mismatch!"
    print("Verified: selectivities match between original and sorted datasets")

    output = {
        "total_rows": total_orig,
        "num_queries": len(test_attrs),
        "queries": [
            {
                "range_start": attrs["range_start"],
                "range_end": attrs["range_end"],
                "count": count,
                "selectivity": count / total_orig,
            }
            for attrs, count in zip(test_attrs, counts_orig)
        ],
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    counts_arr = counts_orig
    print(f"\nSaved {OUTPUT_FILE}")
    print(f"  Min count: {min(counts_arr)}, Max count: {max(counts_arr)}")
    print(f"  Mean count: {sum(counts_arr) / len(counts_arr):.1f}")
    print(f"  Min selectivity: {min(counts_arr) / total_orig:.4f}, Max selectivity: {max(counts_arr) / total_orig:.4f}")
