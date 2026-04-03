"""
Compute ground truth for Cohere6M dataset: non-filtered and range-filtered
nearest neighbors using DuckDB exact cosine similarity search.

The ground truth is independent of dataset ordering (shuffled vs sorted),
so the output files can be used for both variants.

Outputs:
  - neighbors.parquet:   top-K neighbors for each query (no filter)
  - neighbors_r.parquet: top-K neighbors for each query (range filter on original_id)

Usage:
  python compute_cohere_ground_truth.py \
      --train-dir /path/to/cohere_medium_6m \
      --output-dir /path/to/output
"""

import argparse
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

K = 100
DIMS = 768
ROW_GROUP_SIZE = 122_880


def fmt_vector_literal(vec: list[float]) -> str:
    inner = ", ".join(f"{v}" for v in vec)
    return f"[{inner}]::FLOAT[{DIMS}]"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Cohere6M ground truth")
    parser.add_argument("--train-dir", type=str, required=True, help="Directory with train files and r_labels.parquet")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    args = parser.parse_args()

    train_dir = Path(args.train_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load test queries and attrs ---
    print("Loading test queries ...")
    test_table = pq.read_table(train_dir / "test.parquet")
    test_ids = test_table.column("id").to_pylist()
    test_embs = test_table.column("emb").to_pylist()
    query_count = len(test_ids)
    print(f"  {query_count:,} queries")

    print("Loading test_attrs_r.parquet ...")
    attrs_table = pq.read_table(train_dir / "test_attrs_r.parquet")
    r_attrs = [
        {"range_start": attrs_table.column("range_start")[i].as_py(),
         "range_end": attrs_table.column("range_end")[i].as_py()}
        for i in range(attrs_table.num_rows)
    ]
    assert len(r_attrs) == query_count

    # --- Load dataset into DuckDB ---
    print("Loading dataset into DuckDB ...")
    con = duckdb.connect(":memory:")

    # Load train files
    train_files = sorted(train_dir.glob("shuffle_train-*-of-*.parquet"))
    if not train_files:
        train_files = sorted(train_dir.glob("train-*-of-*.parquet"))
    train_glob = str(train_dir / "shuffle_train-*-of-*.parquet")
    if not list(train_dir.glob("shuffle_train-*-of-*.parquet")):
        train_glob = str(train_dir / "train-*-of-*.parquet")

    con.execute(
        f"CREATE TABLE dataset AS "
        f"SELECT id, emb::FLOAT[{DIMS}] AS emb "
        f"FROM read_parquet('{train_glob}')"
    )
    row_count = con.execute("SELECT COUNT(*) FROM dataset").fetchone()[0]
    print(f"  {row_count:,} train rows")

    # Load r_labels and add original_id column
    r_labels_path = str(train_dir / "r_labels.parquet")
    con.execute(
        f"CREATE TABLE labels AS SELECT * FROM read_parquet('{r_labels_path}')"
    )
    con.execute(
        "ALTER TABLE dataset ADD COLUMN original_id BIGINT"
    )
    con.execute(
        "UPDATE dataset SET original_id = (SELECT original_id FROM labels WHERE labels.id = dataset.id)"
    )
    con.execute("DROP TABLE labels")
    print("  Added original_id column")

    # --- Non-filtered ground truth ---
    print("Computing non-filtered ground truth ...")
    nf_ids = []
    nf_neighbors = []
    for i in tqdm(range(query_count), desc="non-filtered", unit="q"):
        vec_lit = fmt_vector_literal(test_embs[i])
        sql = (
            f"SELECT id FROM dataset "
            f"ORDER BY array_cosine_similarity(emb, {vec_lit}) DESC "
            f"LIMIT {K}"
        )
        rows = con.execute(sql).fetchall()
        nf_ids.append(test_ids[i])
        nf_neighbors.append([r[0] for r in rows])

    nf_table = pa.table({
        "id": pa.array(nf_ids, type=pa.int64()),
        "neighbors_id": pa.array(nf_neighbors, type=pa.large_list(pa.int64())),
    })
    nf_path = output_dir / "neighbors.parquet"
    pq.write_table(nf_table, nf_path, row_group_size=ROW_GROUP_SIZE)
    print(f"  Wrote {nf_path.name}: {nf_table.num_rows:,} rows")

    # --- Range-filtered ground truth ---
    print("Computing range-filtered ground truth ...")
    rf_ids = []
    rf_neighbors = []
    for i in tqdm(range(query_count), desc="range-filtered", unit="q"):
        vec_lit = fmt_vector_literal(test_embs[i])
        rs = r_attrs[i]["range_start"]
        re = r_attrs[i]["range_end"]
        sql = (
            f"SELECT id FROM dataset "
            f"WHERE original_id BETWEEN {rs} AND {re} "
            f"ORDER BY array_cosine_similarity(emb, {vec_lit}) DESC "
            f"LIMIT {K}"
        )
        rows = con.execute(sql).fetchall()
        rf_ids.append(test_ids[i])
        rf_neighbors.append([r[0] for r in rows])

    rf_table = pa.table({
        "id": pa.array(rf_ids, type=pa.int64()),
        "neighbors_id": pa.array(rf_neighbors, type=pa.large_list(pa.int64())),
    })
    rf_path = output_dir / "neighbors_r.parquet"
    pq.write_table(rf_table, rf_path, row_group_size=ROW_GROUP_SIZE)
    print(f"  Wrote {rf_path.name}: {rf_table.num_rows:,} rows")

    con.close()
    print("Done.")


if __name__ == "__main__":
    main()
