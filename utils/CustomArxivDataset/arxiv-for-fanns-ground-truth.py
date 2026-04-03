"""
Compute ground truth for 4 query types (non-filtered + 3 filtered) using
DuckDB exact nearest-neighbor search over the PCA'd arxiv-for-fanns dataset (http://arxiv.org/abs/2507.21989).

Outputs one JSON file per query type:
  { "0": [id, id, ...], "1": [id, id, ...], ... }
"""

import json

import duckdb
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent

# --- Configuration -----------------------------------------------------------
K = 100
QUERY_COUNT = 10_000
DIMS = 1024

DATASET_PARQUET = SCRIPT_DIR / "files" / "arxiv-for-fanns-train.parquet"
QUERY_PARQUET = SCRIPT_DIR / "files" / "arxiv-for-fanns-query.parquet"

EM_QUERY_ATTRS = SCRIPT_DIR / "arxiv-for-fanns-large-em_query_attributes.jsonl"
R_QUERY_ATTRS = SCRIPT_DIR / "arxiv-for-fanns-large-r_query_attributes.jsonl"
EMIS_QUERY_ATTRS = SCRIPT_DIR / "arxiv-for-fanns-large-emis_query_attributes.jsonl"

OUTPUT_DIR = SCRIPT_DIR / "files"
OUTPUT_GT = OUTPUT_DIR / "arxiv-for-fanns-ground-truth.json"
OUTPUT_GT_EM = OUTPUT_DIR / "arxiv-for-fanns-ground-truth-em.json"
OUTPUT_GT_R = OUTPUT_DIR / "arxiv-for-fanns-ground-truth-r.json"
OUTPUT_GT_EMIS = OUTPUT_DIR / "arxiv-for-fanns-ground-truth-emis.json"


def load_query_vectors(path: Path) -> list[list[float]]:
    table = pq.read_table(path, columns=["id", "emb"])
    ids = table.column("id").to_pylist()
    embs = table.column("emb").to_pylist()
    order = sorted(range(len(ids)), key=lambda i: ids[i])
    return [embs[i] for i in order]


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def fmt_vector_literal(vec: list[float]) -> str:
    inner = ", ".join(f"{v}" for v in vec)
    return f"[{inner}]::FLOAT[{DIMS}]"


def run_ground_truth(
    con: duckdb.DuckDBPyConnection,
    query_vectors: list[list[float]],
    label: str,
    where_clauses: list[str | None],
    output_path: Path,
) -> None:
    """Run ground truth queries and write results to JSON."""
    assert len(query_vectors) == len(where_clauses) == QUERY_COUNT

    results: dict[str, list[int]] = {}

    for i in tqdm(range(QUERY_COUNT), desc=label, unit="q"):
        vec_literal = fmt_vector_literal(query_vectors[i])
        where = where_clauses[i]

        if where:
            sql = (
                f"SELECT id FROM dataset "
                f"WHERE {where} "
                f"ORDER BY array_distance(emb, {vec_literal}) "
                f"LIMIT {K}"
            )
        else:
            sql = (
                f"SELECT id FROM dataset "
                f"ORDER BY array_distance(emb, {vec_literal}) "
                f"LIMIT {K}"
            )

        rows = con.execute(sql).fetchall()
        results[str(i)] = [row[0] for row in rows]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f)
    print(f"  → {output_path}")


def main() -> None:
    print("Loading query vectors …")
    query_vectors = load_query_vectors(QUERY_PARQUET)
    assert len(query_vectors) == QUERY_COUNT

    print("Loading query attributes …")
    em_attrs = load_jsonl(EM_QUERY_ATTRS)
    r_attrs = load_jsonl(R_QUERY_ATTRS)
    emis_attrs = load_jsonl(EMIS_QUERY_ATTRS)
    assert len(em_attrs) == len(r_attrs) == len(emis_attrs) == QUERY_COUNT

    print("Loading dataset into DuckDB …")
    con = duckdb.connect(":memory:")
    con.execute(
        f"CREATE TABLE dataset AS "
        f"SELECT * REPLACE (emb::FLOAT[{DIMS}] AS emb) "
        f"FROM '{DATASET_PARQUET}'"
    )
    row_count = con.execute("SELECT COUNT(*) FROM dataset").fetchone()[0]
    print(f"  → {row_count:,} rows")

    # --- Non-filtered ---------------------------------------------------------
    print("Computing ground truth: non-filtered …")
    no_filter = [None] * QUERY_COUNT
    run_ground_truth(con, query_vectors, "non-filtered", no_filter, OUTPUT_GT)

    # --- EM (Exact Match on number_of_sub_categories) -------------------------
    print("Computing ground truth: EM …")
    em_wheres = [f"number_of_sub_categories = {attrs['label']}" for attrs in em_attrs]
    run_ground_truth(con, query_vectors, "EM", em_wheres, OUTPUT_GT_EM)

    # --- R (Range on update_date) ---------------------------------------------
    print("Computing ground truth: R …")
    r_wheres = [
        f"update_date BETWEEN {attrs['range_start']} AND {attrs['range_end']}"
        for attrs in r_attrs
    ]
    run_ground_truth(con, query_vectors, "R", r_wheres, OUTPUT_GT_R)

    # --- EMIS (Exact Match In Set on main_categories) -------------------------
    print("Computing ground truth: EMIS …")
    emis_wheres = [
        f"list_contains(main_categories, '{attrs['label']}')" for attrs in emis_attrs
    ]
    run_ground_truth(con, query_vectors, "EMIS", emis_wheres, OUTPUT_GT_EMIS)

    con.close()
    print("Done.")


if __name__ == "__main__":
    main()
