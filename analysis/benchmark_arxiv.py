"""
Benchmark script for the Arxiv-for-FANNs dataset with range-filtered vector search.

Uses VectorDBBench Arxiv dataset files from disk and runs only queries whose
selectivity (from arxiv_range_selectivity.json) is within an inclusive range.
"""

import json
import os
import struct
import base64
import subprocess
import time
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from statistics import mean

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# ---------------------------------------------------------------------------
# Dataset paths
# ---------------------------------------------------------------------------
VECTORDBBENCH_DATASET_BASE = Path("/Users/user/vectordb_bench/dataset")
ARXIV_DATASET_DIRS = {
    "original": VECTORDBBENCH_DATASET_BASE
    / "c_arxivforfanns"
    / "c_arxivforfanns_medium_1m",
    "sorted_by_update_date": VECTORDBBENCH_DATASET_BASE
    / "c_arxivforfannssortedbyupdatedate"
    / "c_arxivforfannssortedbyupdatedate_medium_1m",
    "randomly_shuffled": VECTORDBBENCH_DATASET_BASE
    / "c_arxivforfannsrandom"
    / "c_arxivforfannsrandom_medium_1m",
}
SELECTIVITY_JSON = Path(__file__).parent / "arxiv_range_selectivity.json"

# ---------------------------------------------------------------------------
# PDXearch extension path
# ---------------------------------------------------------------------------
PDXEARCH_EXTENSION_DIR = Path("../../PDXearch/build")

# ---------------------------------------------------------------------------
# Blob encoding helpers (mirrors pdxearch_blob_codec.hpp)
# ---------------------------------------------------------------------------
BLOB_INV_SCALE = [100000.0, 10000.0, 1000.0, 100.0]


def encode_float_to_int16(value: float) -> int:
    for x in range(4):
        q = round(value * BLOB_INV_SCALE[x])
        if -8192 <= q <= 8191:
            return (q << 2) | x
    q = max(-8192, min(8191, round(value * BLOB_INV_SCALE[3])))
    return (q << 2) | 3


def encode_query_blob_base64(query_vec) -> str:
    encoded = struct.pack(
        f"<{len(query_vec)}h", *(encode_float_to_int16(float(v)) for v in query_vec)
    )
    return base64.b64encode(encoded).decode("ascii")


# ---------------------------------------------------------------------------
# Recall
# ---------------------------------------------------------------------------
def compute_recall(result_ids: list[int], ground_truth: list[int], K: int) -> float:
    if K > len(ground_truth):
        raise Exception(
            f"K is greater than the length of the ground truth: {K} > {len(ground_truth)}. Can't determine recall."
        )
    if len(result_ids) == 0:
        return 0.0
    ground_truth_ids = ground_truth[:K]
    return len(set(result_ids) & set(ground_truth_ids)) / len(ground_truth_ids)


# ---------------------------------------------------------------------------
# Profiler helpers
# ---------------------------------------------------------------------------
def get_scan_operator(result_json, operator_names: list[str]):
    def dfs_search(node):
        if (
            node.get("operator_name")
            and node["operator_name"].strip() in operator_names
        ):
            return node
        for child in node.get("children", []):
            result = dfs_search(child)
            if result is not None:
                return result
        return None

    return dfs_search(result_json)


def sum_non_index_operator_timings(
    result_json, index_operator_names: list[str]
) -> float:
    """Sum operator_timing for all operators except the index scan operator(s)."""

    def dfs_sum(node):
        op_name = (node.get("operator_name") or "").strip()
        op_timing = float(node.get("operator_timing") or 0.0)
        total = 0.0 if op_name in index_operator_names else op_timing
        for child in node.get("children", []):
            total += dfs_sum(child)
        return total

    return dfs_sum(result_json)


def estimate_non_execution_overhead(result_json) -> float:
    """
    Estimate root-level non-execution overhead included in latency.

    We subtract only top-level planner/optimizer parent metrics to avoid
    double-counting child metrics (e.g. planner_binding or optimizer_*).
    """
    return (
        float(result_json.get("all_optimizers") or 0.0)
        + float(result_json.get("planner") or 0.0)
        + float(result_json.get("physical_planner") or 0.0)
    )


def get_commit_hash():
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("utf-8")
            .strip()
        )
    except subprocess.CalledProcessError:
        return "unknown"


# ---------------------------------------------------------------------------
# Approach enum
# ---------------------------------------------------------------------------
class Approach(Enum):
    DUCKDB = auto()
    VSS = auto()
    PDXEARCH = auto()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_train_data(dataset_dir: Path) -> pd.DataFrame:
    """Load train embeddings and labels from parquet files."""
    train_parts = sorted(dataset_dir.glob("train-*.parquet"))
    train_df = pd.concat(
        [pq.read_table(p).to_pandas() for p in train_parts], ignore_index=True
    )

    labels_df = pq.read_table(dataset_dir / "r_labels.parquet").to_pandas()
    train_df = train_df.merge(labels_df, on="id")
    return train_df


def load_test_data(dataset_dir: Path):
    """Load test queries, range attributes, and ground truth for range-filtered search."""
    test_df = pq.read_table(dataset_dir / "test.parquet").to_pandas()
    test_attrs_df = pq.read_table(dataset_dir / "test_attrs_r.parquet").to_pandas()
    neighbors_df = pq.read_table(dataset_dir / "neighbors_r.parquet").to_pandas()
    return test_df, test_attrs_df, neighbors_df


def load_selectivity_data(min_selectivity, max_selectivity):
    """Load selectivity info and return indices of queries within [min_selectivity, max_selectivity]."""
    if min_selectivity > max_selectivity:
        raise ValueError(
            f"min_selectivity must be <= max_selectivity, got {min_selectivity} > {max_selectivity}"
        )
    with open(SELECTIVITY_JSON, "r") as f:
        sel_data = json.load(f)

    eligible_indices = []
    for idx, q in enumerate(sel_data["queries"]):
        if min_selectivity <= q["selectivity"] <= max_selectivity:
            eligible_indices.append(idx)

    return sel_data, eligible_indices


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ############################## MAIN SETTINGS ###############################

    DATABASE_PATH = "benchmark_arxiv.db"

    # Row ordering variant: "original", "sorted_by_update_date", or "randomly_shuffled"
    ARXIV_DATASET_ORDER = "sorted_by_update_date"

    BUILD_TYPE = "release"
    DEBUG = False
    PROFILING = True
    THREAD_COUNT = 1  # 0 = auto
    QUERY_K = [10]
    SEED = 0
    QUANTIZATION = "f32"
    USE_BLOB_INTERFACE = True
    SLEEP_TO_ATTACH = False

    SELECTIVITY_RANGE = [0.00, 0.05]  # Inclusive: [min_selectivity, max_selectivity]

    THREAD_COUNT_INDEX_CREATION = 14

    NORMALIZE = False
    APPROACH = Approach.PDXEARCH

    VSS_ENABLED = APPROACH == Approach.VSS
    VSS_EF_SEARCH = [64]
    VSS_EF_CONSTRUCTION = 128
    VSS_M = 16
    VSS_M0 = 2 * VSS_M

    PDXEARCH_ENABLED = APPROACH == Approach.PDXEARCH
    PDXEARCH_N_PROBE = [28]

    ############################################################################

    TABLE_NAME = "arxiv"
    TABLE_EMBEDDING_COLUMN_NAME = "vec"
    DATASET_DIMS = 1024
    NUMBER_OF_PROGRESS_UPDATES = 5
    PROFILE_WORKAROUND_OUTPUT_FILE = "temp_profile_output_arxiv"
    COMMIT_HASH = get_commit_hash()

    DATASET_DIR = ARXIV_DATASET_DIRS[ARXIV_DATASET_ORDER]

    print(f"Process ID: {os.getpid()}")
    print(f"Dataset order: {ARXIV_DATASET_ORDER} ({DATASET_DIR})")

    # Load selectivity data and determine eligible query indices
    sel_data, eligible_indices = load_selectivity_data(*SELECTIVITY_RANGE)
    queries_meta = sel_data["queries"]
    print(
        f"Eligible queries (selectivity in [{SELECTIVITY_RANGE[0]*100:.0f}%, {SELECTIVITY_RANGE[1]*100:.0f}%]): {len(eligible_indices)} / {sel_data['num_queries']}"
    )

    if len(eligible_indices) == 0:
        print("No queries match the selectivity threshold. Exiting.")
        return

    # Load test data
    print("Loading test data...")
    test_df, test_attrs_df, neighbors_df = load_test_data(DATASET_DIR)

    print(f"Approach: {APPROACH}")
    if APPROACH == Approach.PDXEARCH:
        print(f"PDXearch: n_probe={PDXEARCH_N_PROBE}")
    if APPROACH == Approach.VSS:
        print(
            f"VSS: ef_construction={VSS_EF_CONSTRUCTION}, ef_search={VSS_EF_SEARCH}, m={VSS_M}, m0={VSS_M0}"
        )

    # Load train data and create database
    print("Loading train data...")
    train_df = load_train_data(DATASET_DIR)
    DATASET_NUM_ROWS = len(train_df)
    print(f"Train data loaded: {DATASET_NUM_ROWS} rows")

    with duckdb.connect(
        DATABASE_PATH,
        config={"allow_unsigned_extensions": "true", "threads": THREAD_COUNT},
    ) as conn:
        conn.execute("SET late_materialization_max_rows = 0;")

        if VSS_ENABLED:
            conn.execute("install vss;")
            conn.execute("load vss;")
        if PDXEARCH_ENABLED or USE_BLOB_INTERFACE:
            conn.execute(
                f"load '{PDXEARCH_EXTENSION_DIR / BUILD_TYPE / 'extension/pdxearch/pdxearch.duckdb_extension'}';"
            )
        if PROFILING:
            conn.execute("SET explain_output=optimized_only;")
            conn.execute("SET enable_profiling=json;")
            conn.execute("SET profiling_mode=detailed;")
            conn.execute(f"SET profiling_output={PROFILE_WORKAROUND_OUTPUT_FILE};")

        # Create table from train data
        print("Creating table...")
        conn.execute(f"DROP TABLE IF EXISTS {TABLE_NAME};")
        conn.execute(
            f"""
            CREATE TABLE {TABLE_NAME} AS
            SELECT
                id,
                emb::FLOAT[{DATASET_DIMS}] AS {TABLE_EMBEDDING_COLUMN_NAME},
                update_date
            FROM train_df
            """
        )
        print(f"Table '{TABLE_NAME}' created with {DATASET_NUM_ROWS} rows.")

        if SLEEP_TO_ATTACH:
            print("Attach profiler please, I'm about to sleep for 5 seconds...")
            time.sleep(5)

        # Set threads for index creation
        conn.execute(f"SET threads={THREAD_COUNT_INDEX_CREATION};")

        if VSS_ENABLED:
            print(f"Creating VSS index using {THREAD_COUNT_INDEX_CREATION} threads...")
            conn.execute("SET hnsw_enable_experimental_persistence = true;")
            start_time = time.monotonic()
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS {TABLE_NAME}_vss_idx ON {TABLE_NAME} USING HNSW ({TABLE_EMBEDDING_COLUMN_NAME}) "
                f"WITH (ef_construction={VSS_EF_CONSTRUCTION}, ef_search={VSS_EF_SEARCH[0]}, m={VSS_M}, m0={VSS_M0});"
            )
            end_time = time.monotonic()
            print(f"VSS index creation time: {end_time - start_time:.6f} seconds")

        if PDXEARCH_ENABLED:
            print(
                f"Creating PDXearch index using {THREAD_COUNT_INDEX_CREATION} threads..."
            )
            conn.execute(f"DROP INDEX IF EXISTS {TABLE_NAME}_idx;")
            start_time = time.monotonic()
            conn.execute(
                f"CREATE INDEX {TABLE_NAME}_idx ON {TABLE_NAME} USING PDXEARCH ({TABLE_EMBEDDING_COLUMN_NAME}) "
                f"WITH (n_probe={PDXEARCH_N_PROBE[0]}{f', seed={SEED}' if SEED is not None else ''}, quantization='{QUANTIZATION}');"
            )
            end_time = time.monotonic()
            print(f"PDXearch index creation time: {end_time - start_time:.6f} seconds")

        if THREAD_COUNT != 0:
            conn.execute(f"SET threads={THREAD_COUNT};")
        else:
            conn.execute("RESET threads;")

        all_results_df = pd.DataFrame()

        runtime_parameter_arr = [0]
        if PDXEARCH_ENABLED:
            runtime_parameter_arr = PDXEARCH_N_PROBE
        elif VSS_ENABLED:
            runtime_parameter_arr = VSS_EF_SEARCH

        if PDXEARCH_ENABLED:
            conn.execute("SET pdxearch_n_probe=0;")

        # Warm up
        print("Warming up index...")
        if USE_BLOB_INTERFACE:
            warmup_blob = encode_query_blob_base64([0.0] * DATASET_DIMS)
            warmup_vec_literal = f"pdxearch_base64_to_blob('{warmup_blob}')"
        else:
            warmup_vec_literal = f"{[0] * DATASET_DIMS}::FLOAT[{DATASET_DIMS}]"
        warmup_query = (
            f"SELECT id, rowid FROM {TABLE_NAME} "
            f"WHERE update_date >= 14000 AND update_date <= 15000 "
            f"ORDER BY array_distance({TABLE_EMBEDDING_COLUMN_NAME}, {warmup_vec_literal}) "
            f"LIMIT {QUERY_K[0]};"
        )
        conn.execute(warmup_query).fetchall()
        print("Index warmed up")

        if SLEEP_TO_ATTACH:
            print("Starting search in 3 seconds...")
            time.sleep(3)

        num_eligible = len(eligible_indices)

        for current_query_k in QUERY_K:
            for runtime_parameter in runtime_parameter_arr:
                results = []

                if PDXEARCH_ENABLED:
                    conn.execute(f"SET pdxearch_n_probe={runtime_parameter};")
                elif VSS_ENABLED:
                    conn.execute(f"SET hnsw_ef_search={runtime_parameter};")

                for progress_idx, query_idx in enumerate(eligible_indices):
                    if (
                        progress_idx
                        % max(1, num_eligible // NUMBER_OF_PROGRESS_UPDATES)
                        == 0
                    ):
                        print(
                            f"Query {progress_idx} of {num_eligible} (original idx={query_idx})"
                        )

                    # Get query vector
                    query_vec = test_df.iloc[query_idx]["emb"]

                    # Get range filter attributes
                    range_start = int(test_attrs_df.iloc[query_idx]["range_start"])
                    range_end = int(test_attrs_df.iloc[query_idx]["range_end"])

                    # Get ground truth
                    ground_truth_ids = neighbors_df.iloc[query_idx]["neighbors_id"]

                    if USE_BLOB_INTERFACE:
                        query_vec_literal = f"pdxearch_base64_to_blob('{encode_query_blob_base64(query_vec)}')"
                    else:
                        query_vec_literal = f"{list(query_vec)}::FLOAT[{DATASET_DIMS}]"

                    query = (
                        f"SELECT id, rowid FROM {TABLE_NAME} "
                        f"WHERE update_date >= {range_start} AND update_date <= {range_end} "
                        f"ORDER BY array_distance({TABLE_EMBEDDING_COLUMN_NAME}, {query_vec_literal}) "
                        f"LIMIT {current_query_k};"
                    )

                    start_time = time.monotonic()
                    query_result = conn.execute(query).fetchall()
                    end_time = time.monotonic()
                    row_ids = [row[1] for row in query_result]

                    recall = compute_recall(row_ids, ground_truth_ids, current_query_k)

                    # Read profiling output
                    with open(PROFILE_WORKAROUND_OUTPUT_FILE, "r") as prof_file:
                        result_json = json.loads(prof_file.read())

                    index_scan_duration = 0
                    filtered_sequential_scan_duration = 0

                    if APPROACH == Approach.PDXEARCH or APPROACH == Approach.VSS:
                        index_scan_operator = get_scan_operator(
                            result_json,
                            [
                                "PDXEARCH_INDEX_SCAN",
                                "PDXEARCH_INDEX_FILT_SCAN",
                                "HNSW_INDEX_SCAN",
                            ],
                        )
                        if index_scan_operator is None:
                            raise Exception(
                                "Expected PDXEARCH_INDEX_SCAN, PDXEARCH_INDEX_FILT_SCAN, or HNSW_INDEX_SCAN operator, got None"
                            )
                        index_scan_duration = index_scan_operator["operator_timing"]

                    if APPROACH == Approach.PDXEARCH:
                        sequential_scan_operator = get_scan_operator(
                            result_json, ["SEQ_SCAN"]
                        )
                        if sequential_scan_operator is None:
                            raise Exception("Expected SEQ_SCAN operator, got None")
                        filtered_sequential_scan_duration = sequential_scan_operator[
                            "operator_timing"
                        ]

                    # DuckDB's reported index operator timing is not reliable for our purposes.
                    # Approximate index scan time as the residual of total query latency after:
                    # 1) root-level non-execution overhead (planning/optimization), and
                    # 2) execution time explicitly attributed to non-index operators.
                    calc_index_scan_duration = max(
                        0.0,
                        float(result_json["latency"])
                        - estimate_non_execution_overhead(result_json)
                        - sum_non_index_operator_timings(
                            result_json,
                            [
                                "PDXEARCH_INDEX_SCAN",
                                "PDXEARCH_INDEX_FILT_SCAN",
                                "HNSW_INDEX_SCAN",
                            ],
                        ),
                    )

                    results.append(
                        {
                            "query_idx": query_idx,
                            "selectivity": queries_meta[query_idx]["selectivity"],
                            "range_start": range_start,
                            "range_end": range_end,
                            "e2e_duration": end_time - start_time,
                            "index_scan_duration": index_scan_duration,
                            "filtered_sequential_scan_duration": filtered_sequential_scan_duration,
                            "calc_index_scan_duration": calc_index_scan_duration,
                            "latency": result_json["latency"],
                            "cpu_time": result_json["cpu_time"],
                            "system_peak_buffer_memory": result_json[
                                "system_peak_buffer_memory"
                            ],
                            "all_optimizers": result_json["all_optimizers"],
                            "optimizer_extension": result_json["optimizer_extension"],
                            "planner": result_json["planner"],
                            "physical_planner": result_json["physical_planner"],
                            "recall": recall,
                        }
                    )

                if APPROACH == Approach.VSS:
                    estimated_index_memory_bytes = conn.execute(
                        f"SELECT approx_memory_usage FROM pragma_hnsw_index_info() WHERE index_name = '{TABLE_NAME}_vss_idx';"
                    ).fetchone()[0]
                elif APPROACH == Approach.PDXEARCH:
                    estimated_index_memory_bytes = DATASET_NUM_ROWS * DATASET_DIMS * 4
                else:
                    estimated_index_memory_bytes = 0

                iteration_result = {
                    "build_type": BUILD_TYPE,
                    "dataset": "arxiv_for_fanns_1.2M",
                    "dataset_order": ARXIV_DATASET_ORDER,
                    "thread_count": THREAD_COUNT,
                    "query_k": current_query_k,
                    "approach": APPROACH.name,
                    "filtered": True,
                    "min_selectivity": SELECTIVITY_RANGE[0],
                    "max_selectivity": SELECTIVITY_RANGE[1],
                    "num_queries_run": len(results),
                    "runtime_parameter_name": (
                        "n_probe"
                        if PDXEARCH_ENABLED
                        else "ef_search" if VSS_ENABLED else "none"
                    ),
                    "runtime_parameter": runtime_parameter,
                    "avg_all_optimizers": mean(r["all_optimizers"] for r in results),
                    "avg_optimizer_extension": mean(
                        r["optimizer_extension"] for r in results
                    ),
                    "avg_planner": mean(r["planner"] for r in results),
                    "avg_physical_planner": mean(
                        r["physical_planner"] for r in results
                    ),
                    "avg_index_scan_duration": mean(
                        r["index_scan_duration"] for r in results
                    ),
                    "avg_calc_index_scan_duration": mean(
                        r["calc_index_scan_duration"] for r in results
                    ),
                    "avg_filtered_sequential_scan_duration": mean(
                        r["filtered_sequential_scan_duration"] for r in results
                    ),
                    "avg_cpu_time": mean(r["cpu_time"] for r in results),
                    "avg_latency": mean(r["latency"] for r in results),
                    "avg_e2e_duration": mean(r["e2e_duration"] for r in results),
                    "avg_recall": mean(r["recall"] for r in results),
                    "avg_system_peak_buffer_memory": mean(
                        r["system_peak_buffer_memory"] for r in results
                    ),
                    "avg_selectivity": mean(r["selectivity"] for r in results),
                    "estimated_index_memory_bytes": estimated_index_memory_bytes,
                    "normalize": NORMALIZE,
                    "commit_hash": COMMIT_HASH,
                    "seed": SEED if SEED is not None else "None",
                    "debug": DEBUG,
                }

                all_results_df = pd.concat(
                    [all_results_df, pd.DataFrame([iteration_result])],
                    ignore_index=True,
                )

                print("\n--------- Summary: ---------")
                fields_to_print = [
                    "avg_all_optimizers",
                    "avg_optimizer_extension",
                    "avg_planner",
                    "avg_physical_planner",
                    "avg_index_scan_duration",
                    "avg_calc_index_scan_duration",
                    "avg_filtered_sequential_scan_duration",
                    "avg_cpu_time",
                    "avg_latency",
                    "avg_e2e_duration",
                    "avg_recall",
                    "avg_system_peak_buffer_memory",
                    "avg_selectivity",
                    "num_queries_run",
                ]
                print(
                    "\n".join(
                        (f" {k}: {v:.6f}" if isinstance(v, float) else f" {k}: {v}")
                        for k, v in iteration_result.items()
                        if k in fields_to_print
                    )
                )
                print("----------------------------\n")

                # Save per-query results
                os.makedirs("run_results", exist_ok=True)
                output_file = (
                    f"run_results/results_arxiv_filtered_sel{SELECTIVITY_RANGE[0]}-{SELECTIVITY_RANGE[1]}"
                    f"_t{THREAD_COUNT}_k{current_query_k}_{APPROACH.name}_rp{runtime_parameter}.csv"
                )
                pd.DataFrame(results).to_csv(output_file, index=False)

        # Save summary
        os.makedirs("benchmark_summaries", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_output_file = f"benchmark_summaries/benchmark_summary_arxiv_{APPROACH.name}_{timestamp}.csv"
        all_results_df.to_csv(final_output_file, index=False)
        print(f"Results exported to: {final_output_file}")


if __name__ == "__main__":
    main()
