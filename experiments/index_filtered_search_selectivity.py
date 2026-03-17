import json
import sys
from dataclasses import asdict

from tqdm import tqdm

from utils import (
    DuckDBConfig,
    DuckDBPDXearchConfig,
    DuckDBVSSConfig,
    run_vectordbbench,
)


def config_sort_key(cfg):
    runtime_param = getattr(cfg, "runtime_n_probe", None)
    if runtime_param is None:
        runtime_param = getattr(cfg, "runtime_ef_search", None)
    if runtime_param is None:
        runtime_param = 0
    return (
        cfg.cli_name or "",
        cfg.case_type or "",
        cfg.dataset_with_size_type or "",
        cfg.duckdb_threads or 0,
        runtime_param,
        cfg.filter_rate or 0,
    )


if __name__ == "__main__":

    datasets_with_size_types = [
        "Medium OpenAI (1536dim, 500K)",
        "Medium Cohere (768dim, 1M)",
    ]

    filter_rates = [0.01, 0.1, 0.5, 0.9, 0.99]

    duckdb_threads = [14]

    K = 10
    MAX_SEARCH_QUERIES = 1000
    FORCE_LOAD_INDEX = True
    USE_BLOB_INTERFACE = True

    plain_config = DuckDBConfig(
        case_type="NewIntFilterPerformanceCase",
        dataset_with_size_type=datasets_with_size_types,
        filter_rate=filter_rates,
        duckdb_threads=duckdb_threads,
        k=K,
        max_search_queries=MAX_SEARCH_QUERIES,
        force_load_index=FORCE_LOAD_INDEX,
        use_blob_interface=USE_BLOB_INTERFACE,
    )

    n_probe = [1, 2, 3, 4, 5, 8, 16, 32, 64, 128, 256, 384, 512, 768, 0]

    pdxearch_config = DuckDBPDXearchConfig(
        case_type="NewIntFilterPerformanceCase",
        dataset_with_size_type=datasets_with_size_types,
        filter_rate=filter_rates,
        duckdb_threads=duckdb_threads,
        seed=0,
        k=K,
        max_search_queries=MAX_SEARCH_QUERIES,
        force_load_index=FORCE_LOAD_INDEX,
        use_blob_interface=USE_BLOB_INTERFACE,
        runtime_n_probe=n_probe,
        quantization_type=["f32", "u8"],
    )

    ef_search = [4, 8, 16, 32, 64, 128, 256, 512, 1024]

    vss_config = DuckDBVSSConfig(
        case_type="NewIntFilterPerformanceCase",
        dataset_with_size_type=datasets_with_size_types,
        filter_rate=filter_rates,
        duckdb_threads=duckdb_threads,
        k=K,
        max_search_queries=MAX_SEARCH_QUERIES,
        force_load_index=FORCE_LOAD_INDEX,
        runtime_ef_search=ef_search,
    )
    # Note: VSS does not support the blob interface; leave use_blob_interface unset (None).

    flags = [
        "--skip-search-concurrent",
    ]

    configurations = (
        plain_config.expand() + pdxearch_config.expand() + vss_config.expand()
    )
    configurations.sort(key=config_sort_key)
    print(json.dumps([asdict(c) for c in configurations], indent=2))

    # Track the database of the previous iteration such that we can skip data loading and index creation if it's ready.
    previous_iteration_db = None

    for cfg in tqdm(configurations, desc="Running benchmarks"):
        command = [
            cfg.cli_name,
            "--task-label",
            cfg.task_label,
            "--case-type",
            cfg.case_type,
            *flags,
            *cfg.to_cli_args(),
        ]

        db_key = (cfg.cli_name, cfg.case_type, cfg.dataset_with_size_type)

        if previous_iteration_db == db_key:
            command.extend(["--skip-drop-old", "--skip-load"])

        previous_iteration_db = db_key

        result = run_vectordbbench(command, "index_filtered_search_selectivity")
        if result.returncode != 0:
            print(result.stderr, file=sys.stderr)
