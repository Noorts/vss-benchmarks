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

if __name__ == "__main__":

    case_types = [
        "Performance1024D1200K",
        "Performance1024D769K",
        "Performance1536D500K",
        "Performance768D1M",
        "Performance128D4999K",
    ]

    duckdb_threads = [14]

    K = 10
    MAX_SEARCH_QUERIES = 1000

    plain_config = DuckDBConfig(
        case_type=case_types,
        duckdb_threads=duckdb_threads,
        max_search_queries=MAX_SEARCH_QUERIES,
    )

    # n_probe = [1, 2, 4, 8, 16, 28, 32, 64, 128, 256, 0]
    n_probe = list(range(0, 305, 16)) + [0]

    pdxearch_config = DuckDBPDXearchConfig(
        case_type=case_types,
        duckdb_threads=duckdb_threads,
        seed=0,
        k=K,
        max_search_queries=MAX_SEARCH_QUERIES,
        runtime_n_probe=n_probe,
        quantization_type=["f32", "u8"],
    )

    # ef_search = [64, 8, 16, 32, 96, 128, 192, 256]
    ef_search = list(range(0, 305, 16))

    vss_config = DuckDBVSSConfig(
        case_type=case_types,
        duckdb_threads=duckdb_threads,
        k=K,
        max_search_queries=MAX_SEARCH_QUERIES,
        runtime_ef_search=ef_search,
    )

    flags = [
        "--skip-search-concurrent",
    ]

    configurations = (
        plain_config.expand() + pdxearch_config.expand() + vss_config.expand()
    )
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

        db_key = (cfg.cli_name, cfg.case_type)

        if previous_iteration_db == db_key:
            command.extend(["--skip-drop-old", "--skip-load"])

        previous_iteration_db = db_key

        result = run_vectordbbench(command, "index_search")
        if result.returncode != 0:
            print(result.stderr, file=sys.stderr)
