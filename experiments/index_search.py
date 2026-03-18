import json
import sys
from dataclasses import asdict

from tqdm import tqdm

from utils import (
    DuckDBConfig,
    DuckDBPDXearchConfig,
    DuckDBVSSConfig,
    get_vss_extension_path,
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

    duckdb_threads = [14, 1]
    duckdb_threads_during_index_creation = [14]

    K = 10
    MAX_SEARCH_QUERIES = 1000
    FORCE_LOAD_INDEX = True
    USE_BLOB_INTERFACE = True
    USE_CUSTOM_VSS_BUILD = True  # False = install and load the official VSS extension

    plain_config = DuckDBConfig(
        case_type=case_types,
        duckdb_threads=duckdb_threads,
        max_search_queries=MAX_SEARCH_QUERIES,
        force_load_index=FORCE_LOAD_INDEX,
        use_blob_interface=USE_BLOB_INTERFACE,
    )

    n_probe = [1, 2, 3, 4, 5, 8, 16, 32, 64, 128, 256, 384, 512, 768, 0]

    pdxearch_config = DuckDBPDXearchConfig(
        case_type=case_types,
        duckdb_threads=duckdb_threads,
        duckdb_threads_during_index_creation=duckdb_threads_during_index_creation,
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
        case_type=case_types,
        duckdb_threads=duckdb_threads,
        duckdb_threads_during_index_creation=duckdb_threads_during_index_creation,
        k=K,
        max_search_queries=MAX_SEARCH_QUERIES,
        force_load_index=FORCE_LOAD_INDEX,
        use_blob_interface=USE_BLOB_INTERFACE,
        extension_path=get_vss_extension_path() if USE_CUSTOM_VSS_BUILD else None,
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

        results_subdir = "index_search_blob" if getattr(cfg, "use_blob_interface", None) else "index_search"
        result = run_vectordbbench(command, results_subdir)
        if result.returncode != 0:
            print(result.stderr, file=sys.stderr)
