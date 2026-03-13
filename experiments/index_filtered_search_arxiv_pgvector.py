import json
import sys
from dataclasses import asdict

from tqdm import tqdm

from utils import (
    PgVectorHNSWConfig,
    PgVectorIVFFlatConfig,
    run_vectordbbench,
)

if __name__ == "__main__":
    case_types = ["ArxivFilterPerformanceCase"]
    arxiv_filter_types = ["EM", "R", "EMIS"]

    K = 10
    MAX_SEARCH_QUERIES = 1000
    FORCE_LOAD_INDEX = True

    max_parallel_workers = 14
    max_parallel_maintenance_workers = 13  # plus leader
    maintenance_work_mem = "16GB"

    ef_search = [4, 8, 16, 32, 64, 128, 256, 512, 1024]

    # PgVector HNSW configuration
    hnsw_config = PgVectorHNSWConfig(
        case_type=case_types,
        arxiv_filter_type=arxiv_filter_types,
        max_parallel_workers=max_parallel_workers,
        max_parallel_maintenance_workers=max_parallel_maintenance_workers,
        maintenance_work_mem=maintenance_work_mem,
        k=K,
        max_search_queries=MAX_SEARCH_QUERIES,
        force_load_index=FORCE_LOAD_INDEX,
        m=16,
        ef_construction=128,
        ef_search=ef_search,
    )

    n_probe = [1, 2, 3, 4, 5, 8, 16, 32, 64, 128]

    # PgVector IVFFlat configuration
    # `lists` is resolved automatically from PGVECTOR_IVFFLAT_LISTS in utils.py
    ivfflat_config = PgVectorIVFFlatConfig(
        case_type=case_types,
        arxiv_filter_type=arxiv_filter_types,
        max_parallel_workers=max_parallel_workers,
        max_parallel_maintenance_workers=max_parallel_maintenance_workers,
        maintenance_work_mem=maintenance_work_mem,
        k=K,
        max_search_queries=MAX_SEARCH_QUERIES,
        force_load_index=FORCE_LOAD_INDEX,
        probes=n_probe,
    )

    flags = [
        "--skip-search-concurrent",
    ]

    configurations = hnsw_config.expand() + ivfflat_config.expand()
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

        db_key = (cfg.cli_name, cfg.case_type, cfg.arxiv_filter_type)

        if previous_iteration_db == db_key:
            command.extend(["--skip-drop-old", "--skip-load"])

        previous_iteration_db = db_key

        result = run_vectordbbench(command, "index_filtered_search_arxiv")
        if result.returncode != 0:
            print(result.stderr, file=sys.stderr)
