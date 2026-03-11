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

    num_repeats = 3

    case_types = [
        "Performance1024D1200K",
        "Performance1024D769K",
        "Performance1536D500K",
        "Performance768D1M",
        "Performance128D4999K",
    ]

    max_parallel_workers = 14
    max_parallel_maintenance_workers = 13  # plus leader
    maintenance_work_mem = "16GB"

    # PgVector HNSW configuration
    hnsw_config = PgVectorHNSWConfig(
        case_type=case_types,
        max_parallel_workers=max_parallel_workers,
        max_parallel_maintenance_workers=max_parallel_maintenance_workers,
        maintenance_work_mem=maintenance_work_mem,
        # Same index parameters as VSS' HNSW.
        m=16,
        ef_construction=128,
        ef_search=64,
    )

    # PgVector IVFFlat configuration
    # `lists` is resolved automatically from PGVECTOR_IVFFLAT_LISTS in utils.py
    ivfflat_config = PgVectorIVFFlatConfig(
        case_type=case_types,
        max_parallel_workers=max_parallel_workers,
        max_parallel_maintenance_workers=max_parallel_maintenance_workers,
        maintenance_work_mem=maintenance_work_mem,
        # probes=32,  # Typical starting point: sqrt(lists)
    )

    flags = [
        "--skip-search-serial",
        "--skip-search-concurrent",
    ]

    configurations = hnsw_config.expand() + ivfflat_config.expand()
    print(json.dumps([asdict(c) for c in configurations], indent=2))

    configurations = configurations * num_repeats

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
        result = run_vectordbbench(command, "index_creation")
        if result.returncode != 0:
            print(result.stderr, file=sys.stderr)
