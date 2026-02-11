import json
import sys
from dataclasses import asdict

from tqdm import tqdm

from utils import (
    DuckDBPDXearchConfig,
    DuckDBVSSConfig,
    run_vectordbbench,
)

if __name__ == "__main__":

    case_types = [
        "Performance1024D769K",
        "Performance1536D500K",
        "Performance768D1M",
        "Performance1536D999K",
    ]

    duckdb_threads = [1, 14]

    config = DuckDBPDXearchConfig(
        case_type=case_types,
        duckdb_threads=duckdb_threads,
        seed=0,
    )

    vss_config = DuckDBVSSConfig(
        case_type=case_types,
        duckdb_threads=duckdb_threads,
    )

    flags = [
        "--skip-search-serial",
        "--skip-search-concurrent",
    ]

    configurations = config.expand() + vss_config.expand()
    print(json.dumps([asdict(c) for c in configurations], indent=2))

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
