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

    num_repeats = 3

    case_types = [
        "Performance1024D1200K",
        "Performance1024D769K",
        "Performance1536D500K",
        "Performance768D1M",
        "Performance128D4999K",
    ]

    duckdb_threads = [1, 14]

    pdxearch_config = DuckDBPDXearchConfig(
        case_type=case_types,
        duckdb_threads=duckdb_threads,
        seed=0,
        quantization_type=["f32", "u8"],
    )

    vss_config = DuckDBVSSConfig(
        case_type=case_types,
        duckdb_threads=duckdb_threads,
    )

    flags = [
        "--skip-search-serial",
        "--skip-search-concurrent",
    ]

    configurations = pdxearch_config.expand() + vss_config.expand()
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
