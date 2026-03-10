import json
import os
import pathlib
import subprocess
import sys
import time
from dataclasses import dataclass, field, fields
from itertools import product

import duckdb

REPO_ROOT = pathlib.Path(__file__).parent.parent


def get_duckdb_version() -> str:
    """Return the version string of the duckdb Python package in the environment."""
    return duckdb.__version__


def get_git_commit_id(repo_path: str | pathlib.Path) -> str:
    """Return the short commit hash of a git repository."""
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        cwd=str(repo_path),
    )
    return result.stdout.strip()


def _base_db_label() -> dict:
    """Build the base db_label dict with duckdb version and vectordbbench commit."""
    return {
        "duckdb": get_duckdb_version(),
        "vectordbbench": get_git_commit_id(REPO_ROOT / "vectordbbench"),
        "vss-benchmarks": get_git_commit_id(REPO_ROOT),
    }


def get_pdxearch_extension_path() -> str:
    """Return the full path to the PDXearch DuckDB extension.

    Raises:
        FileNotFoundError: If the extension file does not exist.
    """
    extension_path = (
        REPO_ROOT
        / "PDXearch"
        / "build"
        / "release"
        / "extension"
        / "pdxearch"
        / "pdxearch.duckdb_extension"
    )
    resolved_path = extension_path.resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(
            f"PDXearch extension not found at {resolved_path}. "
            "Please build the extension first by running the build process in the PDXearch submodule."
        )
    return str(resolved_path)


def exec_cmd(command, cwd=None, env=None):
    return subprocess.run(command, capture_output=True, text=True, cwd=cwd, env=env)


def run_vectordbbench(command: list[str], results_subdir: str | None = None):
    # Get the path to the vectordbbench submodule
    script_dir = pathlib.Path(__file__).parent
    repo_root = script_dir.parent
    vectordbbench_dir = repo_root / "vectordbbench"

    # Verify the submodule exists
    if not vectordbbench_dir.exists():
        raise FileNotFoundError(
            f"vectordbbench submodule not found at {vectordbbench_dir}. "
            "Run `git submodule update --init` to initialize the submodule."
        )

    # Build the command using Python's -m flag to run the module from the submodule
    full_command = [
        sys.executable,
        "-m",
        "vectordb_bench.cli.vectordbbench",
        *command,
    ]

    # Set up environment variables
    env = os.environ.copy()

    # Add the vectordbbench directory to PYTHONPATH so the module can be found
    pythonpath = env.get("PYTHONPATH", "")
    if pythonpath:
        env["PYTHONPATH"] = f"{vectordbbench_dir}:{pythonpath}"
    else:
        env["PYTHONPATH"] = str(vectordbbench_dir)

    # Set the results directory to a location in the main repository
    # This makes it easier to access results without navigating into the submodule
    if results_subdir:
        results_dir = repo_root / "experiments" / "results" / results_subdir
    else:
        results_dir = repo_root / "experiments" / "results"

    results_dir.mkdir(parents=True, exist_ok=True)
    env["RESULTS_LOCAL_DIR"] = str(results_dir)

    return exec_cmd(full_command, cwd=str(vectordbbench_dir), env=env)


@dataclass
class BaseVectorDBBenchConfig:
    """Base configuration class for VectorDBBench CLI commands.

    Provides common functionality for converting configurations to CLI arguments
    and expanding list-valued fields into individual configs.

    Subclasses must define a `cli_name` field.
    """

    # -- Common benchmark parameters -------------------------------------------
    case_type: str | None = None
    k: int | list[int] | None = None
    max_search_queries: int | None = None
    force_load_index: bool | None = None
    db_label: str = field(default="", init=False)

    # Fields excluded from CLI argument generation.
    _NON_CLI_FIELDS = {"cli_name", "case_type"}

    def __post_init__(self):
        self.db_label = json.dumps(self._build_db_label())

    def _build_db_label(self) -> dict:
        """Return the db_label dict. Subclasses must override this."""
        raise NotImplementedError("Subclasses must implement _build_db_label")

    @property
    def task_label(self) -> str:
        """Generate a descriptive task label including the subcommand, dataset, and timestamp."""
        parts = [self.cli_name]
        if self.case_type:
            parts.append(self.case_type)
        parts.append(time.strftime("%H-%M-%S"))
        return "_".join(parts)

    def to_cli_args(self) -> list[str]:
        """Convert this configuration to a CLI argument list.

        Property names are transformed to CLI names by replacing underscores
        with hyphens and prepending ``--`` (e.g. ``user_name`` becomes
        ``--user-name``).  Properties that are ``None`` are omitted.
        Boolean values are handled specially for flags like reranking.
        """
        args: list[str] = []
        for f in fields(self):
            if f.name in self._NON_CLI_FIELDS:
                continue
            value = getattr(self, f.name)
            if value is None:
                continue
            cli_name = f"--{f.name.replace('_', '-')}"
            # Handle boolean flags specially (for reranking: --reranking/--skip-reranking)
            if isinstance(value, bool):
                if f.name == "reranking":
                    if value:
                        args.append("--reranking")
                    else:
                        args.append("--skip-reranking")
                else:
                    # For other booleans, just use the flag name if True
                    if value:
                        args.append(cli_name)
            else:
                args.extend([cli_name, str(value)])
        return args

    def expand(self) -> list:
        """Expand list-valued fields into individual configs (cartesian product)."""
        list_fields: list[str] = []
        list_values: list[list] = []
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, list):
                list_fields.append(f.name)
                list_values.append(value)
        if not list_fields:
            return [self]
        configs = []
        for combo in product(*list_values):
            kwargs = {f.name: getattr(self, f.name) for f in fields(self) if f.init}
            for fname, val in zip(list_fields, combo):
                kwargs[fname] = val
            configs.append(self.__class__(**kwargs))
        return configs


@dataclass
class DuckDBConfig(BaseVectorDBBenchConfig):
    """Base configuration for DuckDB benchmarks via VectorDBBench CLI.

    Fields correspond to CLI options shared across DuckDB commands.
    Set any field to a *list* of values to sweep over that parameter;
    ``expand()`` returns the cartesian product of all list-valued fields.
    """

    # -- VectorDBBench CLI subcommand name -------------------------------------
    cli_name: str = "duckdb"

    # -- Threads ---------------------------------------------------------------
    duckdb_threads: int | list[int] | None = None
    duckdb_threads_during_index_creation: int | list[int] | None = None

    def _build_db_label(self) -> dict:
        """Return the db_label dict. Subclasses can extend this."""
        return _base_db_label()


@dataclass
class DuckDBPDXearchConfig(DuckDBConfig):
    """Configuration for DuckDB PDXearch benchmarks (``duckdbpdxearch`` command)."""

    cli_name: str = "duckdbpdxearch"

    # -- Connection / extension ------------------------------------------------
    extension_path: str = field(default_factory=get_pdxearch_extension_path)

    # -- Index creation parameters ---------------------------------------------
    quantization_type: str | list[str] | None = None
    n_probe: int | list[int] | None = None
    seed: int | list[int] | None = None

    # -- Runtime parameters ----------------------------------------------------
    runtime_n_probe: int | list[int] | None = None

    def _build_db_label(self) -> dict:
        label = super()._build_db_label()
        label["pdxearch"] = get_git_commit_id(REPO_ROOT / "PDXearch")
        # label["global_version"] = "true"
        return label


@dataclass
class DuckDBVSSConfig(DuckDBConfig):
    """Configuration for DuckDB VSS/HNSW benchmarks (``duckdbvss`` command)."""

    cli_name: str = "duckdbvss"

    # -- Index creation parameters ---------------------------------------------
    ef_construction: int | list[int] | None = None
    ef_search: int | list[int] | None = None
    m: int | list[int] | None = None
    m0: int | list[int] | None = None

    # -- Runtime parameters ----------------------------------------------------
    runtime_ef_search: int | list[int] | None = None


@dataclass
class PgVectorConfig(BaseVectorDBBenchConfig):
    """Base configuration for PgVector benchmarks via VectorDBBench CLI.

    Fields correspond to CLI options shared across PgVector commands.
    Set any field to a *list* of values to sweep over that parameter;
    ``expand()`` returns the cartesian product of all list-valued fields.
    """

    # -- VectorDBBench CLI subcommand name -------------------------------------
    cli_name: str = "pgvector"

    # -- Connection parameters -------------------------------------------------
    user_name: str = "postgres"
    password: str = field(
        default_factory=lambda: os.environ.get("POSTGRES_PASSWORD", "postgres")
    )
    host: str = "localhost"
    port: int = 5432
    db_name: str = "vectordb"

    # -- Index creation performance parameters ----------------------------------
    maintenance_work_mem: str | list[str] | None = None
    max_parallel_workers: int | list[int] | None = None

    # -- Quantization parameters -----------------------------------------------
    quantization_type: str | list[str] | None = None
    table_quantization_type: str | list[str] | None = None

    # -- Reranking parameters --------------------------------------------------
    reranking: bool | list[bool] | None = None
    reranking_metric: str | list[str] | None = None
    quantized_fetch_limit: int | list[int] | None = None

    def _build_db_label(self) -> dict:
        """Return the db_label dict. Subclasses can extend this."""
        label = {
            "vectordbbench": get_git_commit_id(REPO_ROOT / "vectordbbench"),
            "vss-benchmarks": get_git_commit_id(REPO_ROOT),
        }
        return label


@dataclass
class PgVectorHNSWConfig(PgVectorConfig):
    """Configuration for PgVector HNSW benchmarks (``pgvectorhnsw`` command)."""

    cli_name: str = "pgvectorhnsw"

    # -- HNSW index creation parameters -----------------------------------------
    m: int | list[int] | None = None
    ef_construction: int | list[int] | None = None
    ef_search: int | list[int] | None = None


@dataclass
class PgVectorIVFFlatConfig(PgVectorConfig):
    """Configuration for PgVector IVFFlat benchmarks (``pgvectorivfflat`` command)."""

    cli_name: str = "pgvectorivfflat"

    # -- IVFFlat index creation parameters -------------------------------------
    lists: int | list[int] | None = None
    probes: int | list[int] | None = None
