"""Shared constants and helpers for analysis notebooks."""

import re

import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Styling defaults
# ---------------------------------------------------------------------------
PLOT_DPI = 300

LABEL_FONTSIZE = 14
TICK_FONTSIZE = 12
X_TICK_FONTSIZE = 12
BAR_LABEL_FONTSIZE = 10
TITLE_FONTSIZE = 15
MARKER_SIZE = 60

FONT_COLOR = "#333333"
TICK_FONTS_COLOR = "#585858"
BAR_TEXT_COLOR = "#191919"


def apply_style() -> None:
    """Apply shared matplotlib style settings."""
    mpl.rcParams["hatch.linewidth"] = 0.2
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]


_MIN_BOOTSTRAP_SAMPLES = 30


def median_ci(
    values: list[float], n_boot: int = 1_000, ci: float = 0.95, rng_seed: int = 42
) -> tuple[float, float, float]:
    """Return (median, lower_err, upper_err) using bootstrap percentile CI.

    lower_err and upper_err are *distances* from the median (suitable for
    matplotlib's ``yerr=[[lower_errs], [upper_errs]]``).
    """
    import numpy as np
    import warnings

    arr = np.asarray(values, dtype=float)
    med = float(np.median(arr))
    if len(arr) < 2:
        return med, 0.0, 0.0
    if len(arr) < _MIN_BOOTSTRAP_SAMPLES:
        warnings.warn(
            f"median_ci: bootstrapping {ci:.0%} CI with only {len(arr)} measurements "
            f"(recommended >= {_MIN_BOOTSTRAP_SAMPLES})",
            stacklevel=2,
        )
    rng = np.random.default_rng(rng_seed)
    boot_medians = np.median(
        rng.choice(arr, size=(n_boot, len(arr)), replace=True), axis=1
    )
    alpha = (1 - ci) / 2
    lo, hi = float(np.quantile(boot_medians, alpha)), float(
        np.quantile(boot_medians, 1 - alpha)
    )
    return med, med - lo, hi - med


def qps_median_ci(
    latencies: list[float], n_boot: int = 1_000, ci: float = 0.95, rng_seed: int = 42
) -> tuple[float, float, float]:
    """Return (median_qps, lower_err, upper_err) by bootstrapping QPS = 1/median(latency).

    Resamples the per-query latency list, computes median latency per
    resample, converts to QPS, then extracts the percentile CI.
    lower_err and upper_err are distances from median_qps.
    """
    import numpy as np
    import warnings

    arr = np.asarray(latencies, dtype=float)
    med_lat = float(np.median(arr))
    med_qps = 1.0 / med_lat if med_lat > 0 else 0.0
    if len(arr) < 2:
        return med_qps, 0.0, 0.0
    if len(arr) < _MIN_BOOTSTRAP_SAMPLES:
        warnings.warn(
            f"qps_median_ci: bootstrapping {ci:.0%} CI with only {len(arr)} measurements "
            f"(recommended >= {_MIN_BOOTSTRAP_SAMPLES})",
            stacklevel=2,
        )
    rng = np.random.default_rng(rng_seed)
    boot_med_lats = np.median(
        rng.choice(arr, size=(n_boot, len(arr)), replace=True), axis=1
    )
    boot_qps = 1.0 / boot_med_lats
    alpha = (1 - ci) / 2
    lo, hi = float(np.quantile(boot_qps, alpha)), float(
        np.quantile(boot_qps, 1 - alpha)
    )
    return med_qps, med_qps - lo, hi - med_qps


def save_fig(name: str, **kwargs) -> None:
    """Save the current figure as both PDF and PNG.

    Args:
        name: Base filename without extension (e.g. "index_creation").
        **kwargs: Extra keyword arguments forwarded to plt.savefig
                  (e.g. dpi, bbox_inches).
    """
    kwargs.setdefault("dpi", PLOT_DPI)
    kwargs.setdefault("bbox_inches", "tight")
    plt.savefig(f"{name}.pdf", **kwargs)
    plt.savefig(f"{name}.png", **kwargs)


# ---------------------------------------------------------------------------
# Case ID → human-readable dataset name
# ---------------------------------------------------------------------------
CASE_NAMES: dict[int, str] = {
    # Capacity tests
    1: "SIFT (500K × 128)",
    2: "GIST (100K × 960)",
    # Cohere (768-dim)
    3: "LAION (100M × 768)",
    4: "Cohere (10M × 768)",
    5: "Cohere (1M × 768)",
    6: "Cohere (10M × 768)",  # 1% filter
    7: "Cohere (1M × 768)",  # 1% filter
    8: "Cohere (10M × 768)",  # 99% filter
    9: "Cohere (1M × 768)",  # 99% filter
    # OpenAI (1536-dim)
    10: "OpenAI (500K × 1536)",
    11: "OpenAI (5M × 1536)",
    12: "OpenAI (500K × 1536)",  # 1% filter
    13: "OpenAI (5M × 1536)",  # 1% filter
    14: "OpenAI (500K × 1536)",  # 99% filter
    15: "OpenAI (5M × 1536)",  # 99% filter
    # Bioasq (1024-dim)
    17: "Bioasq (1M × 1024)",
    20: "Bioasq (10M × 1024)",
    # OpenAI small
    50: "OpenAI (50K × 1536)",
    # Custom Noorts datasets
    500: "OpenAI (999K × 1536)",
    501: "Agnews (769K × 1024)",
    502: "ArxivForFanns (1.2M × 1024)",
    503: "SIFT (4999K × 128)",
}

# ---------------------------------------------------------------------------
# Dataset properties (exact embedding counts and dimensions)
# ---------------------------------------------------------------------------
DATASET_PROPS: dict[str, dict[str, int]] = {
    "SIFT (500K × 128)": {"n": 500_000, "d": 128},
    "GIST (100K × 960)": {"n": 100_000, "d": 960},
    "LAION (100M × 768)": {"n": 100_000_000, "d": 768},
    "Cohere (1M × 768)": {"n": 1_000_000, "d": 768},
    "Cohere (10M × 768)": {"n": 10_000_000, "d": 768},
    "OpenAI (50K × 1536)": {"n": 50_000, "d": 1536},
    "OpenAI (500K × 1536)": {"n": 500_000, "d": 1536},
    "OpenAI (5M × 1536)": {"n": 5_000_000, "d": 1536},
    "OpenAI (999K × 1536)": {"n": 999_000, "d": 1536},
    "Bioasq (1M × 1024)": {"n": 1_000_000, "d": 1024},
    "Bioasq (10M × 1024)": {"n": 10_000_000, "d": 1024},
    "Agnews (769K × 1024)": {"n": 769_382, "d": 1024},
    "ArxivForFanns (1.2M × 1024)": {"n": 1_200_000, "d": 1024},
    "SIFT (4999K × 128)": {"n": 4_999_000, "d": 128},
}


# ---------------------------------------------------------------------------
# Dataset sorting
# ---------------------------------------------------------------------------
def get_dataset_sort_key(dataset_name: str) -> tuple:
    """Sort datasets by size in bytes first, then dims, then num_embeddings."""
    match = re.search(r"\((\d+(?:\.\d+)?)\s*([KM]?)\s*×\s*(\d+)", dataset_name)
    if not match:
        return (float("inf"), float("inf"), float("inf"))

    num_str, unit, dim_str = match.groups()
    dim = int(dim_str)

    num = float(num_str)
    if unit == "K":
        num_embeddings = int(num * 1000)
    elif unit == "M":
        num_embeddings = int(num * 1_000_000)
    else:
        num_embeddings = int(num)

    size_bytes = num_embeddings * dim * 4

    return (size_bytes, dim, num_embeddings)


# ---------------------------------------------------------------------------
# Index ordering, colors, hatches, and markers
# ---------------------------------------------------------------------------
INDEX_ORDER = [
    "DuckDB",
    "DuckDB VSS (HNSW)",
    "pgvector (HNSW)",
    "pgvector (IVFFlat)",
    "DuckDB PDXearch (IVF, Global, F32)",
    "DuckDB PDXearch (IVF, Global, U8)",
    "DuckDB PDXearch (IVF, Row Group, F32)",
    "DuckDB PDXearch (IVF, Row Group, U8)",
]

try:
    _cmap = plt.colormaps["tab20"]
except (AttributeError, KeyError):
    _cmap = plt.cm.get_cmap("tab20")

index_colors = {
    "DuckDB": _cmap(14),
    "DuckDB VSS (HNSW)": _cmap(15),
    "DuckDB PDXearch (IVF, Global, F32)": _cmap(8),
    "DuckDB PDXearch (IVF, Global, U8)": _cmap(9),
    "DuckDB PDXearch (IVF, Row Group, F32)": _cmap(0),
    "DuckDB PDXearch (IVF, Row Group, U8)": _cmap(1),
    "pgvector (HNSW)": _cmap(2),
    "pgvector (IVFFlat)": _cmap(3),
}

index_hatches = {
    "DuckDB": "++",
    "DuckDB VSS (HNSW)": "//",
    "DuckDB PDXearch (IVF, Global, F32)": "\\\\",
    "DuckDB PDXearch (IVF, Global, U8)": "\\\\",
    "DuckDB PDXearch (IVF, Row Group, F32)": "/\\/\\",
    "DuckDB PDXearch (IVF, Row Group, U8)": "xx",
    "pgvector (HNSW)": "OO",
    "pgvector (IVFFlat)": "..",
}

index_markers = {
    "DuckDB": "X",
    "DuckDB VSS (HNSW)": "s",
    "DuckDB PDXearch (IVF, Global, F32)": "D",
    "DuckDB PDXearch (IVF, Global, U8)": "p",
    "DuckDB PDXearch (IVF, Row Group, F32)": "o",
    "DuckDB PDXearch (IVF, Row Group, U8)": "P",
    "pgvector (HNSW)": "^",
    "pgvector (IVFFlat)": "v",
}


# ---------------------------------------------------------------------------
# Index name transformation (from raw JSON to display name)
# ---------------------------------------------------------------------------
def transform_duckdb_index_name(db_case_cfg: dict, global_version) -> str:
    """Transform a DuckDB index name from raw config to display name."""
    index_name = db_case_cfg["index"]
    if index_name == "FLAT":
        return "DuckDB"
    elif index_name == "HNSW":
        return "DuckDB VSS (HNSW)"
    elif index_name == "PDXEARCH" and global_version is not None:
        quant = db_case_cfg.get("index_quantization_type", "f32").upper()
        return f"DuckDB PDXearch (IVF, Global, {quant})"
    elif index_name == "PDXEARCH":
        quant = db_case_cfg.get("index_quantization_type", "f32").upper()
        return f"DuckDB PDXearch (IVF, Row Group, {quant})"
    return index_name


def transform_pgvector_index_name(db_case_cfg: dict) -> str:
    """Transform a pgvector index name from raw config to display name."""
    index_name = db_case_cfg["index"]
    if index_name.lower() == "hnsw":
        return "pgvector (HNSW)"
    elif index_name.lower() == "ivfflat":
        return "pgvector (IVFFlat)"
    return index_name


# ---------------------------------------------------------------------------
# Dataset label formatting
# ---------------------------------------------------------------------------
def format_dataset_label(dataset_name: str) -> str:
    """Format dataset name with details: name, n, d, and size in GB."""
    match = re.match(r"(.+?)\s*\((\d+(?:\.\d+)?)\s*([KM]?)\s*×\s*(\d+)", dataset_name)
    if not match:
        return dataset_name.split("(")[0].strip()

    name, num_str, unit, dim_str = match.groups()
    name = name.strip()
    dim = int(dim_str)

    n_label = f"{num_str}{unit}" if unit else f"{num_str}"

    # Compute raw F32 data size in GB
    num = float(num_str)
    if unit == "K":
        num_embeddings = int(num * 1000)
    elif unit == "M":
        num_embeddings = int(num * 1_000_000)
    else:
        num_embeddings = int(num)

    size_gb = (num_embeddings * dim * 4) / (1024**3)

    return f"{name}\nn={n_label}, d={dim}\n{size_gb:.2f} GiB"


def format_dataset_title(dataset_name: str) -> str:
    """Format dataset name for subplot title: name (n=X, d=Y)."""
    match = re.match(r"(.+?)\s*\((\d+(?:\.\d+)?)\s*([KM]?)\s*×\s*(\d+)", dataset_name)
    if not match:
        return dataset_name.split("(")[0].strip()

    name, num_str, unit, dim_str = match.groups()
    name = name.strip()
    dim = int(dim_str)

    n_label = f"{num_str}{unit}" if unit else f"{num_str}"

    # Compute raw F32 data size in GB
    num = float(num_str)
    if unit == "K":
        num_embeddings = int(num * 1000)
    elif unit == "M":
        num_embeddings = int(num * 1_000_000)
    else:
        num_embeddings = int(num)

    size_gb = (num_embeddings * dim * 4) / (1024**3)

    return f"{name}\n(n={n_label}, d={dim}, {size_gb:.2f} GiB)"
