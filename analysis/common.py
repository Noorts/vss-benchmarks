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
    mpl.rcParams['hatch.linewidth'] = 0.2
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']

# ---------------------------------------------------------------------------
# Case ID → human-readable dataset name
# ---------------------------------------------------------------------------
CASE_NAMES: dict[int, str] = {
    # Capacity tests
    1:   "SIFT (500K × 128)",
    2:   "GIST (100K × 960)",
    # Cohere (768-dim)
    3:   "LAION (100M × 768)",
    4:   "Cohere (10M × 768)",
    5:   "Cohere (1M × 768)",
    6:   "Cohere (10M × 768)",       # 1% filter
    7:   "Cohere (1M × 768)",        # 1% filter
    8:   "Cohere (10M × 768)",       # 99% filter
    9:   "Cohere (1M × 768)",        # 99% filter
    # OpenAI (1536-dim)
    10:  "OpenAI (500K × 1536)",
    11:  "OpenAI (5M × 1536)",
    12:  "OpenAI (500K × 1536)",     # 1% filter
    13:  "OpenAI (5M × 1536)",       # 1% filter
    14:  "OpenAI (500K × 1536)",     # 99% filter
    15:  "OpenAI (5M × 1536)",       # 99% filter
    # Bioasq (1024-dim)
    17:  "Bioasq (1M × 1024)",
    20:  "Bioasq (10M × 1024)",
    # OpenAI small
    50:  "OpenAI (50K × 1536)",
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
    "SIFT (500K × 128)":              {"n": 500_000,       "d": 128},
    "GIST (100K × 960)":              {"n": 100_000,       "d": 960},
    "LAION (100M × 768)":             {"n": 100_000_000,   "d": 768},
    "Cohere (1M × 768)":              {"n": 1_000_000,     "d": 768},
    "Cohere (10M × 768)":             {"n": 10_000_000,    "d": 768},
    "OpenAI (50K × 1536)":            {"n": 50_000,        "d": 1536},
    "OpenAI (500K × 1536)":           {"n": 500_000,       "d": 1536},
    "OpenAI (5M × 1536)":             {"n": 5_000_000,     "d": 1536},
    "OpenAI (999K × 1536)":           {"n": 999_000,       "d": 1536},
    "Bioasq (1M × 1024)":             {"n": 1_000_000,     "d": 1024},
    "Bioasq (10M × 1024)":            {"n": 10_000_000,    "d": 1024},
    "Agnews (769K × 1024)":           {"n": 769_382,       "d": 1024},
    "ArxivForFanns (1.2M × 1024)":    {"n": 1_200_000,     "d": 1024},
    "SIFT (4999K × 128)":             {"n": 4_999_000,     "d": 128},
}

# ---------------------------------------------------------------------------
# Dataset sorting
# ---------------------------------------------------------------------------
def get_dataset_sort_key(dataset_name: str) -> tuple:
    """Sort datasets by size in bytes first, then dims, then num_embeddings."""
    match = re.search(r"\((\d+(?:\.\d+)?)\s*([KM]?)\s*×\s*(\d+)", dataset_name)
    if not match:
        return (float('inf'), float('inf'), float('inf'))

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
    "VSS HNSW",
    "pgvector HNSW",
    "pgvector IVFFlat",
    "PDXearch SKM (IVF; Global; F32)",
    "PDXearch SKM (IVF; Row Group; F32)",
    "PDXearch SKM (IVF; Row Group; U8)",
]

try:
    _cmap = plt.colormaps['tab20']
except (AttributeError, KeyError):
    _cmap = plt.cm.get_cmap('tab20')

index_colors = {
    "DuckDB":                            _cmap(16),
    "VSS HNSW":                          _cmap(14),
    "PDXearch SKM (IVF; Global; F32)":   _cmap(8),
    "PDXearch SKM (IVF; Row Group; F32)": _cmap(0),
    "PDXearch SKM (IVF; Row Group; U8)": _cmap(1),
    "pgvector HNSW":                     _cmap(2),
    "pgvector IVFFlat":                  _cmap(3),
}

index_hatches = {
    "DuckDB":                            "++",
    "VSS HNSW":                          "//",
    "PDXearch SKM (IVF; Global; F32)":   "\\\\",
    "PDXearch SKM (IVF; Row Group; F32)": "/\\/\\",
    "PDXearch SKM (IVF; Row Group; U8)": "xx",
    "pgvector HNSW":                     "OO",
    "pgvector IVFFlat":                  "..",
}

index_markers = {
    "DuckDB":                            "X",
    "VSS HNSW":                          "s",
    "PDXearch SKM (IVF; Global; F32)":   "D",
    "PDXearch SKM (IVF; Row Group; F32)": "o",
    "PDXearch SKM (IVF; Row Group; U8)": "P",
    "pgvector HNSW":                     "^",
    "pgvector IVFFlat":                  "v",
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
        return "VSS HNSW"
    elif index_name == "PDXEARCH" and global_version is not None:
        quant = db_case_cfg.get("index_quantization_type", "f32").upper()
        return f"PDXearch SKM (IVF; Global; {quant})"
    elif index_name == "PDXEARCH":
        quant = db_case_cfg.get("index_quantization_type", "f32").upper()
        return f"PDXearch SKM (IVF; Row Group; {quant})"
    return index_name


def transform_pgvector_index_name(db_case_cfg: dict) -> str:
    """Transform a pgvector index name from raw config to display name."""
    index_name = db_case_cfg["index"]
    if index_name.lower() == "hnsw":
        return "pgvector HNSW"
    elif index_name.lower() == "ivfflat":
        return "pgvector IVFFlat"
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

    size_gb = (num_embeddings * dim * 4) / (1024 ** 3)

    return f"{name}\nn={n_label}, d={dim}\n{size_gb:.2f}GB"


def format_dataset_title(dataset_name: str) -> str:
    """Format dataset name for subplot title: name (n=X, d=Y)."""
    match = re.match(r"(.+?)\s*\((\d+(?:\.\d+)?)\s*([KM]?)\s*×\s*(\d+)", dataset_name)
    if not match:
        return dataset_name.split("(")[0].strip()
    name, num_str, unit, dim_str = match.groups()
    n_label = f"{num_str}{unit}" if unit else num_str
    return f"{name.strip()} (n={n_label}, d={dim_str})"
