"""Shared constants and helpers for analysis notebooks."""

import re

import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Case ID → human-readable dataset name
# ---------------------------------------------------------------------------
CASE_NAMES: dict[int, str] = {
    5:   "Cohere (1M × 768",
    10:  "OpenAI (500K × 1536",
    50:  "OpenAI (50K × 1536",
    500: "OpenAI (999K × 1536",
    501: "Agnews (769K × 1024",
}

# ---------------------------------------------------------------------------
# Dataset sorting
# ---------------------------------------------------------------------------
def get_dataset_sort_key(dataset_name: str) -> tuple:
    """Sort datasets by dimensions first, then by number of embeddings."""
    match = re.search(r"\((\d+(?:\.\d+)?)\s*([KM]?)\s*×\s*(\d+)", dataset_name)
    if not match:
        return (float('inf'), float('inf'))

    num_str, unit, dim_str = match.groups()
    dim = int(dim_str)

    num = float(num_str)
    if unit == "K":
        num_embeddings = int(num * 1000)
    elif unit == "M":
        num_embeddings = int(num * 1_000_000)
    else:
        num_embeddings = int(num)

    return (dim, num_embeddings)

# ---------------------------------------------------------------------------
# Index ordering, colors, hatches, and markers
# ---------------------------------------------------------------------------
INDEX_ORDER = [
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
    "VSS HNSW":                          _cmap(14),
    "PDXearch SKM (IVF; Global; F32)":   _cmap(8),
    "PDXearch SKM (IVF; Row Group; F32)": _cmap(0),
    "PDXearch SKM (IVF; Row Group; U8)": _cmap(1),
    "pgvector HNSW":                     _cmap(2),
    "pgvector IVFFlat":                  _cmap(3),
}

index_hatches = {
    "VSS HNSW":                          "//",
    "PDXearch SKM (IVF; Global; F32)":   "\\\\",
    "PDXearch SKM (IVF; Row Group; F32)": "/\\/\\",
    "PDXearch SKM (IVF; Row Group; U8)": "xx",
    "pgvector HNSW":                     "OO",
    "pgvector IVFFlat":                  "..",
}

index_markers = {
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
    if index_name == "HNSW":
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
