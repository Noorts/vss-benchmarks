"""Scatter plots showing update_date ordering in original vs sorted ArxivForFANNs dataset,
plus a histogram of the update_date distribution from the full attributes JSONL.

Vertical dashed lines mark DuckDB row group boundaries (every 122,880 rows).
"""

import json as json_mod
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from common import (
    apply_style,
    PLOT_DPI,
    LABEL_FONTSIZE,
    TICK_FONTSIZE,
    TITLE_FONTSIZE,
    FONT_COLOR,
    TICK_FONTS_COLOR,
)

DATASET_BASE = "/Users/user/vectordb_bench/dataset"
ORIG_DIR = f"{DATASET_BASE}/c_arxivforfanns/c_arxivforfanns_medium_1m"
SORTED_DIR = f"{DATASET_BASE}/c_arxivforfannssortedbyupdatedate/c_arxivforfannssortedbyupdatedate_medium_1m"
RANDOM_DIR = f"{DATASET_BASE}/c_arxivforfannsrandom/c_arxivforfannsrandom_medium_1m"

ATTRIBUTES_JSONL = "/Users/user/Code/vss-benchmarks/utils/arxiv-for-fanns-large-database_attributes.jsonl"

ROW_GROUP_SIZE = 122_880


def load_update_dates(directory: str) -> np.ndarray:
    """Load update_date values from r_labels.parquet (positional order matches train rows)."""
    df = pl.read_parquet(f"{directory}/r_labels.parquet")
    return df["update_date"].to_numpy()


def load_attributes_update_dates(path: str) -> np.ndarray:
    """Load update_date values from the full attributes JSONL file."""
    dates = []
    with open(path) as f:
        for line in f:
            obj = json_mod.loads(line)
            dates.append(obj["update_date"])
    return np.array(dates)


def style_ax(ax):
    ax.grid(True, linestyle="-", linewidth=0.4, color="gray", alpha=0.2)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="both", colors=TICK_FONTS_COLOR, length=0)
    ax.tick_params(axis="x", labelsize=TICK_FONTSIZE)
    ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)


def plot_dates(
    ax, dates: np.ndarray, title: str, point_size: float = 0.05, is_bottom: bool = False
):
    n = len(dates)

    # Row group boundaries
    boundaries = list(range(ROW_GROUP_SIZE, n, ROW_GROUP_SIZE))
    for b in boundaries:
        ax.axvline(x=b, color="red", linewidth=0.4, alpha=0.35, linestyle="--")

    # Scatter (use stride to keep rendering fast while preserving shape)
    stride = max(1, n // 200_000)
    xs = np.arange(0, n, stride)
    ys = dates[::stride]
    ax.scatter(xs, ys, s=point_size, alpha=0.4, color="#1f77b4", rasterized=True)

    ax.set_title(title, fontsize=TITLE_FONTSIZE, color=FONT_COLOR)
    if is_bottom:
        ax.set_xlabel(
            "Row index", fontsize=LABEL_FONTSIZE, color=FONT_COLOR, labelpad=10
        )
    ax.set_ylabel("update_date", fontsize=LABEL_FONTSIZE, color=FONT_COLOR, labelpad=10)

    ax.set_xlim(0, n)
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(
            lambda x, _: (
                f"{x / 1e6:.1f}M"
                if x >= 1e6
                else f"{x / 1e3:.0f}K" if x >= 1e3 else f"{int(x)}"
            )
        )
    )

    style_ax(ax)


def plot_histogram(ax, dates: np.ndarray, title: str):
    ax.hist(dates, bins=100, color="#1f77b4", edgecolor="white", linewidth=0.5)

    ax.set_title(title, fontsize=TITLE_FONTSIZE, color=FONT_COLOR)
    ax.set_xlabel("update_date", fontsize=LABEL_FONTSIZE, color=FONT_COLOR, labelpad=10)
    ax.set_ylabel("Count", fontsize=LABEL_FONTSIZE, color=FONT_COLOR, labelpad=10)

    style_ax(ax)


if __name__ == "__main__":
    apply_style()

    print("Loading original dataset update_dates...")
    dates_orig = load_update_dates(ORIG_DIR)
    print("Loading sorted dataset update_dates...")
    dates_sorted = load_update_dates(SORTED_DIR)
    print("Loading randomly shuffled dataset update_dates...")
    dates_random = load_update_dates(RANDOM_DIR)
    print("Loading full attributes JSONL update_dates...")
    dates_full = load_attributes_update_dates(ATTRIBUTES_JSONL)
    print(f"Full attributes: {len(dates_full)} rows")

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(
        4, 1, figsize=(7, 8), gridspec_kw={"height_ratios": [1, 1, 1, 1]}
    )

    # Top: full 2.7M attributes scatter
    plot_dates(ax1, dates_full, "arxiv-for-fanns-large (2.7M × 4096)")

    # Second: original 1.2M scatter
    plot_dates(ax2, dates_orig, "Sampled and PCA'd (1.2M × 1024)")

    # Third: sorted scatter
    plot_dates(ax3, dates_sorted, "Rows sorted by update_date (1.2M × 1024)")

    # Bottom: randomly shuffled scatter
    plot_dates(ax4, dates_random, "Rows shuffled (1.2M × 1024)", is_bottom=True)

    # Share y-axis across all four scatter plots
    all_dates = [dates_full, dates_orig, dates_sorted, dates_random]
    y_lo = min(d.min() for d in all_dates)
    y_hi = max(d.max() for d in all_dates)
    # Round to nearest 2500 boundary with padding so tick marks aren't clipped
    tick_step = 2500
    ylim = (
        (y_lo // tick_step) * tick_step - tick_step * 0.1,
        (y_hi // tick_step) * tick_step + tick_step * 0.1,
    )
    for ax in (ax1, ax2, ax3, ax4):
        ax.set_ylim(ylim)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(2500))

    fig.tight_layout()
    fig.savefig("arxiv_update_date_order.pdf", dpi=PLOT_DPI, bbox_inches="tight")
    print(
        f"Saved arxiv_update_date_order.pdf  ({len(dates_orig)} train rows, {len(dates_full)} attribute rows)"
    )
