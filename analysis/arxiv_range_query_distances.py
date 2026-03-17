"""Histogram of test query range distances (range_end - range_start) and
true selectivities for the ArxivForFANNs dataset."""

import json
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
TEST_ATTRS_R = (
    f"{DATASET_BASE}/c_arxivforfanns/c_arxivforfanns_medium_1m/test_attrs_r.parquet"
)
SELECTIVITY_JSON = "arxiv_range_selectivity.json"


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


if __name__ == "__main__":
    apply_style()

    # Load range distances
    df = pl.read_parquet(TEST_ATTRS_R)
    distances = (df["range_end"] - df["range_start"]).to_numpy()

    print(f"Queries: {len(distances)}")
    print(f"Min distance: {distances.min()}, Max distance: {distances.max()}")
    print(
        f"Mean: {distances.mean():.1f}, Median: {np.median(distances):.1f}, Std: {distances.std():.1f}"
    )

    # Load true selectivities
    with open(SELECTIVITY_JSON) as f:
        sel_data = json.load(f)
    selectivities = np.array([q["selectivity"] for q in sel_data["queries"]])
    print(f"\nSelectivities: {len(selectivities)} queries")
    print(f"Min: {selectivities.min():.4f}, Max: {selectivities.max():.4f}")
    print(f"Mean: {selectivities.mean():.4f}, Median: {np.median(selectivities):.4f}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 5))

    # Top: selectivity histogram
    ax1.hist(
        selectivities * 100, bins=50, color="#1f77b4", edgecolor="white", linewidth=0.5
    )
    ax1.set_title(
        "True selectivity (% of rows passing the filter)",
        fontsize=TITLE_FONTSIZE,
        color=FONT_COLOR,
    )
    ax1.set_xlabel(
        "Selectivity (%)", fontsize=LABEL_FONTSIZE, color=FONT_COLOR, labelpad=10
    )
    ax1.set_ylabel("Count", fontsize=LABEL_FONTSIZE, color=FONT_COLOR, labelpad=10)
    style_ax(ax1)

    # Bottom: range distance histogram
    ax2.hist(distances, bins=50, color="#1f77b4", edgecolor="white", linewidth=0.5)
    ax2.set_title(
        "Range query distances (range_end − range_start)",
        fontsize=TITLE_FONTSIZE,
        color=FONT_COLOR,
    )
    ax2.set_xlabel(
        "Distance (range_end − range_start)",
        fontsize=LABEL_FONTSIZE,
        color=FONT_COLOR,
        labelpad=10,
    )
    ax2.set_ylabel("Count", fontsize=LABEL_FONTSIZE, color=FONT_COLOR, labelpad=10)
    style_ax(ax2)

    fig.tight_layout()
    fig.savefig("arxiv_range_query_distances.pdf", dpi=PLOT_DPI, bbox_inches="tight")
    print("\nSaved arxiv_range_query_distances.pdf")
