"""
figures.py

Generate two static, publication-style figures for the project:

Figure 1 (PNG)
    A choropleth map of median household income by Census Tract, with
    address-count contour lines overlaid (derived from tract centroids).

Figure 2 (HTML)
    An interactive scatter plot (Altair): address count vs. uninsured rate,
    colored by income quartile and sized by total population.

Outputs (written to derived-data/):
    - figure1_choropleth.png
    - figure2_scatter.html

How to run:
    python figures.py
"""

import os
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

import altair as alt
from scipy.interpolate import griddata
from pathlib import Path

# -----------------------------------------------------------------------------
# Robust project paths
# This file lives in: <root>/code/figures.py
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DERIVED_DIR = DATA_DIR / "derived-data"

DATA_PATH = DERIVED_DIR / "merged_tract.geojson"
OUT_DIR = DERIVED_DIR

def _require_columns(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{name} is missing required columns: {missing}")


def make_figure_1(gdf: gpd.GeoDataFrame, out_dir: Path) -> str:
    """
    Figure 1: Choropleth of median household income + address-count contours.
    """
    _require_columns(
        gdf,
        ["geometry", "med_hh_inc", "addr_count"],
        "GeoDataFrame for Figure 1",
    )

    print("\n[Figure 1] Building choropleth + contours...")

    # Reproject to Web Mercator for plotting (meters).
    gdf_plot = gdf.copy().to_crs(epsg=3857)

    # Robust color scaling to reduce the impact of outliers.
    vmin = gdf_plot["med_hh_inc"].quantile(0.05)
    vmax = gdf_plot["med_hh_inc"].quantile(0.95)

    fig, ax = plt.subplots(1, 1, figsize=(10.5, 12))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Choropleth: low income (warm) -> high income (cool)
    gdf_plot.plot(
        column="med_hh_inc",
        cmap="RdYlBu",
        linewidth=0.25,
        edgecolor="#c7c7c7",
        legend=False,
        ax=ax,
        vmin=vmin,
        vmax=vmax,
        missing_kwds={"color": "#efefef"},
    )

    # Build contours from tracts that have address data.
    density_tracts = gdf_plot[gdf_plot["addr_count"] > 0].copy()
    if len(density_tracts) > 10:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            density_tracts["cx"] = density_tracts.geometry.centroid.x
            density_tracts["cy"] = density_tracts.geometry.centroid.y

        x = density_tracts["cx"].to_numpy()
        y = density_tracts["cy"].to_numpy()
        z = density_tracts["addr_count"].to_numpy(dtype=float)

        # Interpolation grid
        xi = np.linspace(x.min(), x.max(), 200)
        yi = np.linspace(y.min(), y.max(), 200)
        xi, yi = np.meshgrid(xi, yi)

        zi = griddata((x, y), z, (xi, yi), method="cubic")

        # Contour levels (address count per tract)
        levels = [500, 1000, 1500, 2000, 3000]
        contour = ax.contour(
            xi,
            yi,
            zi,
            levels=levels,
            colors=["#3b3b3b"],
            alpha=0.5,
            linewidths=[0.8, 1.0, 1.1, 1.3, 1.6],
        )
        ax.clabel(contour, inline=True, fontsize=8, fmt="%d", colors="#3b3b3b")

    # Optional highlight: high address count + low income (illustrative threshold)
    highlight = gdf_plot[(gdf_plot["addr_count"] > 1000) & (gdf_plot["med_hh_inc"] < 35000)].copy()
    if len(highlight) > 0:
        highlight.plot(
            ax=ax,
            color="none",
            edgecolor="#FF6B35",
            linewidth=2.0,
        )

    # Colorbar
    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(cmap="RdYlBu", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02, shrink=0.75)
    cbar.set_label("Median household income ($)", fontsize=11)

    # Legend
    legend_elements = [
        Line2D([0], [0], color="#3b3b3b", alpha=0.6, linewidth=1.2, label="Address-count contour lines"),
        Line2D([0], [0], color="#FF6B35", linewidth=2.0, label="High address count & low income (highlight)"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", frameon=True)

    # Publication-style title + subtitle
    ax.set_title(
        "Chicago South Side\nMedian Household Income by Census Tract (with Address-Count Contours)",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )

    # Maps read better without axes for academic figures
    ax.set_axis_off()

    plt.tight_layout()
    
    out_path = out_dir / "figure1_choropleth.png"
    plt.savefig(str(out_path), dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    
    print(f"  ✓ Saved: {out_path}")
    return str(out_path)


def make_figure_2(gdf: gpd.GeoDataFrame, out_dir: Path) -> str:
    """
    Figure 2: Altair scatter (interactive HTML).
    """
    _require_columns(
        gdf,
        ["addr_count", "pct_no_hlt", "tot_pop", "med_hh_inc", "income_quartile", "GEOID"],
        "GeoDataFrame for Figure 2",
    )

    print("\n[Figure 2] Building Altair scatter...")

    df = gdf[gdf["addr_count"] > 0].copy()

    # Convert to numeric where needed
    df["addr_count"] = df["addr_count"].astype(float)
    df["uninsured_pct"] = df["pct_no_hlt"].astype(float) * 100.0
    df["tot_pop"] = df["tot_pop"].astype(float)
    df["med_hh_inc"] = df["med_hh_inc"].astype(float)

    # Drop geometry for Altair
    if "geometry" in df.columns:
        df = df.drop(columns=["geometry"])

    # Map Chinese quartile labels to English (keep original if already English)
    mapping = {
        "Q1低收入": "Q1 (Lowest)",
        "Q2中低收入": "Q2",
        "Q3中高收入": "Q3",
        "Q4高收入": "Q4 (Highest)",
    }
    df["income_quartile_en"] = df["income_quartile"].map(mapping).fillna(df["income_quartile"])

    domain = ["Q1 (Lowest)", "Q2", "Q3", "Q4 (Highest)"]
    color_scale = alt.Scale(
        domain=domain,
        range=["#d73027", "#fc8d59", "#91bfdb", "#4575b4"],
    )

    base = alt.Chart(df).mark_circle(opacity=0.75, stroke="#ffffff", strokeWidth=0.5).encode(
        x=alt.X(
            "addr_count:Q",
            title="Address count per tract",
            scale=alt.Scale(zero=False),
        ),
        y=alt.Y(
            "uninsured_pct:Q",
            title="Uninsured rate (%)",
        ),
        size=alt.Size(
            "tot_pop:Q",
            title="Total population",
            scale=alt.Scale(range=[30, 500]),
            legend=alt.Legend(title="Population", orient="bottom-right"),
        ),
        color=alt.Color(
            "income_quartile_en:N",
            title="Income quartile",
            scale=color_scale,
            legend=alt.Legend(title="Income quartile", orient="top-right"),
        ),
        tooltip=[
            alt.Tooltip("GEOID:N", title="Tract GEOID"),
            alt.Tooltip("addr_count:Q", title="Address count", format=","),
            alt.Tooltip("uninsured_pct:Q", title="Uninsured (%)", format=".1f"),
            alt.Tooltip("med_hh_inc:Q", title="Median income ($)", format=",.0f"),
            alt.Tooltip("tot_pop:Q", title="Population", format=","),
            alt.Tooltip("income_quartile_en:N", title="Income quartile"),
        ],
    ).properties(
        width=700,
        height=420,
        title=alt.TitleParams(
            text="Address Count and Uninsured Rate by Census Tract (Chicago South Side)",
            subtitle="Point size = population; color = income quartile. Dashed line shows overall linear trend.",
            fontSize=14,
        ),
    )

    # Overall regression line
    reg = base.transform_regression("addr_count", "uninsured_pct").mark_line(
        color="#111111",
        strokeWidth=2,
        strokeDash=[5, 3],
        opacity=0.7,
    )

    # LOESS by income group (optional smooth trend per quartile)
    loess = base.transform_loess(
        "addr_count",
        "uninsured_pct",
        groupby=["income_quartile_en"],
        bandwidth=0.5,
    ).mark_line(
        strokeWidth=1.5,
        opacity=0.35,
    ).encode(color=alt.Color("income_quartile_en:N", scale=color_scale, legend=None))

    chart = (base + reg + loess).configure(
        background="white"
    ).configure_axis(
        labelColor="#111111",
        titleColor="#111111",
        gridColor="#e6e6e6",
        domainColor="#999999",
        tickColor="#999999",
    ).configure_legend(
        labelColor="#111111",
        titleColor="#111111",
        fillColor="white",
        strokeColor="#dddddd",
    ).configure_title(
        color="#111111",
        subtitleColor="#444444",
    ).configure_view(
        stroke=None
    )

    out_path = out_dir / "figure2_scatter.html"
    chart.save(str(out_path))
    print(f"  ✓ Saved: {out_path}")
    return str(out_path)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading merged tract dataset...")
    gdf = gpd.read_file(DATA_PATH)
    print(f"  Tracts: {len(gdf):,}")
    if "addr_count" in gdf.columns:
        print(f"  Tracts with addresses: {(gdf['addr_count'] > 0).sum():,}")

    fig1 = make_figure_1(gdf, OUT_DIR)
    fig2 = make_figure_2(gdf, OUT_DIR)

    print("\nDone.")
    print(f"Figure 1: {fig1}")
    print(f"Figure 2: {fig2}")


if __name__ == "__main__":
    main()
