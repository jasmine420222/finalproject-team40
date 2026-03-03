"""
app.py
Chicago South Side: Housing Density & Health Inequality

This dashboard is designed for non-technical audiences.

What this dashboard shows
1) Map: where density / income / uninsured rates are high or low
2) Relationships: income ↔ uninsured, density ↔ uninsured
3) Priority list: tracts that meet high-density + low-income + high-uninsured thresholds

Data input (no data is fabricated):
- derived-data/merged_tract.geojson
  Produced by preprocessing.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
import altair as alt
import folium
import streamlit as st
from streamlit_folium import st_folium
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="South Side Inequality Dashboard",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# Styling (light theme)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #ffffff; }
    .main .block-container { padding: 1.5rem 2rem; }

    h1 { 
        color: #111111 !important; 
        font-family: 'Georgia', serif;
        border-bottom: 2px solid #FF6B35; 
        padding-bottom: 10px;
    }

    h2, h3 { color: #222222 !important; }

    [data-testid="metric-container"] {
        background-color: #f4f4f4; 
        border: 1px solid #dddddd;
        border-radius: 10px; 
        padding: 12px;
    }

    [data-testid="metric-container"] label { 
        color: #555555 !important; 
    }

    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #FF6B35 !important; 
        font-size: 1.6rem;
    }

    [data-testid="stSidebar"] { 
        background-color: #f8f9fa; 
    }

    [data-testid="stSidebar"] * { 
        color: #111111 !important; 
    }

    .stSelectbox > div > div { 
        background-color: #ffffff !important; 
        color: #111111 !important; 
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data() -> gpd.GeoDataFrame:
    """
    Load the tract-level GeoJSON built by preprocessing.py.
    This function does not create or fabricate any data; it only reads and formats.
    """
    ROOT = Path(__file__).resolve().parents[1]  # root
    DATA_PATH = ROOT / "data" / "derived-data" / "merged_tract.geojson"

    gdf = gpd.read_file(DATA_PATH)

    # Ensure numeric types (safe conversion)
    numeric_cols = [
        "addr_count", "med_hh_inc", "pct_no_hlt", "pop_0_17", "tot_pop",
        "area_sqkm", "addr_per_sqkm", "pop_per_sqkm"
    ]
    for c in numeric_cols:
        if c in gdf.columns:
            gdf[c] = pd.to_numeric(gdf[c], errors="coerce")

    # Derived, user-facing fields
    # pct_no_hlt in your data is likely in [0,1] so convert to percent.
    if "pct_no_hlt" in gdf.columns:
        gdf["uninsured_pct"] = gdf["pct_no_hlt"] * 100
    else:
        gdf["uninsured_pct"] = np.nan

    if "pop_0_17" in gdf.columns and "tot_pop" in gdf.columns:
        gdf["child_pct"] = gdf["pop_0_17"] / gdf["tot_pop"].replace(0, np.nan) * 100
    else:
        gdf["child_pct"] = np.nan

    # Friendly tract label
    if "NAMELSAD" in gdf.columns and gdf["NAMELSAD"].notna().any():
        gdf["tract_name"] = gdf["NAMELSAD"].astype(str)
    elif "TRACTCE" in gdf.columns:
        gdf["tract_name"] = "Census Tract " + gdf["TRACTCE"].astype(str)
    else:
        gdf["tract_name"] = gdf["GEOID"].astype(str)

    # Income quartile label (should exist from preprocessing.py, but keep safe fallback)
    if "income_quartile" not in gdf.columns:
        # Create it if missing (still based on existing income values, not fabricated)
        if "med_hh_inc" in gdf.columns and gdf["med_hh_inc"].notna().any():
            gdf["income_quartile"] = pd.qcut(
                gdf["med_hh_inc"], q=4,
                labels=["Q1 (Lowest)", "Q2", "Q3", "Q4 (Highest)"]
            )
        else:
            gdf["income_quartile"] = "undefined"

    return gdf


def winsorize(s: pd.Series, lo: float = 0.01, hi: float = 0.99) -> pd.Series:
    """Clip extreme values for more readable plots."""
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return s
    a, b = s.quantile(lo), s.quantile(hi)
    return s.clip(a, b)


# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────
gdf = load_data()


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar controls
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧭 Explore the South Side")
    st.caption("A policy-friendly view of density, income, and uninsured rates.")
    st.markdown("---")

    map_metric_options = {
        "Address density (addresses per sq km)": "addr_per_sqkm",
        "Median household income ($)": "med_hh_inc",
        "Uninsured rate (%)": "uninsured_pct",
        "Population density (people per sq km)": "pop_per_sqkm",
        "Children share (%)": "child_pct"
    }
    map_metric_label = st.selectbox("Map metric", list(map_metric_options.keys()), index=0)
    map_metric = map_metric_options[map_metric_label]

    st.markdown("---")
    st.markdown("### Filter by income")

    # Safe min/max for slider
    inc_series = gdf["med_hh_inc"].replace([np.inf, -np.inf], np.nan).dropna()
    if inc_series.empty:
        min_inc, max_inc = 0, 1
    else:
        min_inc, max_inc = int(inc_series.min()), int(inc_series.max())

    income_range = st.slider(
        "Median income range ($)",
        min_value=min_inc,
        max_value=max_inc,
        value=(min_inc, max_inc),
        step=1000
    )

    st.markdown("---")
    st.markdown("### Define priority tracts")
    st.caption("Flag tracts that are **high density + low income + high uninsured**.")

    density_q = st.slider("High density threshold (quantile)", 0.50, 0.95, 0.75, 0.05)
    low_income_q = st.slider("Low income threshold (quantile)", 0.05, 0.50, 0.25, 0.05)
    uninsured_q = st.slider("High uninsured threshold (quantile)", 0.50, 0.95, 0.75, 0.05)

    st.markdown("---")
    st.markdown("### Plot settings")
    use_log_density = st.checkbox("Use log scale for density in scatterplots", value=True)
    clip_outliers = st.checkbox("Clip extreme values (1%–99%)", value=True)

    st.markdown("---")
    st.markdown("### Data notes")
    st.caption("ACS 5-year estimates + tract-level aggregation of address points.")
    st.caption("Address density is a proxy (not official housing units).")


# ─────────────────────────────────────────────────────────────────────────────
# Build filtered dataset (IMPORTANT: must exist BEFORE using metrics/tabs)
# ─────────────────────────────────────────────────────────────────────────────
filtered = gdf[
    (gdf["med_hh_inc"] >= income_range[0]) &
    (gdf["med_hh_inc"] <= income_range[1])
].copy()

# If filter removes everything, stop gracefully
if filtered.empty:
    st.markdown("# 🏙️ Chicago South Side: Housing Density & Health Inequality")
    st.warning("No tracts match the current income filter. Please widen the income range.")
    st.stop()

# Thresholds based on the filtered distribution (transparent & responsive)
d_th = filtered["addr_per_sqkm"].replace([np.inf, -np.inf], np.nan).dropna().quantile(density_q)
i_th = filtered["med_hh_inc"].replace([np.inf, -np.inf], np.nan).dropna().quantile(low_income_q)
u_th = filtered["uninsured_pct"].replace([np.inf, -np.inf], np.nan).dropna().quantile(uninsured_q)

filtered["priority_flag"] = (
    (filtered["addr_per_sqkm"] >= d_th) &
    (filtered["med_hh_inc"] <= i_th) &
    (filtered["uninsured_pct"] >= u_th)
)


# ─────────────────────────────────────────────────────────────────────────────
# Header: research question & structure
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("# 🏙️ Chicago South Side: Housing Density & Health Inequality")

st.markdown("""
### Research question
**Do higher-density tracts in Chicago's South Side tend to be lower-income and have higher uninsured rates?**  
This dashboard provides **descriptive, correlational evidence** using Census Tract-level indicators.

### Dashboard structure
1) **Map** the spatial distribution of key indicators  
2) **Visualize relationships** (income ↔ uninsured, density ↔ uninsured)  
3) **Generate an action list**: priority tracts under clear thresholds
""")


# ─────────────────────────────────────────────────────────────────────────────
# Top KPIs (now safe because 'filtered' is defined)
# ─────────────────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)

with k1:
    st.metric("Avg median income", f"${filtered['med_hh_inc'].mean():,.0f}")

with k2:
    st.metric("Avg uninsured rate", f"{filtered['uninsured_pct'].mean():.1f}%")

with k3:
    st.metric("Avg address density", f"{filtered['addr_per_sqkm'].mean():,.0f} /km²")

with k4:
    st.metric("Priority tracts", f"{int(filtered['priority_flag'].sum()):,}")


st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# Tabs (no regression tab)
# ─────────────────────────────────────────────────────────────────────────────
tabs = st.tabs(["🗺️ 1) Map", "📈 2) Relationships", "📌 3) Priority tracts"])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: MAP
# ─────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    left, right = st.columns([3, 2])

    with left:
        st.markdown(f"## Map: {map_metric_label}")

        # Center map on the average centroid of filtered tracts
        center_lat = float(filtered.geometry.centroid.y.mean())
        center_lon = float(filtered.geometry.centroid.x.mean())

        m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="OpenStreetMap")

        # Compute vmin/vmax for map legend range (robust)
        s = filtered[map_metric].replace([np.inf, -np.inf], np.nan).dropna()
        if s.empty:
            vmin, vmax = 0.0, 1.0
        else:
            vmin = float(s.quantile(0.05))
            vmax = float(s.quantile(0.95))

        fill_color = "YlOrRd" if map_metric != "med_hh_inc" else "RdYlBu_r"

        folium.Choropleth(
            geo_data=filtered.__geo_interface__,
            data=filtered[["GEOID", map_metric]],
            columns=["GEOID", map_metric],
            key_on="feature.properties.GEOID",
            fill_color=fill_color,
            fill_opacity=0.75,
            line_opacity=0.2,
            nan_fill_color="#666666",
            legend_name=map_metric_label
        ).add_to(m)

        tooltip_fields = ["tract_name", "med_hh_inc", "uninsured_pct", "addr_per_sqkm", "tot_pop"]
        tooltip_aliases = ["Tract:", "Median income ($):", "Uninsured (%):", "Address density (/km²):", "Population:"]

        folium.GeoJson(
            filtered.__geo_interface__,
            style_function=lambda x: {"fillOpacity": 0, "weight": 0.5, "color": "#444444"},
            tooltip=folium.GeoJsonTooltip(
                fields=tooltip_fields,
                aliases=tooltip_aliases,
                localize=True,
                sticky=True,
                labels=True,
                style="background-color:#1a1a2e;color:white;font-size:12px;border:1px solid #444466;border-radius:6px;padding:8px;"
            )
        ).add_to(m)

        # Highlight priority tracts with orange outlines
        pr = filtered[filtered["priority_flag"]]
        if not pr.empty:
            folium.GeoJson(
                pr.__geo_interface__,
                style_function=lambda x: {"fillOpacity": 0, "weight": 3, "color": "#FF6B35"},
                name="Priority tracts"
            ).add_to(m)

        map_key = f"map_{map_metric}_{income_range[0]}_{income_range[1]}_{density_q}_{low_income_q}_{uninsured_q}"
        st_folium(m, height=540, use_container_width=True, key=map_key)

    with right:
        st.markdown("## How to read this map")
        st.info(
            "Each polygon is a Census Tract.\n\n"
            "Address density = addresses / tract area (sq km).\n\n"
            "Orange outlines = priority tracts meeting all three conditions:\n"
            "- High density\n- Low income\n- High uninsured"
        )

        st.markdown("## Distribution (filtered tracts)")
        dist = filtered[map_metric].replace([np.inf, -np.inf], np.nan).dropna()
        dist_df = pd.DataFrame({"value": dist})

        hist = alt.Chart(dist_df).mark_bar(opacity=0.85).encode(
            x=alt.X("value:Q", bin=alt.Bin(maxbins=25), title=map_metric_label),
            y=alt.Y("count():Q", title="Number of tracts")
        ).properties(height=200).configure(background="#1a1a2e").configure_axis(
            labelColor="white", titleColor="white", gridColor="#333355"
        ).configure_view(stroke=None)

        st.altair_chart(hist, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: RELATIONSHIPS
# ─────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown("## Relationships (visual evidence)")

    df = filtered[[
        "tract_name", "income_quartile", "med_hh_inc", "uninsured_pct", "addr_per_sqkm", "tot_pop"
    ]].replace([np.inf, -np.inf], np.nan).dropna().copy()

    if df.empty:
        st.warning("Not enough valid data after filtering to plot relationships. Try widening filters.")
        st.stop()

    # Optional outlier clipping for clearer charts (still using real data, just clipped display)
    if clip_outliers:
        df["med_hh_inc_c"] = winsorize(df["med_hh_inc"])
        df["uninsured_pct_c"] = winsorize(df["uninsured_pct"])
        df["addr_per_sqkm_c"] = winsorize(df["addr_per_sqkm"])
    else:
        df["med_hh_inc_c"] = df["med_hh_inc"]
        df["uninsured_pct_c"] = df["uninsured_pct"]
        df["addr_per_sqkm_c"] = df["addr_per_sqkm"]

    st.markdown("### A) Income vs uninsured rate")

    base_a = alt.Chart(df).mark_circle(opacity=0.7, stroke="white", strokeWidth=0.2).encode(
        x=alt.X("med_hh_inc_c:Q", title="Median household income ($)"),
        y=alt.Y("uninsured_pct_c:Q", title="Uninsured rate (%)"),
        color=alt.Color("income_quartile:N", title="Income quartile"),
        size=alt.Size("tot_pop:Q", title="Population", scale=alt.Scale(range=[20, 320])),
        tooltip=[
            alt.Tooltip("tract_name:N", title="Tract"),
            alt.Tooltip("med_hh_inc:Q", title="Income ($)", format=",.0f"),
            alt.Tooltip("uninsured_pct:Q", title="Uninsured (%)", format=".1f")
        ]
    ).properties(height=340)

    reg_a = base_a.transform_regression("med_hh_inc_c", "uninsured_pct_c").mark_line(
        color="white", strokeDash=[5, 3], strokeWidth=2, opacity=0.7
    )

    st.altair_chart(
        (base_a + reg_a)
        .configure(background="#1a1a2e")
        .configure_axis(labelColor="white", titleColor="white", gridColor="#333355")
        .configure_view(stroke=None)
        .configure_legend(labelColor="white", titleColor="white", fillColor="#2a2a4a", strokeColor="#444466"),
        use_container_width=True
    )

    st.markdown("---")
    st.markdown("### B) Address density vs uninsured rate")

    if use_log_density:
        df["log_density"] = np.log1p(df["addr_per_sqkm_c"])
        x_field = "log_density:Q"
        x_title = "Address density (log1p)"
        reg_x = "log_density"
    else:
        x_field = "addr_per_sqkm_c:Q"
        x_title = "Address density (addresses per sq km)"
        reg_x = "addr_per_sqkm_c"

    base_b = alt.Chart(df).mark_circle(opacity=0.7, stroke="white", strokeWidth=0.2).encode(
        x=alt.X(x_field, title=x_title),
        y=alt.Y("uninsured_pct_c:Q", title="Uninsured rate (%)"),
        color=alt.Color("income_quartile:N", title="Income quartile"),
        size=alt.Size("tot_pop:Q", title="Population", scale=alt.Scale(range=[20, 320])),
        tooltip=[
            alt.Tooltip("tract_name:N", title="Tract"),
            alt.Tooltip("addr_per_sqkm:Q", title="Density (/km²)", format=",.0f"),
            alt.Tooltip("uninsured_pct:Q", title="Uninsured (%)", format=".1f"),
            alt.Tooltip("med_hh_inc:Q", title="Income ($)", format=",.0f")
        ]
    ).properties(height=340)

    reg_b = base_b.transform_regression(reg_x, "uninsured_pct_c").mark_line(
        color="white", strokeDash=[5, 3], strokeWidth=2, opacity=0.7
    )

    st.altair_chart(
        (base_b + reg_b)
        .configure(background="#1a1a2e")
        .configure_axis(labelColor="white", titleColor="white", gridColor="#333355")
        .configure_view(stroke=None)
        .configure_legend(labelColor="white", titleColor="white", fillColor="#2a2a4a", strokeColor="#444466"),
        use_container_width=True
    )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: PRIORITY TRACTS
# ─────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.markdown("## Priority tracts (action list)")
    st.caption("Flagged tracts meet: high density + low income + high uninsured under current thresholds.")

    pr = filtered[filtered["priority_flag"]].copy()

    if pr.empty:
        st.warning("No tracts meet the current thresholds. Adjust the quantiles in the sidebar.")
        st.stop()

    # Keep only columns we need
    dfp = pr[["tract_name", "med_hh_inc", "uninsured_pct", "addr_per_sqkm", "tot_pop"]].copy()

    # Guard against edge cases
    dfp = dfp.replace([np.inf, -np.inf], np.nan).dropna()
    if dfp.empty:
        st.warning("No priority tracts remain after removing missing values. Try adjusting filters/thresholds.")
        st.stop()

    # Create a simple priority score using standardized components (transparent weights)
    # Higher uninsured, lower income, higher density -> higher score.
    dfp["z_density"] = (dfp["addr_per_sqkm"] - dfp["addr_per_sqkm"].mean()) / dfp["addr_per_sqkm"].std(ddof=0)
    dfp["z_uninsured"] = (dfp["uninsured_pct"] - dfp["uninsured_pct"].mean()) / dfp["uninsured_pct"].std(ddof=0)
    dfp["z_income_low"] = (dfp["med_hh_inc"].mean() - dfp["med_hh_inc"]) / dfp["med_hh_inc"].std(ddof=0)

    # If a standard deviation is zero (rare), avoid NaNs
    dfp[["z_density", "z_uninsured", "z_income_low"]] = dfp[["z_density", "z_uninsured", "z_income_low"]].fillna(0)

    dfp["priority_score"] = 0.40 * dfp["z_uninsured"] + 0.35 * dfp["z_income_low"] + 0.25 * dfp["z_density"]

    # Top 15 table
    top = dfp.sort_values("priority_score", ascending=False).head(15).copy()

    top = top.rename(columns={
        "tract_name": "Tract",
        "med_hh_inc": "Median income ($)",
        "uninsured_pct": "Uninsured (%)",
        "addr_per_sqkm": "Address density (/km²)",
        "tot_pop": "Population",
        "priority_score": "Priority score"
    })

    top["Median income ($)"] = top["Median income ($)"].round(0).astype(int)
    top["Population"] = top["Population"].round(0).astype(int)
    for c in ["Uninsured (%)", "Address density (/km²)", "Priority score"]:
        top[c] = top[c].astype(float).round(2)

    st.markdown("### Top 15 flagged tracts (by a simple priority score)")
    st.dataframe(top, use_container_width=True, hide_index=True)

    # Download full list of flagged tracts
    # CSV export (sort BEFORE renaming, otherwise sort_values will not find the old column name)
if "priority_score" in dfp.columns:
    export_df = (
        dfp.sort_values("priority_score", ascending=False)
           .rename(columns={
               "tract_name": "Tract",
               "med_hh_inc": "Median income ($)",
               "uninsured_pct": "Uninsured (%)",
               "addr_per_sqkm": "Address density (/km²)",
               "tot_pop": "Population",
               "priority_score": "Priority score"
           })
    )
    csv = export_df.to_csv(index=False).encode("utf-8")
else:
    csv = None

st.download_button(
    "Download all flagged tracts (CSV)",
    data=csv,
    file_name="priority_tracts.csv",
    mime="text/csv",
    disabled=(csv is None)
)