
# streamlit_app.py
# Streamlit app to visualize power vs. cost heatmap and budget-power curve
# keeping naming conventions & columns from the original notebooks.
import io
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from utils_hp.utils_single_sweeps import (
    DatasetConfig,
    load_power_tables,
    prepare_power_tables_for_plotting,
    compute_cost_per_row,
    compute_budget_boundary,
    find_best_combo_under_budget,
    build_budget_power_curve,
)

st.set_page_config(page_title="CRISPR Power vs Budget", layout="wide")

# ------------------------------
# Sidebar controls
# ------------------------------
# Switch Paages
st.sidebar.header("Navigation")
if st.sidebar.button("Go to Single Gene Calculator"):
    st.switch_page("pages/streamlit_sg.py")

st.sidebar.header("Inputs")

dataset_name = st.sidebar.selectbox(
    "Dataset",
    options=["Gasperini (high MOI)", "Weissman (low MOI)"],
    index=0,
    help="Choose which result table to load (affects MOI and file names).",
)

alpha = st.sidebar.selectbox(
    "alpha",
    options=[0.05, 0.10],
    index=1,
    help="Choose the significance threshold",
)

budget_eur = st.sidebar.number_input(
    "Budget (EUR)",
    min_value=0.0, value=10000.0, step=1000.0, format="%.0f",
    help="Total available budget (euros).",
)

st.sidebar.markdown("---")
st.sidebar.subheader("Cost parameters")

num_reads_per_flow_cell = st.sidebar.number_input(
    "Reads per flow cell (Mio.)", min_value=1, value=400, step=10, format="%.0f",
    help="Total sequencing reads produced per flow cell, in millions."
) * 1e6
num_cells_per_lane = st.sidebar.number_input(
    "Cells per lane", min_value=1.0, value=20000.0, step=1000.0, format="%.0f",
    help="Number of cells loaded per sequencing lane."
)
num_lanes_per_kit = st.sidebar.number_input(
    "Lanes per kit", min_value=1, value=6, step=1, format="%d",
    help="Number of sequencing lanes included in one kit."
)
lib_prep_cost_per_cell = st.sidebar.number_input(
    "Lib prep cost / cell (EUR)", min_value=0.0, value=0.05, step=0.01, format="%.2f",
    help="Library preparation cost per cell in EUR."
)
seq_cost_per_mio = st.sidebar.number_input(
    "Seq cost / 1M reads (EUR)", min_value=0.0, value=3.42, step=0.1, format="%.2f",
    help="Sequencing cost in EUR per million reads"
)

# --- Run button gating ---
if "run_pressed" not in st.session_state:
    st.session_state.run_pressed = False
run_now = st.sidebar.button("Run", type="primary")
if run_now:
    st.session_state.run_pressed = True

if not st.session_state.run_pressed:
    st.info("Set parameters on the left, then click **Run** to generate plots.")
    st.stop()



# ------------------------------
# Load + prepare data
# ------------------------------
cfg = DatasetConfig(
    dataset_name=dataset_name,
    test_type_ui="empirical",
    mt_method="FDR",
    alpha=alpha,
    #data_dir=Path(data_dir),
)

with st.spinner("Loading power tables..."):
    power_allLFC_df = load_power_tables(cfg)

power_allLFC_df_plot = prepare_power_tables_for_plotting(power_allLFC_df)

# Compute cost per row on the ALL-LFC table (aggregated power)
power_cost_df = compute_cost_per_row(
    power_allLFC_df_plot,
    num_reads_per_flow_cell=num_reads_per_flow_cell,
    num_cells_per_lane=num_cells_per_lane,
    num_lanes_per_kit=num_lanes_per_kit,
    lib_prep_cost_per_cell=lib_prep_cost_per_cell,
    seq_cost_per_mio=seq_cost_per_mio,
)

# ------------------------------
# Heatmap + budget boundary + star for best combo
# ------------------------------
st.header("Results")
plot_left, plot_right = st.columns([1, 1], gap="large")

with plot_left:
    st.subheader("Power heatmap (All LFC)")
    # Pivot for heatmap: rows = NCellsPerElement, cols = ReadDepth
    # Ensure correct ordering via categoricals already set
    pivot_power = power_cost_df.pivot_table(
        index="NCellsPerElement_cat",
        columns="ReadDepth",
        values="Power",
        aggfunc="mean",
    )

    # Build grid-aligned heatmap with black background (legacy style)
    pivot_power = power_cost_df.pivot_table(
        index="NCellsPerElement_cat",
        columns="ReadDepth",
        values="Power",
        aggfunc="mean",
    ).sort_index(axis=0).sort_index(axis=1)

    ny, nx = pivot_power.shape
    fig_hm, ax_hm = plt.subplots(figsize=(7.5, 9))

    # pcolormesh on unit grid so the boundary "steps" sit on cell edges
    x_edges = np.arange(nx + 1)
    y_edges = np.arange(ny + 1)
    mesh = ax_hm.pcolormesh(
        x_edges, y_edges, pivot_power.values,
        cmap="viridis", vmin=0, vmax=1, shading="flat"
    )

    # Axes styling (white like legacy)
    ax_hm.set_xlabel("Read Depth (UMIs per cell)")
    ax_hm.set_ylabel("Cells per Element")
    ax_hm.set_xticks(np.arange(nx) + 0.5)
    ax_hm.set_yticks(np.arange(ny) + 0.5)
    ax_hm.set_xticklabels([str(int(c)) for c in pivot_power.columns], rotation=45, ha="right")
    ax_hm.set_yticklabels([str(int(ix)) for ix in pivot_power.index])

    cbar = plt.colorbar(mesh, ax=ax_hm)
    cbar.ax.tick_params()
    cbar.set_label("Power", rotation=90)

    # Optional title like "MOI: 1"
    if "MOI" in power_cost_df.columns and not power_cost_df["MOI"].isna().all():
        try:
            moi_val = int(round(float(power_cost_df["MOI"].iloc[0])))
        except Exception:
            moi_val = str(power_cost_df["MOI"].iloc[0])
        ax_hm.set_title(f"MOI: {moi_val}", pad=8)

    # Stepwise budget boundary along cell edges
    boundary_points = compute_budget_boundary(power_cost_df, budget_eur)
    if boundary_points:
        col_vals = list(pivot_power.columns)
        row_vals = list(pivot_power.index)
        col_to_idx = {int(c): i for i, c in enumerate([int(cv) for cv in col_vals])}

        def row_to_idx(v):
            return int(np.argmin([abs(int(rv) - int(v)) for rv in row_vals]))

        xs, ys = [], []
        for rd, max_cells in boundary_points:
            if int(rd) in col_to_idx:
                xs.append(col_to_idx[int(rd)])
                ys.append(row_to_idx(max_cells))
        if xs and ys:
            x_steps = np.array(xs + [xs[-1] + 1], dtype=float)
            y_steps = np.array([y + 1 for y in ys] + [ys[-1] + 1], dtype=float)
            ax_hm.step(x_steps, y_steps, where="post", color="black", linewidth=2.5)

    # Star at best affordable cell center
    best = find_best_combo_under_budget(power_cost_df, budget_eur)
    if best is not None:
        col_vals = list(pivot_power.columns)
        row_vals = list(pivot_power.index)
        try:
            col_idx = [int(c) for c in col_vals].index(int(best["ReadDepth"]))
        except ValueError:
            col_idx = 0
        row_idx = int(np.argmin([abs(int(rv) - int(best["NCellsPerElement"])) for rv in row_vals]))
        ax_hm.scatter([col_idx + 0.5], [row_idx + 0.5], marker="*", s=180, edgecolor="black", facecolor="black", zorder=5)

    # 👉 Save BEFORE rendering (fixes empty downloads) and keep the black background
    png_buf, pdf_buf = io.BytesIO(), io.BytesIO()
    fig_hm.savefig(png_buf, format="png", bbox_inches="tight", dpi=300, facecolor="white")
    fig_hm.savefig(pdf_buf, format="pdf", bbox_inches="tight", facecolor="white")

    # Show (do NOT clear)
    st.pyplot(fig_hm, clear_figure=False)

    # Downloads
    st.download_button("Download heatmap (PNG)", data=png_buf.getvalue(), file_name="power_heatmap.png", mime="image/png")
    st.download_button("Download heatmap (PDF)", data=pdf_buf.getvalue(), file_name="power_heatmap.pdf", mime="application/pdf")
    

    st.markdown("### Best affordable combination")
    if best is None:
        st.info("No combination fits within the provided budget.")
    else:
        st.metric("Max Power", f"{best['Power']:.3f}")
        st.write(
            f"- ReadDepth (UMIs/cell): **{int(best['ReadDepth'])}**\n"
            f"- Cells per Element: **{int(best['NCellsPerElement'])}**\n"
            f"- Estimated Cost: **€{best['cost']:,.0f}**"
        )

# ------------------------------
# Budget → max power curve
# ------------------------------

with plot_right:
    st.subheader("Maximum reachable power as budget increases")
    budgets, max_powers = build_budget_power_curve(power_cost_df, max_budget=None)

    fig_curve, ax_curve = plt.subplots(figsize=(4, 3), dpi=150)
    ax_curve.plot(budgets, max_powers, color = "black", lw=2)
    ax_curve.set_xlabel("Budget (EUR)")
    ax_curve.set_ylabel("Max Power")
    ax_curve.grid(True, alpha=0.3)
    fig_curve.tight_layout()
    st.pyplot(fig_curve, use_container_width=False, clear_figure=False) 

    png2, pdf2 = io.BytesIO(), io.BytesIO()
    fig_curve.savefig(png2, format="png", bbox_inches="tight", dpi=300, facecolor="white")
    fig_curve.savefig(pdf2, format="pdf", bbox_inches="tight", facecolor="white")
    st.download_button("Download curve (PNG)", data=png2.getvalue(), file_name="budget_power_curve.png", mime="image/png")
    st.download_button("Download curve (PDF)", data=pdf2.getvalue(), file_name="budget_power_curve.pdf", mime="application/pdf")

    st.caption("Tip: adjust the cost parameters in the sidebar to reflect your lab's sequencing platform and kit pricing.")
