import os
from io import BytesIO
import time
import zipfile


import streamlit as st
import pandas as pd

import torch, inspect
st.write("Torch version:", torch.__version__)
st.write("torch.load signature:", inspect.signature(torch.load))

# Your project utilities
from utils.utils_single_core import SimulationConfig, TestConfig  # configs  ← :contentReference[oaicite:3]{index=3}
from utils.utils_single_sweeps import (  # sweep runners               ← :contentReference[oaicite:4]{index=4}
    power_vs_cells, power_vs_lfc, power_vs_nguides, power_vs_moi, power_vs_gene_mean
)
from utils.utils_single_plots import plot_power_generic  # matplotlib figure  ← :contentReference[oaicite:5]{index=5}

def render_results_from_state():
    """Render plot, table, and download buttons using st.session_state['results']."""
    res = st.session_state.get("results")
    if not res:
        return

    df = res["df"]
    fig = res["fig"]
    x_col = res["x_col"]
    title = res["title"]

    # Buffers were already prepared and saved to session_state
    csv_bytes = res["csv_bytes"]
    png_buf = res["png_buf"]
    pdf_buf = res["pdf_buf"]
    zip_buf = res["zip_buf"]

    st.subheader("📊 Results")

    # Show the plot (do NOT clear the figure)
    st.pyplot(fig, width='stretch')

    # Table
    st.markdown("**Summary table**")
    st.dataframe(df, width='stretch', hide_index=True)

    # Downloads (CSV, PNG, PDF, ZIP)
    st.download_button(
        "Download data (CSV)",
        data=csv_bytes,
        file_name="power_summary.csv",
        mime="text/csv",
        width='stretch',
        key="dl_csv",
    )
    st.download_button(
        "Download plot (PNG)",
        data=png_buf,
        file_name="power_plot.png",
        mime="image/png",
        width='stretch',
        key="dl_png",
    )
    st.download_button(
        "Download plot (PDF)",
        data=pdf_buf,
        file_name="power_plot.pdf",
        mime="application/pdf",
        width='stretch',
        key="dl_pdf",
    )
    st.download_button(
        "Download ALL (ZIP)",
        data=zip_buf,
        file_name=f"power_results_{int(time.time())}.zip",
        mime="application/zip",
        width='stretch',
        key="dl_zip",
    )


st.set_page_config(page_title="PerTurbo Power Explorer", layout="wide")
st.title("🔬 PerTurbo Power Explorer — Streamlit")
st.caption("Configure a simulation on the left, run a sweep, and download the plot/data.")

with st.expander("ℹ️ How this works", expanded=False):
    st.markdown(
        "- Uses your existing utilities to simulate data and estimate power.\n"
        "- Heavy lifting happens in `utils_single_*` modules (training a small PerTurbo model each run).\n"
        "- For quick tests, lower `max_epochs` / `batch_size` or set a small #cells.\n"
    )

# ------------------------------
# Sidebar: Simulation & Test cfg
# ------------------------------
st.sidebar.header("⚙️ Simulation")
model_dir = st.sidebar.text_input("Model directory", value="save_model")
real_path = st.sidebar.text_input("Parameters from Real data (.npz)", value="save_model/model/reference_stats_compact.npz")
accelerator = st.sidebar.selectbox("Accelerator (for simulation)", ["cpu", "gpu"], index=0)

st.sidebar.divider()
gene_name = st.sidebar.text_input("Gene name", value="GATA1")
n_genes = st.sidebar.number_input("#Genes (positives)", 10, 20000, 500, step=10)

mean_mode = st.sidebar.selectbox("Gene mean mode", ["original", "fixed"], index=0)
mean_expression = (
    st.sidebar.number_input("Mean expression (fixed)", 0.0, 1e9, 10.0, step=0.5)
    if mean_mode == "fixed" else None
)

lfc_mode = st.sidebar.selectbox("LFC mode", ["fixed", "normal", "original"], index=0)
lfc_value = st.sidebar.number_input("LFC value (fixed mode)", -100.0, 100.0, -0.5, step=0.05)
lfc_normal_mean = st.sidebar.number_input("LFC normal mean", -100.0, 100.0, -0.5, step=0.05)
lfc_normal_sd = st.sidebar.number_input("LFC normal sd", 0.0, 10.0, 1.0, step=0.05)

guide_eff_mode = st.sidebar.selectbox("Guide efficacy mode", ["beta", "fixed_list", "original"], index=0)
if guide_eff_mode == "fixed_list":
    ge_list_str = st.sidebar.text_input("Guide efficacy list (comma-separated)", value="1,0.67,0.33,0.0")
    ge_fixed_list = [float(x) for x in ge_list_str.split(",")] if ge_list_str.strip() else None
    ge_a = ge_b = None
elif guide_eff_mode == "beta":
    ge_a = st.sidebar.number_input("Guide efficacy Beta a", 0.1, 100.0, 2.5, step=0.1)
    ge_b = st.sidebar.number_input("Guide efficacy Beta b", 0.1, 100.0, 1.5, step=0.1)
    ge_fixed_list = None
else:
    ge_fixed_list = None
    ge_a = ge_b = None

n_grna_per_element = st.sidebar.number_input("Guides per element", 1, 32, 4, step=1)
moi = st.sidebar.number_input("MOI (used to scale total cells)", 1.0, 1e5, 30.0, step=1.0)
read_depth_adjust_factor = st.sidebar.number_input("Read-depth scaling", 0.0, 1000.0, 1.0, step=0.1)
random_seed = st.sidebar.number_input("Random seed", 0, 10_000_000, 57, step=1)

st.sidebar.header("🧪 Testing")
alpha = st.sidebar.number_input("Alpha (test size)", 0.0, 1.0, 0.1, step=0.01)
mt_method = st.sidebar.selectbox("Multiple testing", ["FDR", "none"], index=0)
test_type = st.sidebar.selectbox("p-value type", ["empirical", "fixed"], index=0)
max_epochs = st.sidebar.number_input("Max epochs", 1, 10_000, 200, step=10)
lr = st.sidebar.number_input("Learning rate", 1e-5, 1.0, 0.01, step=0.001, format="%.5f")
batch_size = st.sidebar.number_input("Batch size", 8, 200_000, 1024, step=8)
early_stopping_patience = st.sidebar.number_input("ES Patience", 1, 1000, 10, step=1)
early_stopping_min_delta = st.sidebar.number_input("ES Min Delta", 0.0, 1.0, 0.001, step=0.001, format="%.4f")

# ------------------------------
# Main: Sweep selector & params
# ------------------------------
left, right = st.columns([1, 2], gap="large")
with left:
    st.subheader("📈 Sweep")
    mode = st.selectbox("X-axis", ["Cells per element", "LFC", "Guides per element", "MOI", "Gene mean (approx)"])

    fixed_cells = st.number_input("Fixed cells/element (for LFC/Guides/MOI/GeneMean)", 10, 1_000_000, 200, step=10)
    min_val = st.number_input("Min", value=0.2)
    max_val = st.number_input("Max", value=1.0)
    n_bins = st.number_input("Bins (or points)", 2, 200, 5, step=1)
    step = st.number_input("Step (Guides mode only)", 1, 100, 1, step=1)

    col_run, col_stop = st.columns(2)
    with col_run:
        run_btn = st.button("🚀 Run", type="primary", width='stretch', key="run_btn")
    with col_stop:
        stop_btn = st.button("⏹️ Stop", type="secondary", width='stretch', key="stop_btn")

    # Update an abort flag in session_state
    if stop_btn:
        st.session_state["abort"] = True
    if run_btn:
        st.session_state["abort"] = False  # reset on new run

# Construct configs
cfg = SimulationConfig(
    model_dir=model_dir,
    real_data_ref_path=real_path,
    accelerator=accelerator,
    gene_name=gene_name,
    n_genes=int(n_genes),
    mean_mode=mean_mode,
    mean_expression=float(mean_expression) if mean_mode == "fixed" and mean_expression is not None else None,
    lfc_mode=lfc_mode,
    lfc_value=float(lfc_value),
    lfc_normal_mean=float(lfc_normal_mean),
    lfc_normal_sd=float(lfc_normal_sd),
    guide_eff_mode=guide_eff_mode,
    guide_eff_fixed_list=ge_fixed_list,
    guide_eff_beta_a=float(ge_a) if ge_a is not None else 2.5,
    guide_eff_beta_b=float(ge_b) if ge_b is not None else 1.5,
    moi=float(moi),
    n_grna_per_element=int(n_grna_per_element),
    read_depth_adjust_factor=float(read_depth_adjust_factor),
    random_seed=int(random_seed),
)
tcfg = TestConfig(
    alpha=float(alpha), mt_method=mt_method, test_type=test_type,
    max_epochs=int(max_epochs), lr=float(lr), batch_size=int(batch_size),
    early_stopping_patience=int(early_stopping_patience), early_stopping_min_delta=float(early_stopping_min_delta),
)

if run_btn:
    if not os.path.exists(model_dir):
        st.error("Model directory not found. Point to the folder that contains a saved PerTurbo model (subdir 'model').")
        st.stop()
    if not os.path.exists(real_path):
        st.error("Real data path not found. Provide a file path to '.npz' or a directory that contains it.")
        st.stop()

df = None
if run_btn:
    try:
        with st.status("Running sweep… this can be compute-intensive (training involved).", expanded=False) as status:
            abort = lambda: st.session_state.get("abort", False)
            
            if mode == "Cells per element":
                df = power_vs_cells(int(min_val), int(max_val), int(n_bins), cfg, tcfg, abort_flag=abort)
                x_col, x_label, title = "NCellsPerElement", "Number of Cells per Element", "Power vs Number of Cells per Element"
            elif mode == "LFC":
                df = power_vs_lfc(float(min_val), float(max_val), int(n_bins), fixed_cells, cfg, tcfg, abort_flag=abort)
                x_col, x_label, title = "LFC", "LFC", "Power vs LFC"
            elif mode == "Guides per element":
                df = power_vs_nguides(int(min_val), int(max_val), int(step), fixed_cells, cfg, tcfg, abort_flag=abort)
                x_col, x_label, title = "NGuidesPerElement", "Number of Guides per Element", "Power vs #Guides/Element"
            elif mode == "MOI":
                df = power_vs_moi(float(min_val), float(max_val), int(n_bins), fixed_cells, cfg, tcfg, abort_flag=abort)
                x_col, x_label, title = "MOI", "MOI", "Power vs MOI"
            else:  # Gene mean (approx)
                df = power_vs_gene_mean(float(min_val), float(max_val), int(n_bins), fixed_cells, cfg, tcfg, abort_flag=abort)
                x_col, x_label, title = "GeneMeanApprox", "Gene mean (approx)", "Power vs Gene Mean (approx)"
            status.update(label="Finished.", state="complete", expanded=False)
    except Exception as e:
        st.exception(e)

# If we got results this run, build the figure and all download buffers, then persist.
if df is not None and not df.empty:
    # 1) Build the figure
    fig, _ = plot_power_generic(df, x_col=x_col, x_label=x_label, title=title)

    # 2) Build download buffers BEFORE showing the plot (prevents blank images)
    csv_bytes = df.to_csv(index=False).encode("utf-8")

    png_buf = BytesIO()
    fig.savefig(png_buf, format="png", dpi=200, bbox_inches="tight")
    png_buf.seek(0)

    pdf_buf = BytesIO()
    fig.savefig(pdf_buf, format="pdf", bbox_inches="tight")
    pdf_buf.seek(0)

    zip_buf = BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("power_summary.csv", csv_bytes)
        zf.writestr("power_plot.png", png_buf.getvalue())
        zf.writestr("power_plot.pdf", pdf_buf.getvalue())
    zip_buf.seek(0)

    # 3) Persist everything so reruns (triggered by downloads) don't lose the view
    st.session_state["results"] = {
        "df": df,
        "fig": fig,
        "x_col": x_col,
        "title": title,
        "csv_bytes": csv_bytes,
        "png_buf": png_buf,
        "pdf_buf": pdf_buf,
        "zip_buf": zip_buf,
    }

with right:
    # If we have new results this run, they were already saved to session_state above.
    # Always render from session_state so clicking a download (which triggers a rerun)
    # doesn't wipe the plot/table.
    if st.session_state.get("results"):
        render_results_from_state()
    elif run_btn:
        st.warning("No results returned. Try different parameters.")
