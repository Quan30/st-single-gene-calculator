import os
from io import BytesIO
import time
import zipfile


import streamlit as st
import pandas as pd

# Your project utilities
from utils_sg.utils_single_core import SimulationConfig, TestConfig  # configs  ← :contentReference[oaicite:3]{index=3}
from utils_sg.utils_single_sweeps import (  # sweep runners               ← :contentReference[oaicite:4]{index=4}
    power_vs_cells, power_vs_lfc, power_vs_nguides, power_vs_moi, power_vs_gene_mean
)
from utils_sg.utils_single_plots import plot_power_generic  # matplotlib figure  ← :contentReference[oaicite:5]{index=5}

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
st.title("🔬 PerTurbo Single Gene Calculator")
st.caption("Configure a simulation on the left, run a sweep, and download the plot/data.")

with st.expander("ℹ️ How this works", expanded=False):
    st.markdown(
        """
        **What this app does**

        - You choose a real dataset (Gasperini or Weissman) and a simulation design on the left
          (gene, MOI, guides per element, effect sizes, read depth, etc.).
        - The app builds a `SimulationConfig` and a `TestConfig` and calls the
          `power_vs_*` helper functions in `utils_sg.utils_single_sweeps`.
        - For each point along the selected x-axis (cells per element, LFC, guides per element,
          MOI, or gene mean), it:
          1. Simulates a single-gene CRISPR screen under the chosen design,
          2. Fits the PerTurbo model with your training settings,
          3. Runs the statistical test and records whether the effect is detected.
        - Power is estimated as the fraction of simulations where the effect is detected.

        **What you see on the right**

        - A power curve: estimated power vs. the chosen x-axis quantity.
        - A summary table with the exact values for each sweep point.
        - Download buttons for:
          - CSV (summary table),
          - PNG/PDF (plot),
          - ZIP (all of the above in one file).

        **Tips for faster runs**

        - Reduce **Max epochs** or increase **Batch size**.
        - Use fewer **Bins (or points)** or a narrower Min/Max range.
        - Start with smaller designs (fewer cells / guides) to explore settings,
          then scale up once you’ve found interesting regions.
        """
    )

# ------------------------------
# Sidebar: Simulation & Test cfg
# ------------------------------
# Switch Pages
st.sidebar.header("Navigation")
if st.sidebar.button("Back to Heatmap"):
    st.switch_page("streamlit_heatmap.py")

st.sidebar.header("⚙️ Simulation")
#model_dir = st.sidebar.text_input("Model directory", value="path/to/saved_model_dir")
#real_path = st.sidebar.text_input("Real data (mdata.h5mu or folder)", value="path/to/data_or_dir")
orig_data_name = st.sidebar.selectbox("Simulate from data", ["Gasperini (high MOI)", "Weissman (low MOI)"], index=0,
                                     help="Choose which real dataset to base the simulation on (affects MOI and other settings).")

st.sidebar.divider()
gene_name = st.sidebar.text_input("Gene name", value="GATA1",
                                 help="Target gene for which to simulate effects and compute power.")
if orig_data_name == "Gasperini (high MOI)":
    n_genes = st.sidebar.number_input("#Genes (positives)", 10, 20000, 500, step=10,
                                     help="Number of gene replicates to simulate.")
elif orig_data_name == "Weissman (low MOI)":
    n_genes = st.sidebar.number_input("#Genes (positives)", 10, 20000, 100, step=10,
                                     help="Number of gene replicates to simulate.")

mean_mode = st.sidebar.selectbox("Gene mean mode", ["original", "fixed"], index=0,
                                help="Use original gene means from data or fix all positive genes to the same mean.")
mean_expression = (
    st.sidebar.number_input("Mean expression (fixed)", 0.0, 1e9, 10.0, step=0.5,
                           help="Mean expression used for simulating gene expression when 'Gene mean mode' is set to 'fixed'.",)
    if mean_mode == "fixed" else None
)

lfc_mode = st.sidebar.selectbox("LFC mode", ["original", "fixed", "normal"], index=0,
                               help="How to assign log-fold changes: fixed value, drawn from a normal, or taken from the original data.")
lfc_value = st.sidebar.number_input("LFC value (fixed mode)", -100.0, 100.0, -0.5, step=0.05,
                                   help="Log-fold change used for all replicates when LFC mode is 'fixed'.")
lfc_normal_mean = st.sidebar.number_input("LFC normal mean", -100.0, 100.0, -0.5, step=0.05,
                                         help="Mean of the normal distribution for all replicates when LFC mode is 'normal'.")
lfc_normal_sd = st.sidebar.number_input("LFC normal sd", 0.0, 10.0, 1.0, step=0.05,
                                       help="Standard deviation of the normal distribution for all replicates when LFC mode is 'normal'.")

if orig_data_name == "Gasperini (high MOI)":
    n_grna_per_element = st.sidebar.number_input("Guides per element", 1, 32, 4, step=1,
                                                help="Number of gRNAs per element in the simulation.")
elif orig_data_name == "Weissman (low MOI)":
    n_grna_per_element = st.sidebar.number_input("Guides per element", 1, 32, 1, step=1,
                                                help="Number of gRNAs per element in the simulation.")
guide_eff_mode = st.sidebar.selectbox("Guide efficacy mode", ["beta", "fixed_list", "original"], index=0,
                                     help="Distribution of guide efficacies: Beta prior, user-provided fixed list, or original estimates.")
def uniform_split_str(n: int, decimals: int = 2) -> str:
    if n <= 1:
        vals = [1.0]
    else:
        vals = [round(1 - i/(n-1), decimals) for i in range(n)]  # 1..0 inclusive
    return ",".join(str(v) for v in vals)

# Track prior inputs to decide when to (re)prefill the field
if "prev_n" not in st.session_state:
    st.session_state.prev_n = n_grna_per_element
if "prev_mode" not in st.session_state:
    st.session_state.prev_mode = guide_eff_mode
if "ge_list_str" not in st.session_state:
    st.session_state.ge_list_str = uniform_split_str(n_grna_per_element)

# Only (re)seed when the user selects "fixed_list" AND n/mode changed.
if guide_eff_mode == "fixed_list":
    if (st.session_state.prev_mode != guide_eff_mode) or (st.session_state.prev_n != n_grna_per_element):
        st.session_state.ge_list_str = uniform_split_str(n_grna_per_element)
        st.session_state.prev_n = n_grna_per_element
        st.session_state.prev_mode = guide_eff_mode

    ge_list_str = st.sidebar.text_input(
        "Guide efficacy list (comma-separated)",
        key="ge_list_str",
        help="Comma-separated guide efficacies between 0 and 1, one value per guide (e.g. 1,0.67,0.33,0)."
    )
    ge_fixed_list = [float(x) for x in ge_list_str.split(",")] if ge_list_str.strip() else None
    ge_a = ge_b = None

#if guide_eff_mode == "fixed_list":
#    ge_list_str = st.sidebar.text_input("Guide efficacy list (comma-separated)", value="1,0.67,0.33,0.0")
#    ge_fixed_list = [float(x) for x in ge_list_str.split(",")] if ge_list_str.strip() else None
#    ge_a = ge_b = None
elif guide_eff_mode == "beta":
    ge_a = st.sidebar.number_input("Guide efficacy Beta a", 0.1, 100.0, 2.5, step=0.1,
                                  help="Alpha parameter (a) of the Beta prior for guide efficacies.")
    ge_b = st.sidebar.number_input("Guide efficacy Beta b", 0.1, 100.0, 1.5, step=0.1,
                                  help="Beta parameter (b) of the Beta prior for guide efficacies.")
    ge_fixed_list = None
else:
    ge_fixed_list = None
    ge_a = ge_b = None

if "low" in orig_data_name:
    moi = st.sidebar.number_input("MOI (used to scale total cells)", 1.0, 1e5, 1.0, step=1.0,
                                 help="Multiplicity of infection; for low MOI, typically 1.")
else:
    moi = st.sidebar.number_input("MOI (used to scale total cells)", 1.0, 1e5, 30.0, step=1.0,
                                 help="Multiplicity of infection; for low MOI, typically 1.")
read_depth_adjust_factor = st.sidebar.number_input("Read-depth scaling", 0.0, 1000.0, 1.0, step=0.1,
                                                  help="Scaling factor applied to read depth relative to the original data (1.0 = same depth).")
random_seed = st.sidebar.number_input("Random seed", 0, 10_000_000, 57, step=1,
                                     help="Seed for all random number generators to make simulations reproducible.")

st.sidebar.header("🧪 Testing")
alpha = st.sidebar.number_input("Alpha (test size)", 0.0, 1.0, 0.1, step=0.01,
                               help="Significance level for hypothesis tests (per test).")
mt_method = st.sidebar.selectbox("Multiple testing", ["fdr_bh", None], index=0,
                                help="Multiple testing correction method for p-values.")
test_type = st.sidebar.selectbox("p-value type", ["empirical", "fixed"], index=0,
                                help="Use empirical p-values (e.g. from negative controls) or a fixed theoretical test.")
max_epochs = st.sidebar.number_input("Max epochs", 1, 10_000, 200, step=10,
                                    help="Maximum number of training epochs per simulation run.")
lr = st.sidebar.number_input("Learning rate", 1e-5, 1.0, 0.01, step=0.001, format="%.5f",
                            help="Learning rate for model optimization.")
if orig_data_name == "Gasperini (high MOI)":
    batch_size = st.sidebar.number_input("Batch size", 8, 200_000, 1024, step=8,
                                        help="Mini-batch size used during training.")
    early_stopping_patience = st.sidebar.number_input("ES Patience", 1, 1000, 10, step=1,
                                                     help="Early stopping patience: number of epochs without improvement before stopping.")
elif orig_data_name == "Weissman (low MOI)":
    batch_size = st.sidebar.number_input("Batch size", 8, 200_000, 4096, step=8,
                                        help="Mini-batch size used during training.")
    early_stopping_patience = st.sidebar.number_input("ES Patience", 1, 1000, 5, step=1,
                                                     help="Early stopping patience: number of epochs without improvement before stopping.")
early_stopping_min_delta = st.sidebar.number_input("ES Min Delta", 0.0, 1.0, 0.001, step=0.001, format="%.4f",
                                                  help="Minimum improvement in monitored metric required to reset early stopping patience.")

# ------------------------------
# Main: Sweep selector & params
# ------------------------------
left, right = st.columns([1, 2], gap="large")
with left:
    st.subheader("📈 Sweep")
    mode = st.selectbox("X-axis", ["Cells per element", "LFC", "Guides per element", "MOI", "Gene mean (approx)"],
                       help="Choose which parameter to sweep along the x-axis for the power curve.")

    fixed_cells = st.number_input("Fixed cells/element (for LFC/Guides/MOI/GeneMean)", 10, 1_000_000, 200, step=10,
                                 help="Number of cells per element used when sweeping LFC/Guides/MOI/Gene mean.")
    # min_val = st.number_input("Min", value=50.0,
    #                          help="Lower bound of the sweep range for the selected x-axis parameter.")
    # max_val = st.number_input("Max", value=500.0,
    #                          help="Upper bound of the sweep range for the selected x-axis parameter.",)
    if mode == "LFC":
        min_val = st.number_input("Min", value=-1.0, step=0.1, format="%.3f", key="min_lfc")
        max_val = st.number_input("Max", value=-0.2, step=0.1, format="%.3f", key="max_lfc")

    elif mode == "MOI":
        min_val = st.number_input("Min", value=1, step=1, key="min_moi")
        max_val = st.number_input("Max", value=30, step=1, key="max_moi")

    elif mode == "Gene mean (approx)":
        min_val = st.number_input("Min", value=0.1, step=0.1, format="%.3f", key="min_gmean")
        max_val = st.number_input("Max", value=5.0, step=0.1, format="%.3f", key="max_gmean")

    elif mode == "Guides per element":
        min_val = st.number_input("Min", min_value=1, value=1, step=1, key="min_guides")
        max_val = st.number_input("Max", min_value=1, value=5, step=1, key="max_guides")

    else:  # "Cells per element"
        min_val = st.number_input("Min", min_value=1, value=50, step=10, key="min_cells")
        max_val = st.number_input("Max", min_value=1, value=500, step=10, key="max_cells")

    n_bins = st.number_input("Bins (or points)", 2, 200, 5, step=1,
                            help="Number of points between Min and Max for the sweep.")
    step = st.number_input("Step (Guides mode only)", 1, 100, 1, step=1,
                          help="Step size in guides per element when the x-axis is 'Guides per element'.")

    st.caption(
        "⚠️ Running the sweep trains a PerTurbo model for each point and can take a few minutes "
        "depending on the number of bins, cells, and training settings."
    )
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
    #model_dir=model_dir,
    #real_data_path=real_path,
    orig_data_name = orig_data_name,
    accelerator="cpu",
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

#if run_btn:
#    if not os.path.exists(model_dir):
#        st.error("Model directory not found. Point to the folder that contains a saved PerTurbo model (subdir 'model').")
#        st.stop()
#    if not os.path.exists(real_path):
#        st.error("Real data path not found. Provide a file path to 'mdata.h5mu' or a directory that contains it.")
#        st.stop()

df = None
if run_btn:
    try:
        with st.status("Running sweep… this can be compute-intensive (training involved).", expanded=False) as status:
            print("ready to run")
            #abort = lambda: st.session_state.get("abort", False)
            abort = False
            
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
