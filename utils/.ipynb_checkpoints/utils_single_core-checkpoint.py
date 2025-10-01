# utils_core.py
"""
Core utilities for PerTurbo power experiments:
- configs
- loading real data & model
- parameter samplers / structures
- simulation from a trained model (chunked, robust MuData handling)
- training + detailed output table
- empirical MT + power summary
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Dict, Any, Tuple, List
from pathlib import Path

import numpy as np
import pandas as pd
import warnings

# external deps
import pyro
import torch
import mudata as md
import anndata as ad
from scipy import stats
import scipy.sparse as sp
from scipy.sparse import random as sparse_random
import matplotlib.pyplot as plt

# scipy ECDF (SciPy >=1.11); fallback implemented below
try:
    from scipy.stats import ecdf as _scipy_ecdf
except Exception:
    _scipy_ecdf = None
    
torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])

# ---- PerTurbo import ----
# Assumes perturbo is importable. If not, uncomment and customize:
# import sys
# sys.path.insert(0, r"/path/to/PerTurbo/src")
import perturbo  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# -----------------
# Config dataclasses
# -----------------

@dataclass
class SimulationConfig:
    # data & model
    model_dir: str              # path containing saved PerTurbo model (subdir "model")
    real_data_ref_path: str         # path to .npz (file or directory containing it)
    accelerator: str = "cpu"    # 'cpu' or 'gpu'

    # experiment setup
    gene_name: str = "GATA1"
    n_genes: int = 500
    mean_mode: str = "original"    # 'original' or 'fixed'
    mean_expression: Optional[float] = None

    lfc_mode: str = "fixed"     # 'fixed', 'normal', 'original'
    lfc_value: float = 0.5
    lfc_normal_mean: float = 0.5
    lfc_normal_sd: float = 0.2

    guide_eff_mode: str = "beta"  # 'fixed_list', 'beta', 'original'
    guide_eff_fixed_list: Optional[Sequence[float]] = None  # values in [0,1]
    guide_eff_beta_a: float = 2.0
    guide_eff_beta_b: float = 5.0

    n_grna_per_element: int = 4
    moi: float = 30
    read_depth_adjust_factor: float = 1.0

    random_seed: int = 57


@dataclass
class TestConfig:
    alpha: float = 0.1
    mt_method: str = "FDR"          # 'none' or 'FDR'
    test_type: str = "empirical"    # 'fixed' or 'empirical'
    max_epochs: int = 500
    lr: float = 0.01
    batch_size: int = 2048
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-3
    max_steps: Optional[int] = None
    devices: Optional[int] = None


# -----------------
# Load resources
# -----------------

def make_minimal_adata(gene_names):
    '''Create pseudo anndata s.t. loading the model do not have errors.'''

    # zero cells, same variables (genes)
    X = sp.csr_matrix((0, len(gene_names)))  # shape (0, n_genes)
    var = pd.DataFrame(index=pd.Index(gene_names, name="gene_name"))  # name optional
    obs = pd.DataFrame(index=pd.Index([], name="cell"))

    adata = ad.AnnData(X=X, obs=obs, var=var)

    # Minimal PerTurbo/scvi setup; if your training used custom keys,
    perturbo.PERTURBO.setup_anndata(adata)  # defaults are fine if you used defaults
    return adata


def load_resources(cfg: SimulationConfig) -> Tuple[Any, md.MuData, Dict[str, Any]]:
    """Load trained PerTurbo model + Parameter extracted from real MuData (.npz) and build reference_stats."""
    print("Start Loading Model and MuData_real.")
    pyro.clear_param_store()

    real_path = Path(cfg.real_data_ref_path)
    candidate = real_path / "reference_stats_compact.npz" if real_path.is_dir() else real_path
    if not candidate.exists() or not candidate.is_file():
        raise FileNotFoundError(
            f"Could not find file. Checked: {candidate}\n"
            "Hint: pass full file path or a directory that contains the '.npz' file."
        )
    ref_real = np.load(str(candidate, allow_pickle=True))
    
    gene_names = ref_real["gene_names"]
    adata_min = make_minimal_adata(gene_names)
    
    model_dir = Path(cfg.model_dir)
    model = perturbo.PERTURBO.load(model_dir / "model", adata=adata_min)

    reference_stats: Dict[str, Any] = {}
    # per-gene means
    if "gene_name" in ref_real and "_gene_mean" in ref_real:
        gn = ref_real["gene_name"]
        gm = ref_real["_gene_mean"]
        reference_stats["gene_means"] = pd.Series(gm.values, index=gn.values)
    # guide-efficacy distribution from trained model (if present)
    try:
        eff = model.module.guide.median()["guide_efficacy"].cpu().numpy().reshape(-1)
        reference_stats["guide_eff_samples"] = eff
    except Exception:
        pass
    # optional empirical LFC samples
    try:
        lfc_mat = ref["lfc_mat"]
        if lfc_mat is not None:
            reference_stats["lfc_samples"] = np.asarray(lfc_mat).ravel()
    except Exception:
        pass

    print("Finish Loading Model and MuData_real.")
    return model, ref_real, reference_stats


# -----------------
# Parameter helpers
# -----------------

def _sample_lfc(cfg: SimulationConfig, size: int, reference_stats: Dict[str, Any]) -> np.ndarray:
    if cfg.lfc_mode == "fixed":
        return np.full(size, cfg.lfc_value, dtype=float)
    if cfg.lfc_mode == "normal":
        return np.random.normal(cfg.lfc_normal_mean, cfg.lfc_normal_sd, size=size)
    if cfg.lfc_mode == "original":
        arr = reference_stats.get("lfc_samples", None)
        if arr is not None and len(arr) > 0:
            return np.random.choice(np.asarray(arr, dtype=float), size=size, replace=True)
    # fallback
    return np.random.normal(cfg.lfc_normal_mean, cfg.lfc_normal_sd, size=size)


def _sample_guide_efficacy(n_elements: int, cfg: SimulationConfig, reference_stats: Dict[str, Any]) -> np.ndarray:
    if cfg.guide_eff_mode == "fixed_list":
        base = np.array([1.0, 2/3, 1/3, 0.0]) if not cfg.guide_eff_fixed_list else np.array(cfg.guide_eff_fixed_list, dtype=float)
        if len(base) != cfg.n_grna_per_element:
            eff = np.resize(base, n_elements * cfg.n_grna_per_element)
        else:
            eff = np.tile(base, n_elements)
        return np.clip(eff, 0, 1)
    if cfg.guide_eff_mode == "beta":
        return stats.beta(cfg.guide_eff_beta_a, cfg.guide_eff_beta_b).rvs(size=n_elements * cfg.n_grna_per_element)
    # 'original' from model posterior if present
    arr = reference_stats.get("guide_eff_samples", None)
    if arr is not None and len(arr) > 0:
        return np.random.choice(np.asarray(arr, dtype=float), size=n_elements * cfg.n_grna_per_element, replace=True)
    # fallback
    return stats.beta(cfg.guide_eff_beta_a, cfg.guide_eff_beta_b).rvs(size=n_elements * cfg.n_grna_per_element)


# -----------------
# Structures for simulation
# -----------------

def _build_element_gene_map(n_elements_pos: int, n_genes: int, cfg: SimulationConfig):
    """Binary [n_elements_pos x n_genes] matrix selecting the gene each positive element targets."""
    element_gene_map = np.zeros((n_elements_pos, n_genes), dtype=int)
    affected_gene_idx = np.random.choice(n_genes, size=n_elements_pos, replace=False)
    element_gene_map[np.arange(n_elements_pos), affected_gene_idx] = 1
    return element_gene_map


def _build_element_by_gene_lfc(element_gene_map: np.ndarray,
                               n_elements: int, n_elements_pos: int,
                               n_elements_ntc: int, n_genes: int,
                               cfg: SimulationConfig, reference_stats: Dict[str, Any]) -> np.ndarray:
    """Stack positive LFC rows with zeros for NTC rows -> [n_elements x n_genes]."""
    lfc = _sample_lfc(cfg, size=n_elements_pos, reference_stats=reference_stats)
    element_by_gene_lfc_pos = lfc * element_gene_map
    element_by_gene_lfc_ntc = np.zeros((n_elements_ntc, n_genes), dtype=float)
    return np.vstack((element_by_gene_lfc_pos, element_by_gene_lfc_ntc))


def _build_tested_elements(element_gene_map: np.ndarray,
                           n_elements_pos: int, n_elements_ntc: int, n_genes: int, cfg: SimulationConfig):
    """Which element-gene pairs are tested: positives + a matched number of NTC pairs."""
    tested_elements_pos = element_gene_map
    n_ntc_pairs = n_elements_pos  # balance positives
    test_rate = n_ntc_pairs / (n_elements_ntc * n_genes)
    tested_elements_ntc = np.random.binomial(1, test_rate, size=(n_elements_ntc, n_genes)).astype(np.float32)
    tested_elements = np.vstack((tested_elements_pos, tested_elements_ntc))
    return tested_elements


def _hierarchical_concat(chunk_list: List[md.MuData]) -> md.MuData:
    """Concatenate MuData objects progressively in pairs to avoid slowdowns; assert consistent modalities."""
    assert all(("rna" in ch.mod and "grna" in ch.mod) for ch in chunk_list), "Each chunk must have 'rna' and 'grna'."
    while len(chunk_list) > 1:
        new_chunk_list = []
        for i in range(0, len(chunk_list), 2):
            if i + 1 < len(chunk_list):
                merged = md.concat([chunk_list[i], chunk_list[i + 1]])
                new_chunk_list.append(merged)
            else:
                new_chunk_list.append(chunk_list[i])
        chunk_list = new_chunk_list
        print(f"Intermediate concatenation: {len(chunk_list)} chunks remaining")
    return chunk_list[0]


# -----------------
# Simulate from trained model (chunked)
# -----------------

def simulate_mudata_from_model(model, ref_real: md.MuData,
                               n_cells_per_element: int,
                               element_gene_map: np.ndarray,
                               element_by_gene_lfc: np.ndarray,
                               guide_efficacy: np.ndarray,
                               new_genes_idx: np.ndarray,
                               cfg: SimulationConfig,
                               reference_stats: Dict[str, Any],
                               chunk_size: int = 100000) -> md.MuData:
    """
    Use perturbo.simulation.simulate_data_from_trained_model to generate MuData.
    Preserves modalities 'rna' and 'grna'; de-duplicates obs & var names; registers with PERTURBO.
    """
    print("Start Simulating MuData from Model.")
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
    torch.cuda.manual_seed_all(cfg.random_seed)
    pyro.set_rng_seed(cfg.random_seed)

    # sizes
    n_elements_pos = cfg.n_genes
    n_elements_ntc = max(1, round(0.05 * n_elements_pos))
    n_elements = n_elements_pos + n_elements_ntc
    n_grna_per_element = cfg.n_grna_per_element
    n_grna = n_elements * n_grna_per_element

    # gRNA assignment
    n_cells_per_guide = n_cells_per_element / n_grna_per_element
    n_cells = int(n_cells_per_element * n_elements // cfg.moi)
    n_cells = max(n_cells, 1)
    pert_rate = n_cells_per_guide / n_cells

    grna_counts = sparse_random(
        n_cells, n_grna, density=pert_rate, format="csr", dtype=np.float32, random_state=np.random
    )
    grna_counts.data[:] = 1.0

    # guide->element map
    guide_by_element = np.zeros((n_grna, n_elements), dtype=np.float32)
    for j in range(n_elements):
        start_row = n_grna_per_element * j
        end_row = min(start_row + n_grna_per_element, n_grna)
        guide_by_element[start_row:end_row, j] = 1.0

    # chunked simulation
    num_cells = n_cells
    num_chunks = int(np.ceil(num_cells / chunk_size))
    simulated_chunks: List[md.MuData] = []
    existing_obs_names = set()
    print(f"num_cells: {num_cells}")

    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, num_cells)
        grna_counts_chunk = grna_counts[start:end].toarray()

        # borrow covariates from real data
        n_cells_origin = int(ref_real.get("n_cells_origin", 0))
        chunk_indices = np.random.choice(n_cells_origin, size=grna_counts_chunk.shape[0], replace=False)

        pyro.clear_param_store()
        mdata_chunk = perturbo.simulation.simulate_data_from_trained_model(
            model,
            guide_obs=grna_counts_chunk,
            cell_indices=chunk_indices,
            guide_by_element=guide_by_element,
            element_by_gene_lfc=element_by_gene_lfc,
            guide_efficacy=guide_efficacy,
            read_depth_adjust_factor=cfg.read_depth_adjust_factor,
            gene_indices=new_genes_idx,
            module_init_kwargs={"efficiency_mode": "mixture_high_moi"},
            accelerator=('cuda' if cfg.accelerator == 'gpu' else cfg.accelerator),
        )

        if not ("rna" in mdata_chunk.mod and "grna" in mdata_chunk.mod):
            raise KeyError("Simulated chunk missing 'rna' or 'grna' modality.")

        # unique obs names across chunks
        new_obs = []
        for obs_name in mdata_chunk.obs_names:
            name = obs_name
            while name in existing_obs_names:
                name = f"{name}_{i+1}"
            new_obs.append(name)
            existing_obs_names.add(name)

        mdata_chunk.obs.index = new_obs
        for modality in mdata_chunk.mod:
            mdata_chunk.mod[modality].obs.index = new_obs

        # make var names unique inside chunk (only rna)
        var_names_old = list(mdata_chunk["rna"].var_names)
        var_names_new = [f"{gene}_{i}" for i, gene in enumerate(var_names_old)]
        mdata_chunk["rna"].var.index = var_names_new

        # rebuild MuData explicitly with both modalities
        mdata_chunk = md.MuData({"rna": mdata_chunk["rna"], "grna": mdata_chunk["grna"]})
        simulated_chunks.append(mdata_chunk)
        #print(f"simulated chunk: {mdata_chunk}")

        torch.cuda.empty_cache()

    # concatenate
    mdata_simu = _hierarchical_concat(simulated_chunks)

    # attach varm & var from first chunk (structure); element_tested from our design
    tested_elements = _build_tested_elements(element_gene_map, n_elements_pos, n_elements_ntc, cfg.n_genes, cfg)
    mdata_simu["rna"].varm["element_tested"] = tested_elements.T
    mdata_simu["rna"].varm["lfc"] = simulated_chunks[0].mod["rna"].varm["lfc"]
    mdata_simu["rna"].var = simulated_chunks[0].mod["rna"].var
    mdata_simu["grna"].varm["element_targeted"] = simulated_chunks[0].mod["grna"].varm["element_targeted"]

    # filter cells with zero RNA counts
    sel = np.asarray(mdata_simu["rna"].X.sum(axis=1)).ravel() > 0
    mdata_filtered = mdata_simu[sel, :].copy()
    n_selected_cells = mdata_filtered["rna"].X.shape[0]
    print(f"Preparing data with {n_selected_cells} cells among {n_cells} cells.")
    print(mdata_filtered)

    # z-score covariates (if present)
    obs = mdata_filtered.mod['rna'].obs.copy()
    for col in ("percent_mito", "log1p_guide_count"):
        if col in obs.columns:
            vals = obs[col].values
            sd = vals.std() if vals.std() != 0 else 1.0
            obs[f"{col}_z"] = (vals - vals.mean()) / sd
    mdata_filtered.mod['rna'].obs = obs

    # register with perturbo
    perturbo.PERTURBO.setup_mudata(
        mdata_filtered,
        batch_key="prep_batch",
        continuous_covariates_keys=[c for c in ["log1p_guide_count_z", "percent_mito_z"] if c in obs.columns],
        gene_by_element_key='element_tested',
        guide_by_element_key="element_targeted",
        modalities={"rna_layer": "rna", "perturbation_layer": "grna"},
    )

    return mdata_filtered


# -----------------
# Training + detailed output
# -----------------

def _create_empty_list(method: str) -> Dict[str, list]:
    d = {
        "Gene_id": [], "Element_id": [], "Gene_index": [], "Gene_Name": [],
        "Gene_Mean": [], "NCells": [], "NGuides": [], "NGuidesPerElement": [], "NCellsPerElement": [],
        "MOI": [], "LogFoldChange": [], "MeanReads": [], "ReadScaling": [],
        "LFC_hat": [], "P_value": [], "alpha_cor": [], "Efficacy_type": [],
        "Method": [], "MTmethod": [], "TrueLabel": []
    }
    if method == "wilcoxon":
        del d["LFC_hat"]
    return d


def _update_detailed_output(
    list_dict: Dict[str, list],
    element_effects: pd.DataFrame,
    method: str,
    gene_indices: List[int],
    observed_gene_names: str,
    gene_mean: pd.Series,
    ncells: int,
    nguides: int,
    nguides_per_element: int,
    n_cells_per_element: int,
    moi: float,
    lfc: np.ndarray,
    mean_reads_per_gene: float,
    read_scaling: float,
    alpha_base: float,
    guide_efficacy_type: str,
    MTmethod: str,
) -> Dict[str, list]:
    # model_eval.get_element_effects() columns: loc, scale, element, gene, z_value, q_value
    list_dict["Gene_id"].extend(element_effects["gene"])
    list_dict["Element_id"].extend(element_effects["element"])

    list_dict["Gene_index"].extend(gene_indices)
    list_dict["Gene_Mean"].extend(gene_mean)

    npairs = len(element_effects["gene"])
    list_dict["NCells"].extend([ncells] * npairs)
    list_dict["NGuides"].extend([nguides] * npairs)
    list_dict["NGuidesPerElement"].extend([nguides_per_element] * npairs)
    list_dict["Gene_Name"].extend([observed_gene_names] * npairs)
    list_dict["MOI"].extend([moi] * npairs)
    list_dict["NCellsPerElement"].extend([n_cells_per_element] * npairs)
    #list_dict["LogFoldChange"].extend(lfc)
    list_dict["MeanReads"].extend([mean_reads_per_gene] * npairs)
    list_dict["ReadScaling"].extend([read_scaling] * npairs)

    if method in {"perturbo", "glm"}:
        LFC_hats = [x * np.log2(np.e) if x is not None else None for x in element_effects["loc"]]
        list_dict["LFC_hat"].extend(LFC_hats)
    elif method == "sceptre":
        list_dict["LFC_hat"].extend([x for x in element_effects["loc"]])

    # treat q_value as the base p-value column as per your request
    list_dict["P_value"].extend(element_effects["q_value"])
    list_dict["alpha_cor"].extend([alpha_base] * npairs)
    list_dict["Efficacy_type"].extend([guide_efficacy_type] * npairs)
    list_dict["Method"].extend([method] * npairs)
    list_dict["MTmethod"].extend([MTmethod] * npairs)

    true_label = np.where(lfc == 0, "ntc", "cis").tolist()
    list_dict["TrueLabel"].extend(true_label)
    
    ## reset LogFoldChange for "ntc"
    nz = []
    for x in lfc:
        try:
            fx = float(x); 
            if np.isfinite(fx) and fx != 0.0: nz.append(fx)
        except (TypeError, ValueError): 
            pass
    u = np.unique(np.round(nz, 12))
    lfc = [float(u[0])]*npairs if u.size == 1 else lfc
    #print(f"modified lfc {lfc}")
    list_dict["LogFoldChange"].extend(lfc)

        
    return list_dict


def _train_and_get_effects(mdata_sim: md.MuData, accelerator: str,
                           cfg: SimulationConfig, tcfg: TestConfig,
                           n_cells_per_element: int, list_dict: Dict[str, list]) -> Dict[str, list]:
    print("Start training.")
    accel = "gpu" if torch.cuda.is_available() else "cpu"
    print(f"training using {accel.upper()}")

    model_eval = perturbo.PERTURBO(
        mdata_sim,
        likelihood="nb",
        efficiency_mode="scaled",
        effect_prior_dist="normal",
        fit_guide_efficacy=False,
    )
    pyro.clear_param_store()
    model_eval.train(
        max_epochs=int(tcfg.max_epochs),
        lr=float(tcfg.lr),
        batch_size=int(tcfg.batch_size),
        accelerator=accel,
        early_stopping=True,
        early_stopping_patience=int(tcfg.early_stopping_patience),
        early_stopping_min_delta=float(tcfg.early_stopping_min_delta),
        early_stopping_monitor="elbo_train",
    )
    element_effects = model_eval.get_element_effects().sort_index()

    # bookkeeping for output
    gene_names_tested = element_effects["gene"]
    element_names_tested = element_effects["element"]
    gene_names_all = mdata_sim["rna"].var_names

    # mean per gene (1D)
    gene_mean_1d = np.asarray(mdata_sim["rna"].X.mean(axis=0)).ravel()
    simu_mean = pd.DataFrame(gene_mean_1d, index=mdata_sim["rna"].var_names).loc[gene_names_tested][0]
    gene_indices = [gene_names_all.get_loc(g) for g in gene_names_tested]

    element_gene_lfc = mdata_sim["rna"].varm["lfc"].T
    lfc = element_gene_lfc[element_names_tested, gene_indices]

    # recompute sizes to log in table
    n_elements_pos = cfg.n_genes
    n_elements_ntc = max(1, round(0.05 * n_elements_pos))
    n_elements = n_elements_pos + n_elements_ntc
    n_grna = n_elements * cfg.n_grna_per_element
    n_cells = int(n_cells_per_element * n_elements // cfg.moi)
    n_cells = max(n_cells, 1)
    mean_reads_per_gene = float(gene_mean_1d.mean())

    list_dict = _update_detailed_output(
        list_dict=list_dict,
        element_effects=element_effects,
        method="perturbo",
        gene_indices=gene_indices,
        observed_gene_names=cfg.gene_name,
        gene_mean=simu_mean,
        ncells=n_cells,
        nguides=n_grna,
        nguides_per_element = cfg.n_grna_per_element,
        n_cells_per_element=n_cells_per_element,
        moi=cfg.moi,
        lfc=lfc,
        mean_reads_per_gene=mean_reads_per_gene,
        read_scaling=cfg.read_depth_adjust_factor,
        alpha_base=tcfg.alpha,
        guide_efficacy_type=cfg.guide_eff_mode,
        MTmethod=tcfg.mt_method,
    )
    return list_dict


# -----------------
# Empirical MT + Power summary
# -----------------

def _ecdf_values(x: np.ndarray, sample: np.ndarray) -> np.ndarray:
    """Return ECDF values for 'sample' evaluated at x."""
    if _scipy_ecdf is not None:
        e = _scipy_ecdf(sample)
        return e.cdf.evaluate(x)
    # fallback
    s = np.sort(sample)
    return np.searchsorted(s, x, side="right") / (len(s) + 1.0)


def _empirical_multipletesting_correction(
    detail_df: pd.DataFrame,
    test_type: str,
    MTmethod: str,
    alpha_base: float,
    grouping_columns: Sequence[str],
) -> pd.DataFrame:
    if test_type == "empirical":
        pos = detail_df[detail_df["TrueLabel"] == "cis"].copy()
        neg = detail_df[detail_df["TrueLabel"] == "ntc"].copy()

        for key, pos_grp in pos.groupby(grouping_columns):
            mask_neg = (neg[grouping_columns].apply(tuple, axis=1) == tuple(key))
            neg_grp = neg.loc[mask_neg]
            if len(neg_grp) == 0:
                # fallback: use raw P_value
                detail_df.loc[pos_grp.index, "P_value_empi"] = pos_grp["P_value"].values
                continue
            emp_pos = _ecdf_values(pos_grp["P_value"].values, neg_grp["P_value"].values)
            emp_neg = _ecdf_values(neg_grp["P_value"].values, neg_grp["P_value"].values)
            pos.loc[pos_grp.index, "P_value_empi"] = emp_pos
            neg.loc[neg_grp.index, "P_value_empi"] = emp_neg

        detail_df_empi = pd.concat([pos, neg], axis=0)
    else:
        detail_df_empi = detail_df.copy()
        detail_df_empi["P_value_empi"] = detail_df_empi["P_value"]

    if MTmethod == "FDR":
        from statsmodels.stats.multitest import multipletests
        detail_df_empi["P_value_cor"] = np.nan
        for _, grp in detail_df_empi.groupby(grouping_columns):
            idx = grp.index
            pvals = grp["P_value_empi"].dropna().values
            if len(pvals) == 0:
                continue
            _, pvals_corrected, _, _ = multipletests(pvals, alpha=alpha_base, method="fdr_bh")
            detail_df_empi.loc[idx, "P_value_cor"] = pvals_corrected
    else:
        detail_df_empi["P_value_cor"] = detail_df_empi["P_value_empi"]
    #print(f"Detailed DataFrame with eFDR corrected P_value: {detail_df_empi}")

    return detail_df_empi


def _detail_to_power_summary(list_dict: Dict[str, list],
                             grouping_columns: Sequence[str],
                             tcfg: TestConfig) -> pd.DataFrame:
    detail = pd.DataFrame(list_dict)

    # group-wise empirical correction
    corrected = _empirical_multipletesting_correction(
        detail_df=detail,
        test_type=tcfg.test_type,
        MTmethod=tcfg.mt_method,
        alpha_base=tcfg.alpha,
        grouping_columns=grouping_columns,
    )
    corrected["alpha_cor"] = tcfg.alpha

    # power over positives
    pos = corrected[corrected["TrueLabel"] == "cis"].copy()
    pos["significance"] = pos["P_value_cor"] <= tcfg.alpha
    power = (
        pos.groupby(grouping_columns)
           .agg(Power=("significance", "mean"),
                CatCount=("significance", "count"))
           .reset_index()
    )
    return power
