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
from scipy import sparse as sp
from scipy.sparse import random as sparse_random
from scipy.sparse import csr_matrix, vstack
from scipy.stats import nbinom

import matplotlib.pyplot as plt

# scipy ECDF (SciPy >=1.11); fallback implemented below
try:
    from scipy.stats import ecdf as _scipy_ecdf
except Exception:
    _scipy_ecdf = None

#from utils.load_nodata_model import load_perturbo_nodata

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
    #model_dir: str              # path containing saved PerTurbo model (subdir "model")
    #real_data_path: str         # path to mdata.h5mu (file or directory containing it)
    orig_data_name: str = "Gasperini (high MOI)"
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

def load_resources(cfg: SimulationConfig) -> Tuple[Any, np.lib.npyio.NpzFile, Dict[str, Any]]:
    """Load trained PerTurbo model + real MuData (mdata.h5mu) and build reference_stats."""
    print("Start Loading Model and MuData_real.")
    pyro.clear_param_store()
    
    # Read differnet datasets
    if cfg.orig_data_name == "Gasperini (high MOI)": 
        model_dir = Path("../save_model_gasperini/model")
        mdata_tiny = md.read_h5mu(f'{model_dir}/mdata_tiny.h5mu')
        print(f"Model is loaded from path: {model_dir}")
        model = perturbo.PERTURBO.load(model_dir, adata=mdata_tiny)

        real_path = Path("../save_model_gasperini/model/reference_stats_compact.npz")
        mdata_real = np.load(real_path, allow_pickle=True)
        print(f"mdata_real: {mdata_real}")
        
    elif cfg.orig_data_name == "Weissman (low MOI)": 
        model_dir = Path("../save_model_replogle/model")
        mdata_tiny = md.read_h5mu(f'{model_dir}/mdata_tiny.h5mu')
        print(f"Model is loaded from path: {model_dir}")
        model = perturbo.PERTURBO.load(model_dir, adata=mdata_tiny)

        #real_path = Path("../Weissman_ess/save_model/model_mixture/mdata.h5mu")
        real_path = Path("../save_model_replogle/model/reference_stats_compact.npz")
        mdata_real = np.load(real_path, allow_pickle=True)
        print(f"mdata_real: {mdata_real}")

    reference_stats: Dict[str, Any] = {}
    # per-gene means
    if "gene_name" in mdata_real and "_gene_mean" in mdata_real:
        print(f"mdata_real: {mdata_real}")
        gn = mdata_real["gene_name"]
        gm = mdata_real["_gene_mean"]
        print(f"gn: {gn}")
        print(f"gm: {gm}")
        reference_stats["gene_means"] = pd.Series(gm, index=gn)
    # guide-efficacy distribution from trained model (if present)
    try:
        eff = model.module.guide.median()["guide_efficacy"].cpu().numpy().reshape(-1)
        reference_stats["guide_eff_samples"] = eff
    except Exception:
        pass
    # optional empirical LFC samples
    try:
        lfc_mat = mdata_real["rna"].varm.get("lfc", None)
        if lfc_mat is not None:
            reference_stats["lfc_samples"] = np.asarray(lfc_mat).ravel()
    except Exception:
        pass

    print("Finish Loading Model and MuData_real.")
    return model, mdata_real, reference_stats


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
    element_by_gene_lfc = np.vstack((element_by_gene_lfc_pos, element_by_gene_lfc_ntc))
    return element_by_gene_lfc


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

def simulate_mudata_from_model(model, mdata_real: np.lib.npyio.NpzFile,
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
    n_grna_ntc = n_elements_ntc * n_grna_per_element
    n_grna_pos = n_elements_pos * n_grna_per_element
    n_grna = n_grna_ntc + n_grna_pos

    # gRNA assignment
    n_cells_per_guide = n_cells_per_element / n_grna_per_element
    n_cells = int(n_cells_per_element * n_elements // cfg.moi)
    n_cells = max(n_cells, 1)
    
    n_cells_origin = mdata_real["n_cells_origin"]
    subset_indices=np.random.choice(n_cells_origin, size=n_cells, replace=False)
    pert_rate = n_cells_per_guide / n_cells

    if cfg.moi>1:
        grna_counts = sparse_random(
            n_cells, n_grna, density=pert_rate, format="csr", dtype=np.float32, random_state=np.random
        )
        grna_counts.data[:] = 1.0
    elif cfg.moi==1:
        n_cells_per_guide_list = np.full(n_grna, int(n_cells_per_guide), dtype=int)
        
        num_batches = n_cells // chunk_size + (n_cells % chunk_size != 0)

        rows, cols = [], []
        perm = np.random.permutation(n_cells)
        ptr  = 0
        for g, k in enumerate(n_cells_per_guide_list):
            if ptr >= n_cells:
                break                      # no cells left
            take = min(k, n_cells - ptr)   # clip the remainder
            rows.append(perm[ptr:ptr+take])
            cols.append(np.full(take, g, dtype=np.int32))
            ptr += take
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        data = np.ones_like(rows, dtype=np.float32)
        grna_counts = csr_matrix((data, (rows, cols)), shape=(n_cells, n_grna), dtype=np.float32)



    # guide->element map
    guide_by_element = np.zeros((n_grna, n_elements), dtype=np.float32)
    for j in range(n_elements):
        start_row = n_grna_per_element * j
        end_row = min(start_row + n_grna_per_element, n_grna)
        if end_row > n_grna:
            break  # break the loop if the end_row exceeds the matrix size
        guide_by_element[start_row:end_row, j] = 1.0

    # chunked simulation
    num_cells = len(subset_indices)
    num_chunks = int(np.ceil(num_cells / chunk_size))
    simulated_chunks: List[md.MuData] = []
    existing_obs_names = set()
    print(f"num_cells: {num_cells}")

    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, num_cells)
        grna_counts_chunk = grna_counts[start:end].toarray()

        # borrow covariates from real data
        chunk_indices = subset_indices[start:end]

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
            #accelerator=('cuda' if cfg.accelerator == 'gpu' else cfg.accelerator),
            accelerator = 'cpu'
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
    if cfg.orig_data_name == "Gasperini (high MOI)":
        batch_key="prep_batch"
        library_size_key="umi_count"
        gene_by_element_key='element_tested'
        guide_by_element_key="element_targeted"
    elif cfg.orig_data_name == "Weissman (low MOI)":
        batch_key="gem_group"
        library_size_key="UMI_count"
        gene_by_element_key='gene_by_element'
        guide_by_element_key="guide_by_element"
        
    tested_elements = _build_tested_elements(element_gene_map, n_elements_pos, n_elements_ntc, cfg.n_genes, cfg)
    mdata_simu["rna"].varm[gene_by_element_key] = tested_elements.T
    mdata_simu["rna"].varm["lfc"] = simulated_chunks[0].mod["rna"].varm["lfc"]
    mdata_simu["rna"].var = simulated_chunks[0].mod["rna"].var
    mdata_simu["grna"].varm[guide_by_element_key] = simulated_chunks[0].mod["grna"].varm[guide_by_element_key]
    mdata_simu.update()

    # filter cells with zero RNA counts
    # sel = np.asarray(mdata_simu["rna"].X.sum(axis=1)).ravel() > 0
    # mdata_filtered = mdata_simu[sel, :].copy()
    rna_mod = mdata_simu["rna"]
    X = rna_mod.X
    n_obs = rna_mod.n_obs
    s0, s1 = X.shape

    def _to1d(a):
        return np.ravel(a.A) if sp.issparse(a) else np.ravel(a)

    # Figure out which axis is "cells" by matching n_obs
    if s0 == n_obs:
        sums = _to1d(X.sum(axis=1))
    elif s1 == n_obs:
        sums = _to1d(X.sum(axis=0))
    else:
        # Fallbacks
        sums = _to1d(X.sum(axis=1))
        if sums.size != n_obs:
            sums = _to1d(X.sum(axis=0))
        if sums.size != n_obs:
            sums = _to1d((X.T).sum(axis=1))

    mask_rna = sums > 0

    # Convert mask -> positions -> names, then intersect with the MuData top-level obs
    pos = np.flatnonzero(mask_rna)
    obs_keep_rna = rna_mod.obs.index[pos]

    # Extra safety: intersect with mdata_simu.obs index to avoid out-of-bounds
    obs_keep = mdata_simu.obs.index.intersection(obs_keep_rna)

    mdata_filtered = mdata_simu[obs_keep.tolist(), :].copy()

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
    if cfg.orig_data_name == "Gasperini (high MOI)":
        perturbo.PERTURBO.setup_mudata(
            mdata_filtered,
            batch_key=batch_key,
            continuous_covariates_keys=[c for c in ["log1p_guide_count_z", "percent_mito_z"] if c in obs.columns],
            gene_by_element_key=gene_by_element_key,
            guide_by_element_key=guide_by_element_key,
            modalities={"rna_layer": "rna", "perturbation_layer": "grna"},
        )
    elif cfg.orig_data_name == "Weissman (low MOI)":
        n_cells_more_than_one_guide = (mdata_filtered["grna"].X.sum(axis=1)>1).sum()
        print(f"There are {n_cells_more_than_one_guide} cells with more than 1 guide.")
        perturbo.PERTURBO.setup_mudata(
            mdata_filtered, 
            batch_key = batch_key,
            library_size_key = library_size_key,
            #gene_by_element_key=gene_by_element_key,
            guide_by_element_key = guide_by_element_key,
            modalities={
                "rna_layer": "rna",
                "perturbation_layer": "grna"},
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
    
    if cfg.moi == 1:
        efficiency_mode = "mixture"
    else:
        efficiency_mode = "scaled"
    if cfg.orig_data_name == "Gasperini (high MOI)":
        model_eval = perturbo.PERTURBO(
                mdata_sim,
                likelihood="nb",
                efficiency_mode=efficiency_mode,
                effect_prior_dist="normal",
                fit_guide_efficacy=True,
            )
    elif cfg.orig_data_name == "Weissman (low MOI)":
        model_eval = perturbo.PERTURBO(
                mdata_sim,
                likelihood="nb",
                efficiency_mode=efficiency_mode,
                #effect_prior_dist="normal",
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
                             cfg: SimulationConfig, tcfg: TestConfig
    ) -> pd.DataFrame:
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

def fit_nb_moments(x, eps=1e-8):
    # method-of-moments NB fit (SciPy's param: n=size=r, p=success prob)
    mu = np.mean(x)
    var = np.var(x, ddof=1)
    if var <= mu + eps:   # near/under-Poisson: fall back to Poisson-ish
        return np.inf, mu  # mark as Poisson with mean mu
    r = mu**2 / (var - mu)  # size/dispersion
    return r, mu

def sample_cells_per_guide(real_counts, mu_target, n_grna, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    r, mu_real = fit_nb_moments(real_counts)
    if np.isinf(r):  # Poisson fallback
        return rng.poisson(lam=mu_target, size=n_grna).astype(int)
    p_target = r / (r + mu_target)
    # SciPy: nbinom.rvs(n=r, p=p) returns number of failures before n successes
    return nbinom.rvs(r, p_target, size=n_grna, random_state=rng).astype(int)
