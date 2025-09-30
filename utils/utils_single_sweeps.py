# utils_sweeps.py
"""
Power-vs-X sweep functions that reuse the core utilities.
"""

from typing import Optional, Callable
from dataclasses import replace
import numpy as np
import pandas as pd

from .utils_single_core import (
    SimulationConfig, TestConfig, load_resources,
    _build_element_gene_map, _build_element_by_gene_lfc, _sample_guide_efficacy,
    simulate_mudata_from_model, _create_empty_list, _train_and_get_effects,
    _detail_to_power_summary
)


def _precompute_static_structures(model, mdata_real, cfg, reference_stats):
    """Compute pieces reused across sweeps."""
    gene_names_origin = list(mdata_real["rna"].var["gene_name"])
    if cfg.gene_name not in gene_names_origin:
        raise ValueError(f"Gene {cfg.gene_name} not found in real data.")
    new_genes_idx = np.array([gene_names_origin.index(cfg.gene_name)] * cfg.n_genes)

    n_pos = cfg.n_genes
    n_ntc = max(1, round(0.05 * n_pos))
    n_all = n_pos + n_ntc

    element_gene_map = _build_element_gene_map(n_elements_pos=n_pos, n_genes=cfg.n_genes, cfg=cfg)
    guide_efficacy = _sample_guide_efficacy(n_all, cfg, reference_stats)
    return new_genes_idx, element_gene_map, guide_efficacy


def power_vs_cells(cells_min, cells_max, n_bins, cfg: SimulationConfig, tcfg: TestConfig,
                   abort_flag: Optional[Callable[[], bool]] = None) -> pd.DataFrame:
    abort = abort_flag or (lambda: False)
    model, mdata_real, reference_stats = load_resources(cfg)
    new_genes_idx, element_gene_map, guide_efficacy = _precompute_static_structures(
        model, mdata_real, cfg, reference_stats
    )

    n_pos = cfg.n_genes
    n_ntc = max(1, round(0.05 * n_pos))
    n_all = n_pos + n_ntc
    element_by_gene_lfc = _build_element_by_gene_lfc(
        element_gene_map=element_gene_map,
        n_elements=n_all, n_elements_pos=n_pos, n_elements_ntc=n_ntc,
        n_genes=cfg.n_genes, cfg=cfg, reference_stats=reference_stats
    )

    xs = np.linspace(cells_min, cells_max, num=n_bins, dtype=int)
    list_dict = _create_empty_list(method="perturbo")

    for n_cpe in xs:
        print(f"Number of Cells per element: {n_cpe}")
        
        if abort():
            # Return whatever summary you’ve accumulated so far (partial results)
            return _detail_to_power_summary(list_dict, grouping_columns, tcfg)  # <-- this util already exists in your file

        mdata_sim = simulate_mudata_from_model(
            model, mdata_real, int(n_cpe),
            element_gene_map, element_by_gene_lfc,
            guide_efficacy, new_genes_idx,
            cfg, reference_stats
        )
        list_dict = _train_and_get_effects(mdata_sim, cfg.accelerator, cfg, tcfg, int(n_cpe), list_dict)

    grouping_columns = ['NCells', 'NGuides', 'NCellsPerElement', 'MOI',
                        'MeanReads', "ReadScaling", "alpha_cor",
                        'Efficacy_type', 'Method', 'MTmethod']
    power_summary = _detail_to_power_summary(list_dict, grouping_columns, tcfg)
    return power_summary


def power_vs_lfc(lfc_min: float, lfc_max: float, n_bins: int,
                 fixed_cells_per_element: int,
                 cfg: SimulationConfig, tcfg: TestConfig,
                 abort_flag: Optional[Callable[[], bool]] = None) -> pd.DataFrame:
    abort = abort_flag or (lambda: False)
    model, mdata_real, reference_stats = load_resources(cfg)
    new_genes_idx, element_gene_map, guide_efficacy = _precompute_static_structures(
        model, mdata_real, cfg, reference_stats
    )
    xs = np.linspace(lfc_min, lfc_max, num=n_bins)
    list_dict = _create_empty_list(method="perturbo")
    n_pos = cfg.n_genes
    n_ntc = max(1, round(0.05 * n_pos))
    n_all = n_pos + n_ntc

    for l in xs:
        cfg_l = replace(cfg, lfc_mode="fixed", lfc_value=float(l))
        
        if abort():
            # Return whatever summary you’ve accumulated so far (partial results)
            return _detail_to_power_summary(list_dict, grouping_columns, tcfg)  # <-- this util already exists in your file
        
        element_by_gene_lfc = _build_element_by_gene_lfc(
            element_gene_map=element_gene_map,
            n_elements=n_all, n_elements_pos=n_pos, n_elements_ntc=n_ntc,
            n_genes=cfg_l.n_genes, cfg=cfg_l, reference_stats=reference_stats
        )
        mdata_sim = simulate_mudata_from_model(
            model, mdata_real, int(fixed_cells_per_element),
            element_gene_map, element_by_gene_lfc,
            guide_efficacy, new_genes_idx,
            cfg_l, reference_stats
        )
        list_dict = _train_and_get_effects(mdata_sim, cfg_l.accelerator, cfg_l, tcfg,
                                           int(fixed_cells_per_element), list_dict)

    grouping_columns = ['NCells', 'NGuides', 'NCellsPerElement', 'MOI',
                        'LogFoldChange', 'MeanReads', "ReadScaling", "alpha_cor",
                        'Efficacy_type', 'Method', 'MTmethod']
    #print(f"Power vs LFC list_dict: {list_dict}")
    power_summary = _detail_to_power_summary(list_dict, grouping_columns, tcfg)
    return power_summary.rename(columns={"LogFoldChange": "LFC"})


def power_vs_nguides(n_guides_min: int, n_guides_max: int, step: int,
                     fixed_cells_per_element: int,
                     cfg: SimulationConfig, tcfg: TestConfig,
                     abort_flag: Optional[Callable[[], bool]] = None) -> pd.DataFrame:
    abort = abort_flag or (lambda: False)
    model, mdata_real, reference_stats = load_resources(cfg)
    new_genes_idx, element_gene_map, _ = _precompute_static_structures(
        model, mdata_real, cfg, reference_stats
    )
    n_pos = cfg.n_genes
    n_ntc = max(1, round(0.05 * n_pos))
    n_all = n_pos + n_ntc

    xs = list(range(n_guides_min, n_guides_max + 1, step))
    list_dict = _create_empty_list(method="perturbo")

    element_by_gene_lfc = _build_element_by_gene_lfc(
        element_gene_map=element_gene_map,
        n_elements=n_all, n_elements_pos=n_pos, n_elements_ntc=n_ntc,
        n_genes=cfg.n_genes, cfg=cfg, reference_stats=reference_stats
    )

    for g in xs:
        print(f"Number of Guides per Element: {g}")
        
        if abort():
            # Return whatever summary you’ve accumulated so far (partial results)
            return _detail_to_power_summary(list_dict, grouping_columns, tcfg)  # <-- this util already exists in your file
        
        cfg_g = replace(cfg, n_grna_per_element=int(g))
        guide_eff = _sample_guide_efficacy(n_all, cfg_g, reference_stats)
        mdata_sim = simulate_mudata_from_model(
            model, mdata_real, int(fixed_cells_per_element),
            element_gene_map, element_by_gene_lfc,
            guide_eff, new_genes_idx,
            cfg_g, reference_stats
        )
        list_dict = _train_and_get_effects(mdata_sim, cfg_g.accelerator, cfg_g, tcfg,
                                           int(fixed_cells_per_element), list_dict)

    grouping_columns = ['NCells', 'NGuides', 'NGuidesPerElement', 'NCellsPerElement', 'MOI',
                        'MeanReads', "ReadScaling", "alpha_cor",
                        'Efficacy_type', 'Method', 'MTmethod']
    power_summary = _detail_to_power_summary(list_dict, grouping_columns, tcfg)
    return power_summary


def power_vs_moi(moi_min: float, moi_max: float, n_bins: int,
                 fixed_cells_per_element: int,
                 cfg: SimulationConfig, tcfg: TestConfig,
                 abort_flag: Optional[Callable[[], bool]] = None) -> pd.DataFrame:
    abort = abort_flag or (lambda: False)
    model, mdata_real, reference_stats = load_resources(cfg)
    new_genes_idx, element_gene_map, guide_efficacy = _precompute_static_structures(
        model, mdata_real, cfg, reference_stats
    )
    n_pos = cfg.n_genes
    n_ntc = max(1, round(0.05 * n_pos))
    n_all = n_pos + n_ntc

    element_by_gene_lfc = _build_element_by_gene_lfc(
        element_gene_map=element_gene_map,
        n_elements=n_all, n_elements_pos=n_pos, n_elements_ntc=n_ntc,
        n_genes=cfg.n_genes, cfg=cfg, reference_stats=reference_stats
    )

    xs = np.linspace(moi_min, moi_max, num=n_bins)
    list_dict = _create_empty_list(method="perturbo")

    for m in xs:
        print(f"MOI: {m}")
        
        if abort():
            # Return whatever summary you’ve accumulated so far (partial results)
            return _detail_to_power_summary(list_dict, grouping_columns, tcfg)  # <-- this util already exists in your file
        
        cfg_m = replace(cfg, moi=float(m))
        mdata_sim = simulate_mudata_from_model(
            model, mdata_real, int(fixed_cells_per_element),
            element_gene_map, element_by_gene_lfc,
            guide_efficacy, new_genes_idx,
            cfg_m, reference_stats
        )
        list_dict = _train_and_get_effects(mdata_sim, cfg_m.accelerator, cfg_m, tcfg,
                                           int(fixed_cells_per_element), list_dict)

    grouping_columns = ['NCells', 'NGuides', 'NCellsPerElement', 'MOI',
                        'MeanReads', "ReadScaling", "alpha_cor",
                        'Efficacy_type', 'Method', 'MTmethod']
    power_summary = _detail_to_power_summary(list_dict, grouping_columns, tcfg)
    return power_summary


def power_vs_gene_mean(mean_min: float, mean_max: float, n_bins: int,
                       fixed_cells_per_element: int,
                       cfg: SimulationConfig, tcfg: TestConfig,
                       abort_flag: Optional[Callable[[], bool]] = None) -> pd.DataFrame:
    """
    Approximates different gene means by scaling read_depth_adjust_factor
    relative to the original dataset mean for cfg.gene_name.
    """
    abort = abort_flag or (lambda: False)
    model, mdata_real, reference_stats = load_resources(cfg)
    new_genes_idx, element_gene_map, guide_efficacy = _precompute_static_structures(
        model, mdata_real, cfg, reference_stats
    )
    n_pos = cfg.n_genes
    n_ntc = max(1, round(0.05 * n_pos))
    n_all = n_pos + n_ntc

    element_by_gene_lfc = _build_element_by_gene_lfc(
        element_gene_map=element_gene_map,
        n_elements=n_all, n_elements_pos=n_pos, n_elements_ntc=n_ntc,
        n_genes=cfg.n_genes, cfg=cfg, reference_stats=reference_stats
    )

    base_mean = float(reference_stats["gene_means"].get(cfg.gene_name, 1.0))
    xs = np.linspace(mean_min, mean_max, num=n_bins)
    list_dict = _create_empty_list(method="perturbo")

    for target_mean in xs:
        print(f"Gene Mean: {target_mean}")
        
        if abort():
            # Return whatever summary you’ve accumulated so far (partial results)
            return _detail_to_power_summary(list_dict, grouping_columns, tcfg)  # <-- this util already exists in your file
        
        scale = (target_mean / base_mean) if base_mean > 0 else 1.0
        cfg_m = replace(cfg,
                        mean_mode="fixed",
                        mean_expression=float(target_mean),
                        read_depth_adjust_factor=cfg.read_depth_adjust_factor * scale)

        mdata_sim = simulate_mudata_from_model(
            model, mdata_real, int(fixed_cells_per_element),
            element_gene_map, element_by_gene_lfc,
            guide_efficacy, new_genes_idx,
            cfg_m, reference_stats
        )
        list_dict = _train_and_get_effects(mdata_sim, cfg_m.accelerator, cfg_m, tcfg,
                                           int(fixed_cells_per_element), list_dict)

    grouping_columns = ['NCells', 'NGuides', 'NCellsPerElement', 'MOI',
                        'MeanReads', "ReadScaling", "alpha_cor",
                        'Efficacy_type', 'Method', 'MTmethod']
    power_summary = _detail_to_power_summary(list_dict, grouping_columns, tcfg)
    return power_summary.rename(columns={"MeanReads": "GeneMeanApprox"})
