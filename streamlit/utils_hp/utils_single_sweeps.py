
# utils/utils_single_sweeps.py
# Utilities extracted/adapted from the notebooks to support the Streamlit app.
from __future__ import annotations

import os
import glob

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ------------------------------
# Config & file resolution
# ------------------------------
@dataclass
class DatasetConfig:
    dataset_name: str  # "Gasperini (high MOI)" or "Weissman (low MOI)"
    test_type_ui: str  # "empirical" or "fixed" (UI spelling kept)
    mt_method: str     # "none" or "FDR"
    alpha: float       # 0.1 by default
    #data_dir: Path

    def test_type_token(self) -> str:
        # Map UI spelling "empitical" -> "empirical" which appears in filenames
        return self.test_type_ui.lower().strip()

    def moi_token(self) -> str:
        # Use the same tokens we observed in the notebooks' file names
        if "gasperini" in self.dataset_name.lower():
            return "moi30"   # high MOI
        elif "weissman" in self.dataset_name.lower():
            return "moi1"       # low MOI (Weissman)

    def mt_token(self) -> str:
        # Match examples like *_FDR_* or *_none_*
        return self.mt_method

    def alpha_token(self) -> str:
        # Examples showed 0.1 printed as "0.1"
        return f"{self.alpha}"
    
    def power_file_path(self) -> str:
        if "gasperini" in self.dataset_name.lower():
            return "gasperini_res"
            # return "/srv/perturbo/st-single-gene-calculator/streamlit/gasperini_res"
        elif "weissman" in self.dataset_name.lower():
            return "weissman_res"
            # return "/srv/perturbo/st-single-gene-calculator/streamlit/weissman_res"
        
    def power_filename_base(self) -> str:
        # Example: "power_moi30_empirical_FDR_0.1" and "power_moi30_allLFC_empirical_FDR_0.1"
        parts = ["power", self.moi_token(), "allLFC"]
        parts.extend([self.test_type_token(), self.mt_token(), self.alpha_token()])
        return "_".join(parts)
    
    def resolve_path(self) -> str:
        fname = self.power_filename_base() + ".csv"
        return os.path.join(self.power_file_path(), fname)

# ------------------------------
# Loading & preparation
# ------------------------------
def load_power_tables(cfg: DatasetConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the per-LFC and ALL-LFC power tables based on the naming convention."""
    p2 = cfg.resolve_path()
    power_allLFC_df = pd.read_csv(p2)
    return power_allLFC_df

def prepare_power_tables_for_plotting(power_allLFC_df: pd.DataFrame):
    """Mirror notebook transformations: compute ReadDepth, categoricals, filters."""
    def _add_readdepth(df: pd.DataFrame) -> pd.DataFrame:
        if "ReadDepth" in df.columns:
            return df
        if "ReadScaling" in df.columns and "MeanReads" in df.columns:
            # From notebook: ReadDepth = groupby(ReadScaling).mean(MeanReads) * 13135 (then int)
            # Keep behavior but guard if 'MeanReads' is already total UMIs per cell.
            df = df.copy()
            rd = df.groupby("ReadScaling")["MeanReads"].transform("mean") * 13135
            df["ReadDepth"] = rd.round().astype(int)
        elif "ReadDepth" not in df.columns and "MeanReads" in df.columns:
            df = df.copy()
            df["ReadDepth"] = df["MeanReads"].round().astype(int)
        return df

    power_allLFC_df_plot = _add_readdepth(power_allLFC_df)

    # Filter like in the notebook: restrict very large cells-per-element if present
    if "NCellsPerElement" in power_allLFC_df_plot.columns:
        power_allLFC_df_plot = power_allLFC_df_plot.loc[power_allLFC_df_plot["NCellsPerElement"] <= 2000].copy()

    # Create ordered categories for plotting
    def _ordered_levels(series: pd.Series) -> List:
        vals = pd.unique(series.dropna())
        try:
            vals = sorted(vals, key=lambda x: int(x))
        except Exception:
            vals = list(vals)
        return vals

    if "ReadDepth" in power_allLFC_df_plot.columns:
        power_allLFC_df_plot["ReadDepth"] = pd.Categorical(
            power_allLFC_df_plot["ReadDepth"],
            categories=_ordered_levels(power_allLFC_df_plot["ReadDepth"]),
            ordered=True,
        )

    if "NCellsPerElement" in power_allLFC_df_plot.columns:
        power_allLFC_df_plot["NCellsPerElement_cat"] = pd.Categorical(
            power_allLFC_df_plot["NCellsPerElement"],
            categories=_ordered_levels(power_allLFC_df_plot["NCellsPerElement"]),
            ordered=True,
        )

    # Round power to 2 decimals for display (keep original too if needed)
    power_allLFC_df_plot["Power"] = pd.to_numeric(power_allLFC_df_plot["Power"], errors="coerce")
    power_allLFC_df_plot["Power_rounded"] = np.round(power_allLFC_df_plot["Power"], 2)

    return power_allLFC_df_plot

# ------------------------------
# Cost model (adapted from notebook signature)
# ------------------------------
def get_cost(
    umi_count: Optional[float] = None,
    saturation_rate: float = 0.6,
    read_count: Optional[float] = None,
    num_elements: Optional[int] = None,
    num_guides: Optional[int] = None,
    num_cells: Optional[int] = None,
    num_guides_per_element: int = 4,
    num_cells_per_element: Optional[int] = None,
    moi: Optional[float] = None,
    num_reads_per_flow_cell: float = 400e6,
    num_cells_per_lane: int = 20000,
    num_lanes_per_kit: int = 6,
    lib_prep_cost_per_cell: float = 0.05,
    seq_cost_per_mio: float = 3.42,
) -> float:
    """
    Estimate total experiment cost (library prep + sequencing).

    - If umi_count is given (UMIs per cell) and read_count is None, we convert
      to reads-per-cell using: reads = umi_count / (1 - saturation_rate).
    - Library prep: assume a kit with num_lanes_per_kit lanes, each lane handles
      num_cells_per_lane cells. We pay per cell for a full number of kits
      to cover the required number of cells.
    - Sequencing: compute total reads = num_cells * reads_per_cell, then pay
      per whole flow cell (ceiling) with capacity num_reads_per_flow_cell.

    The signature mirrors your notebook; 'moi', 'num_elements', 'num_guides'
    are accepted but not explicitly used in this simple cost model.
    """
    if read_count is None:
        if umi_count is None:
            raise ValueError("Provide either read_count or umi_count.")
        if saturation_rate >= 1.0:
            raise ValueError("saturation_rate must be < 1.0")
        read_count = float(umi_count) / (1.0 - float(saturation_rate))

    if num_cells is None:
        # We expect 'NCells' column to provide total number of cells
        raise ValueError("num_cells must be provided for cost computation.")

    # Library prep
    kit_capacity = int(num_cells_per_lane) * int(num_lanes_per_kit)
    kits_needed = math.ceil(float(num_cells) / float(kit_capacity))
    library_prep_cost = kits_needed * kit_capacity * float(lib_prep_cost_per_cell)

    # Sequencing
    cost_per_flow_cell = (float(num_reads_per_flow_cell) / 1e6) * float(seq_cost_per_mio)
    total_reads = float(num_cells) * float(read_count)
    flowcells_needed = math.ceil(total_reads / float(num_reads_per_flow_cell))
    sequencing_cost = flowcells_needed * cost_per_flow_cell

    return library_prep_cost + sequencing_cost

def compute_cost_per_row(
    df: pd.DataFrame,
    num_reads_per_flow_cell: float = 400e6,
    num_cells_per_lane: int = 20000,
    num_lanes_per_kit: int = 6,
    lib_prep_cost_per_cell: float = 0.05,
    seq_cost_per_mio: float = 3.42,
) -> pd.DataFrame:
    """Add a 'cost' column using get_cost for each row (expects columns used in the notebook)."""
    req_cols = ["ReadDepth", "NGuides", "NCells", "NCellsPerElement"]
    for c in req_cols:
        if c not in df.columns:
            raise KeyError(f"Expected column '{c}' in the power table for cost computation.")

    out = df.copy()
    out["cost"] = out.apply(
        lambda row: get_cost(
            umi_count=row["ReadDepth"],
            saturation_rate=0.6,
            num_guides=int(row["NGuides"]),
            num_cells=int(row["NCells"]),
            num_guides_per_element=4,
            num_cells_per_element=int(row["NCellsPerElement"]),
            moi=row["MOI"] if "MOI" in out.columns else None,
            num_reads_per_flow_cell=num_reads_per_flow_cell,
            num_cells_per_lane=num_cells_per_lane,
            num_lanes_per_kit=num_lanes_per_kit,
            lib_prep_cost_per_cell=lib_prep_cost_per_cell,
            seq_cost_per_mio=seq_cost_per_mio,
        ),
        axis=1,
    )
    return out

# ------------------------------
# Budget helpers
# ------------------------------
def compute_budget_boundary(df_cost: pd.DataFrame, budget_eur: float):
    """
    For each ReadDepth, find the maximum NCellsPerElement affordable under the budget.
    Returns list of (ReadDepth, max_cells_per_element) pairs.
    """
    pts = []
    for rd, sub in df_cost.groupby("ReadDepth"):
        affordable = sub.loc[sub["cost"] <= budget_eur]
        if affordable.empty:
            continue
        # take the row with max cells-per-element; if ties, take max Power
        best_row = affordable.sort_values(["NCellsPerElement", "Power"], ascending=[True, False]).iloc[-1]
        pts.append((int(rd), int(best_row["NCellsPerElement"])))
    pts.sort(key=lambda x: x[0])
    return pts

def find_best_combo_under_budget(df_cost: pd.DataFrame, budget_eur: float):
    affordable = df_cost.loc[df_cost["cost"] <= budget_eur]
    if affordable.empty:
        return None
    return affordable.sort_values("Power", ascending=False).iloc[0]

def build_budget_power_curve(df_cost: pd.DataFrame, max_budget: Optional[float] = None, num_points: int = 60):
    """
    Compute max achievable power as a function of budget.
    Budgets range from 0 to max observed cost (or provided max_budget).
    """
    if max_budget is None:
        max_budget = float(df_cost["cost"].max())
    budgets = np.linspace(0.0, max_budget, num_points)
    max_powers = []
    for b in budgets:
        affordable = df_cost.loc[df_cost["cost"] <= b]
        max_powers.append(affordable["Power"].max() if not affordable.empty else np.nan)
    return budgets, max_powers
