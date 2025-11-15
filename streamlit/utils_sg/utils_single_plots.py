# utils_plots.py
import matplotlib.pyplot as plt
import pandas as pd

def plot_power_generic(df: pd.DataFrame, x_col: str, x_label: str, title: str):
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.plot(
        df[x_col], df["Power"],
        linestyle='-', color='black',
        marker='o', markerfacecolor='black', markeredgecolor='black',
        linewidth=1.5, markersize=5
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel("Power")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.axhline(0.8, color='red', linestyle='--', linewidth=1)
    ax.text(0.99, 0.8, "0.8", color='red', va='center', ha='right',
            transform=ax.get_yaxis_transform())
    fig.tight_layout()
    return fig, ax

def save_figure(fig, path: str, dpi: int = 300):
    ext = str(path).split(".")[-1].lower()
    if ext not in {"png", "pdf"}:
        raise ValueError("Please use a .png or .pdf filename")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path
