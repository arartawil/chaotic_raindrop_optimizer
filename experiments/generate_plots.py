"""CRD-specific plots that supplement heurilab's convergence/box plots.

Produces, per suite-dir:
  - Statistical Plots/wilcoxon_heatmap.{pdf,png}
  - Statistical Plots/win_tie_loss.{pdf,png}
  - Statistical Plots/cd_diagram.{pdf,png}
  - Statistical Plots/radar.{pdf,png}
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.statistical_analysis import discover_suite_dirs  # noqa: E402

plt.rcParams.update(
    {
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "DejaVu Sans",
    }
)


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)


def _load_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


# ───────────────────────────────────────────────────────────── heatmap (8.4)
def heatmap(suite_dir: Path) -> None:
    w = _load_csv(suite_dir / "Statistical Reports" / "wilcoxon_results.csv")
    if w is None or w.empty:
        return
    w["signed_log_p"] = np.where(
        w["mean_algo"] < w["mean_baseline"],
        -np.log10(np.clip(w["p_value"], 1e-12, 1.0)),
        np.log10(np.clip(w["p_value"], 1e-12, 1.0)),
    )
    pivot = w.pivot_table(index="algorithm", columns="benchmark", values="signed_log_p")
    pivot = pivot.sort_index()
    fig, ax = plt.subplots(figsize=(max(8, 0.2 * pivot.shape[1] + 2), 0.45 * pivot.shape[0] + 1))
    sns.heatmap(
        pivot, cmap="RdYlGn", center=0.0,
        cbar_kws=dict(label="signed -log10(p)"), ax=ax,
        linewidths=0.2, linecolor="white",
    )
    ax.set_xlabel("Function")
    ax.set_ylabel("")
    ax.set_title("Wilcoxon p-values vs RD (green = better, red = worse)")
    ax.tick_params(axis="x", rotation=70)
    _save(fig, suite_dir / "Statistical Plots" / "wilcoxon_heatmap")


# ─────────────────────────────────────────────────────── win/tie/loss (8.5)
def win_tie_loss(suite_dir: Path) -> None:
    s = _load_csv(suite_dir / "Statistical Reports" / "wilcoxon_summary.csv")
    if s is None or s.empty:
        return
    crds = s[s["algorithm"].str.startswith("CRD-")].copy()
    if crds.empty:
        crds = s.copy()
    crds = crds.sort_values("W+", ascending=False)
    fig, ax = plt.subplots(figsize=(max(7, 0.5 * len(crds) + 3), 3.6))
    x = np.arange(len(crds))
    ax.bar(x - 0.25, crds["W+"], width=0.25, color="seagreen", label="Win (+)")
    ax.bar(x, crds["W="], width=0.25, color="lightgray", label="Tie (=)")
    ax.bar(x + 0.25, crds["W-"], width=0.25, color="firebrick", label="Loss (-)")
    ax.set_xticks(x)
    ax.set_xticklabels(crds["algorithm"], rotation=60, ha="right")
    ax.set_ylabel("Number of functions")
    ax.set_title("Wilcoxon W+/W=/W- vs RD")
    ax.legend()
    _save(fig, suite_dir / "Statistical Plots" / "win_tie_loss")


# ───────────────────────────────────────────────────── critical-difference (8.6)
def cd_diagram(suite_dir: Path) -> None:
    rdf = _load_csv(suite_dir / "Statistical Reports" / "friedman_rankings.csv")
    sdf = _load_csv(suite_dir / "Statistical Reports" / "friedman_summary.csv")
    if rdf is None or rdf.empty or sdf is None or sdf.empty:
        return
    ranks = rdf.sort_values("mean_rank")
    cd_value = float(sdf.iloc[0]["nemenyi_cd"])
    algos = ranks["algorithm"].tolist()
    rs = ranks["mean_rank"].tolist()

    n = len(algos)
    fig, ax = plt.subplots(figsize=(9.0, 0.45 * n + 2.0))
    x_min = max(1, int(np.floor(min(rs) - 0.5)))
    x_max = int(np.ceil(max(rs) + 0.5))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-1.5, n + 1)
    ax.hlines(n + 0.5, x_min, x_max, color="black")
    for tick in range(x_min, x_max + 1):
        ax.vlines(tick, n + 0.3, n + 0.5, color="black")
        ax.text(tick, n + 0.7, str(tick), ha="center", va="bottom", fontsize=8)

    for i, (algo, r) in enumerate(zip(algos, rs)):
        y = n - i
        ax.plot([r, r], [n + 0.5, y], color="black", linewidth=0.8)
        if r <= (x_min + x_max) / 2:
            ax.plot([r, x_min], [y, y], color="black", linewidth=0.6)
            ax.text(x_min, y, f" {algo} ({r:.2f}) ", ha="right", va="center", fontsize=9)
        else:
            ax.plot([r, x_max], [y, y], color="black", linewidth=0.6)
            ax.text(x_max, y, f" {algo} ({r:.2f}) ", ha="left", va="center", fontsize=9)

    if np.isfinite(cd_value):
        ax.hlines(-0.6, x_min + 0.2, x_min + 0.2 + cd_value, color="red", linewidth=2.5)
        ax.text(
            x_min + 0.2 + cd_value / 2, -1.0,
            f"CD = {cd_value:.3f}", ha="center", va="top",
            color="red", fontsize=9,
        )
    ax.set_axis_off()
    ax.set_title("Friedman–Nemenyi Critical Difference")
    _save(fig, suite_dir / "Statistical Plots" / "cd_diagram")


# ────────────────────────────────────────────────────────────── radar (8.3)
def radar(suite_dir: Path) -> None:
    rdf = _load_csv(suite_dir / "Statistical Reports" / "friedman_rankings.csv")
    if rdf is None or rdf.empty:
        return
    ranks = rdf.set_index("algorithm")["mean_rank"]
    inv = ranks.max() - ranks + 1.0  # bigger = better
    algos = inv.index.tolist()
    angles = np.linspace(0, 2 * np.pi, len(algos), endpoint=False).tolist()
    values = inv.values.tolist()
    angles += angles[:1]
    values += values[:1]
    fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw=dict(polar=True))
    ax.plot(angles, values, color="tab:blue", linewidth=1.6)
    ax.fill(angles, values, color="tab:blue", alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(algos, fontsize=8)
    ax.set_title(f"Average rank — {suite_dir.name}\n(higher = better)")
    _save(fig, suite_dir / "Statistical Plots" / "radar")


def run_for_suite(suite_dir: Path) -> None:
    heatmap(suite_dir)
    win_tie_loss(suite_dir)
    cd_diagram(suite_dir)
    radar(suite_dir)
    print(f"[plots] extras saved under {suite_dir / 'Statistical Plots'}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default=str(PROJECT_ROOT / "results"))
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    for s in discover_suite_dirs(Path(args.results_dir)):
        run_for_suite(s)
