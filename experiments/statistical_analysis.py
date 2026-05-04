"""CRD-specific statistical extras on top of heurilab's CSV outputs.

heurilab already emits per-run CSVs and Wilcoxon/Friedman Excel files.
This module adds, into the same suite folder:
  - cohens_d_results.csv  (best CRD vs RD per function)
  - wilcoxon_results.csv  (RD vs every other algorithm, signed p-values)
  - friedman_rankings.csv (mean Friedman rank per algorithm)
  - friedman_summary.csv  (chi², p, Nemenyi CD)
  - statistical_summary.csv (W+/=/-, mean rank)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HEURILAB_ROOT = Path("c:/Users/ROG SRTIX/Desktop/pkg/heurilab")
for p in (PROJECT_ROOT, HEURILAB_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


ALPHA = 0.05


def _load_runs(suite_dir: Path) -> pd.DataFrame:
    """Load heurilab's raw_runs.csv files from a suite dir.

    Two layouts are supported:
      1) ``suite_dir/CSV Data/raw_runs.csv``                 (single combined dir)
      2) ``suite_dir/<func_name>/CSV Data/raw_runs.csv``     (per-function dirs,
         emitted by the multiprocessing runner)
    """
    direct = suite_dir / "CSV Data" / "raw_runs.csv"
    paths: List[Path] = []
    if direct.exists():
        paths.append(direct)
    else:
        for sub in sorted(suite_dir.iterdir()):
            if sub.is_dir() and (sub / "CSV Data" / "raw_runs.csv").exists():
                paths.append(sub / "CSV Data" / "raw_runs.csv")
    if not paths:
        raise FileNotFoundError(f"No raw_runs.csv under {suite_dir}")
    frames = [
        pd.read_csv(p, usecols=["Benchmark", "Algorithm", "Run", "BestFitness", "Time_s"])
        for p in paths
    ]
    return pd.concat(frames, ignore_index=True)


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    var_a = np.var(a, ddof=1)
    var_b = np.var(b, ddof=1)
    pooled = np.sqrt(((len(a) - 1) * var_a + (len(b) - 1) * var_b) / (len(a) + len(b) - 2))
    if pooled == 0 or not np.isfinite(pooled):
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled)


def classify_d(d: float) -> str:
    a = abs(d)
    if a < 0.2:
        return "negligible"
    if a < 0.5:
        return "small"
    if a < 0.8:
        return "medium"
    return "large"


# ─── Wilcoxon ────────────────────────────────────────────────────────────────
def wilcoxon_table(df: pd.DataFrame, baseline: str = "RD") -> pd.DataFrame:
    rows = []
    benches = sorted(df["Benchmark"].unique())
    algos = sorted(df["Algorithm"].unique())
    for bench in benches:
        sub = df[df.Benchmark == bench]
        b_vals = sub[sub.Algorithm == baseline]["BestFitness"].values
        if len(b_vals) == 0:
            continue
        for algo in algos:
            if algo == baseline:
                continue
            a_vals = sub[sub.Algorithm == algo]["BestFitness"].values
            if len(a_vals) == 0:
                continue
            try:
                stat, p = stats.ranksums(a_vals, b_vals)
            except Exception:
                stat, p = float("nan"), float("nan")
            ma, mb = float(np.mean(a_vals)), float(np.mean(b_vals))
            sig = "="
            if np.isfinite(p) and p < ALPHA:
                sig = "+" if ma < mb else "-"
            rows.append(
                dict(
                    benchmark=bench,
                    algorithm=algo,
                    baseline=baseline,
                    statistic=float(stat) if np.isfinite(stat) else float("nan"),
                    p_value=float(p) if np.isfinite(p) else float("nan"),
                    mean_algo=ma,
                    mean_baseline=mb,
                    significance=sig,
                )
            )
    return pd.DataFrame(rows)


def wilcoxon_summary(wdf: pd.DataFrame) -> pd.DataFrame:
    g = wdf.groupby("algorithm")["significance"].value_counts().unstack(fill_value=0)
    for col in ("+", "=", "-"):
        if col not in g.columns:
            g[col] = 0
    g = g.rename(columns={"+": "W+", "=": "W=", "-": "W-"})
    return g[["W+", "W=", "W-"]].reset_index()


# ─── Friedman + Nemenyi ──────────────────────────────────────────────────────
def friedman_rankings(df: pd.DataFrame):
    pivot = df.groupby(["Benchmark", "Algorithm"])["BestFitness"].mean().reset_index()
    matrix = pivot.pivot_table(index="Benchmark", columns="Algorithm", values="BestFitness").dropna()
    if matrix.empty:
        return pd.DataFrame(columns=["algorithm", "mean_rank"]), float("nan"), float("nan")
    ranks = matrix.rank(axis=1, method="average")
    mean_ranks = ranks.mean(axis=0).sort_values()
    out = mean_ranks.reset_index()
    out.columns = ["algorithm", "mean_rank"]
    try:
        stat, p = stats.friedmanchisquare(*[matrix[c].values for c in matrix.columns])
    except Exception:
        stat, p = float("nan"), float("nan")
    return out, float(stat), float(p)


def nemenyi_cd(n_algos: int, n_datasets: int, alpha: float = 0.05) -> float:
    q_alpha = {
        2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
        8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313,
        14: 3.354, 15: 3.391, 16: 3.426, 17: 3.458, 18: 3.489, 19: 3.517,
        20: 3.544, 21: 3.569, 22: 3.593, 23: 3.616, 24: 3.638, 25: 3.660,
    }
    q = q_alpha.get(int(n_algos))
    if q is None or n_datasets < 1:
        return float("nan")
    return float(q * np.sqrt(n_algos * (n_algos + 1) / (6.0 * n_datasets)))


# ─── Cohen's d ───────────────────────────────────────────────────────────────
def cohens_d_table(df: pd.DataFrame, baseline: str = "RD") -> pd.DataFrame:
    rows = []
    crds = [a for a in df["Algorithm"].unique() if a.startswith("CRD-")]
    for bench in sorted(df["Benchmark"].unique()):
        sub = df[df.Benchmark == bench]
        b_vals = sub[sub.Algorithm == baseline]["BestFitness"].values
        if len(b_vals) == 0:
            continue
        best_algo, best_mean, best_vals = None, np.inf, None
        for c in crds:
            v = sub[sub.Algorithm == c]["BestFitness"].values
            if len(v) == 0:
                continue
            m = float(np.mean(v))
            if m < best_mean:
                best_mean = m
                best_algo = c
                best_vals = v
        if best_algo is None:
            continue
        d = cohen_d(best_vals, b_vals)
        rows.append(
            dict(
                benchmark=bench,
                best_crd=best_algo,
                mean_crd=best_mean,
                mean_rd=float(np.mean(b_vals)),
                cohen_d=d,
                effect=classify_d(d),
            )
        )
    return pd.DataFrame(rows)


# ─── Driver ──────────────────────────────────────────────────────────────────
def run_for_suite(suite_dir: Path, baseline: str = "RD") -> None:
    df = _load_runs(suite_dir)
    out_dir = suite_dir / "Statistical Reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    wdf = wilcoxon_table(df, baseline=baseline)
    wdf.to_csv(out_dir / "wilcoxon_results.csv", index=False)
    wsum = wilcoxon_summary(wdf)
    wsum.to_csv(out_dir / "wilcoxon_summary.csv", index=False)

    rdf, fstat, fp = friedman_rankings(df)
    rdf.to_csv(out_dir / "friedman_rankings.csv", index=False)
    n_algos = len(rdf)
    n_datasets = df["Benchmark"].nunique()
    cd = nemenyi_cd(n_algos, n_datasets)
    pd.DataFrame(
        [
            dict(
                friedman_statistic=fstat,
                friedman_p_value=fp,
                nemenyi_cd=cd,
                n_algorithms=n_algos,
                n_datasets=n_datasets,
            )
        ]
    ).to_csv(out_dir / "friedman_summary.csv", index=False)

    cdf = cohens_d_table(df, baseline=baseline)
    cdf.to_csv(out_dir / "cohens_d_results.csv", index=False)

    summary = wsum.merge(rdf, on="algorithm", how="outer").sort_values("mean_rank")
    summary.to_csv(out_dir / "statistical_summary.csv", index=False)

    print(
        f"[stats] {suite_dir.name}: Friedman chi2={fstat:.3f}  p={fp:.3g}  "
        f"CD={cd:.3f}  algos={n_algos}  datasets={n_datasets}"
    )


def discover_suite_dirs(root: Path) -> List[Path]:
    """Find suite dirs under ``root``.

    A suite dir is either:
      * an immediate child with ``CSV Data/raw_runs.csv``, OR
      * an immediate child whose grandchildren each contain ``CSV Data/raw_runs.csv``
        (per-function layout from the multiprocessing runner).
    """
    out: List[Path] = []
    if not root.exists():
        return out
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        if (p / "CSV Data" / "raw_runs.csv").exists():
            out.append(p)
            continue
        # per-function layout
        if any(
            sub.is_dir() and (sub / "CSV Data" / "raw_runs.csv").exists()
            for sub in p.iterdir()
        ):
            out.append(p)
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default=str(PROJECT_ROOT / "results"))
    p.add_argument("--baseline", default="RD")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    root = Path(args.results_dir)
    suites = discover_suite_dirs(root)
    if not suites:
        print(f"[stats] no suite dirs found under {root}")
        sys.exit(0)
    for s in suites:
        run_for_suite(s, baseline=args.baseline)
