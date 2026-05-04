"""Generate LaTeX tables for the CRD paper from heurilab CSV outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.statistical_analysis import discover_suite_dirs  # noqa: E402


def _esc(s: str) -> str:
    return str(s).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%").replace("#", r"\#")


def _fmt(v: float, digits: int = 3) -> str:
    if v is None or not np.isfinite(v):
        return "--"
    if v == 0:
        return "0.000e+00"
    return f"{v:.{digits}e}"


def _wrap(latex_body: str, caption: str, label: str) -> str:
    return (
        "\\begin{table}[!ht]\n"
        "\\centering\n"
        "\\small\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{tab:{label}}}\n"
        + latex_body +
        "\n\\end{table}\n"
    )


def _load_results_csv(suite_dir: Path) -> pd.DataFrame | None:
    """Concatenate ``results.csv`` from a suite dir or its per-function children."""
    direct = suite_dir / "CSV Data" / "results.csv"
    if direct.exists():
        return pd.read_csv(direct)
    frames = []
    for sub in sorted(suite_dir.iterdir()):
        p = sub / "CSV Data" / "results.csv"
        if p.exists():
            frames.append(pd.read_csv(p))
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def table_results(suite_dir: Path, out_dir: Path, caption_prefix: str) -> None:
    """Mean ± Std table (best per row in bold)."""
    df = _load_results_csv(suite_dir)
    if df is None:
        return
    benches = sorted(df["Benchmark"].unique(), key=lambda s: (len(s), s))
    algos = list(dict.fromkeys(df["Algorithm"].tolist()))  # preserve order

    rows = []
    for bench in benches:
        sub = df[df.Benchmark == bench].set_index("Algorithm")
        means = sub["Mean"].to_dict()
        best_algo = min(means, key=lambda a: means[a]) if means else None
        cells = [_esc(bench)]
        for a in algos:
            if a not in sub.index:
                cells.append("--")
                continue
            mean = float(sub.loc[a, "Mean"])
            std = float(sub.loc[a, "Std"])
            txt = f"{_fmt(mean)} $\\pm$ {_fmt(std, 2)}"
            if a == best_algo:
                txt = "\\textbf{" + txt + "}"
            cells.append(txt)
        rows.append(" & ".join(cells) + " \\\\")

    col_spec = "l" + "c" * len(algos)
    header = "Function & " + " & ".join(_esc(a) for a in algos) + " \\\\"
    body = (
        f"\\begin{{tabular}}{{{col_spec}}}\n"
        "\\hline\n"
        f"{header}\n"
        "\\hline\n"
        + "\n".join(rows)
        + "\n\\hline\n"
        "\\end{tabular}"
    )
    out = _wrap(body, f"{caption_prefix}: Mean $\\pm$ Std for {suite_dir.name}", f"results-{suite_dir.name}")
    (out_dir / f"table_results_{suite_dir.name}.tex").write_text(out, encoding="utf-8")


def table_wilcoxon(suite_dir: Path, out_dir: Path) -> None:
    p = suite_dir / "Statistical Reports" / "wilcoxon_results.csv"
    if not p.exists():
        return
    w = pd.read_csv(p)
    if w.empty:
        return
    benches = sorted(w["benchmark"].unique(), key=lambda s: (len(s), s))
    algos = sorted(w["algorithm"].unique())

    pivot_p = w.pivot_table(index="benchmark", columns="algorithm", values="p_value")
    pivot_sig = w.pivot_table(index="benchmark", columns="algorithm", values="significance", aggfunc="first")

    rows = []
    for bench in benches:
        cells = [_esc(bench)]
        for a in algos:
            p_val = pivot_p.loc[bench, a] if a in pivot_p.columns else float("nan")
            sig = pivot_sig.loc[bench, a] if a in pivot_sig.columns else "="
            cells.append(f"{p_val:.2e} ({sig})" if np.isfinite(p_val) else "--")
        rows.append(" & ".join(cells) + " \\\\")

    # summary footer
    summary = w.groupby("algorithm")["significance"].value_counts().unstack(fill_value=0)
    for col in ("+", "=", "-"):
        if col not in summary.columns:
            summary[col] = 0
    summary_row = ["W+/=/-"]
    for a in algos:
        if a in summary.index:
            summary_row.append(f"{int(summary.loc[a, '+'])}/{int(summary.loc[a, '='])}/{int(summary.loc[a, '-'])}")
        else:
            summary_row.append("--")
    rows.append("\\hline")
    rows.append(" & ".join(summary_row) + " \\\\")

    col_spec = "l" + "c" * len(algos)
    header = "Function & " + " & ".join(_esc(a) for a in algos) + " \\\\"
    body = (
        f"\\begin{{tabular}}{{{col_spec}}}\n"
        "\\hline\n"
        f"{header}\n"
        "\\hline\n"
        + "\n".join(rows)
        + "\n\\hline\n"
        "\\end{tabular}"
    )
    out = _wrap(body, f"Wilcoxon p-values vs RD on {suite_dir.name}", f"wilcoxon-{suite_dir.name}")
    (out_dir / f"table_wilcoxon_{suite_dir.name}.tex").write_text(out, encoding="utf-8")


def table_friedman(suite_dir: Path, out_dir: Path) -> None:
    rpath = suite_dir / "Statistical Reports" / "friedman_rankings.csv"
    spath = suite_dir / "Statistical Reports" / "friedman_summary.csv"
    if not rpath.exists():
        return
    r = pd.read_csv(rpath).sort_values("mean_rank")
    rows = [
        f"{_esc(row.algorithm)} & {row.mean_rank:.3f} \\\\"
        for row in r.itertuples()
    ]
    if spath.exists():
        s = pd.read_csv(spath).iloc[0]
        rows.append(
            f"\\hline\nFriedman $\\chi^2$ & {s.friedman_statistic:.3f} \\\\\n"
            f"p-value & {s.friedman_p_value:.3g} \\\\\n"
            f"Nemenyi CD & {s.nemenyi_cd:.3f} \\\\"
        )
    body = (
        "\\begin{tabular}{lc}\n"
        "\\hline\nAlgorithm & Mean rank \\\\\n\\hline\n"
        + "\n".join(rows)
        + "\n\\hline\n\\end{tabular}"
    )
    out = _wrap(body, f"Friedman rankings on {suite_dir.name}", f"friedman-{suite_dir.name}")
    (out_dir / f"table_friedman_{suite_dir.name}.tex").write_text(out, encoding="utf-8")


def table_cohen(suite_dir: Path, out_dir: Path) -> None:
    p = suite_dir / "Statistical Reports" / "cohens_d_results.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    if d.empty:
        return
    rows = [
        f"{_esc(r.benchmark)} & {_esc(r.best_crd)} & {_fmt(r.mean_crd)} & {_fmt(r.mean_rd)} & {r.cohen_d:.3f} & {r.effect} \\\\"
        for r in d.itertuples()
    ]
    body = (
        "\\begin{tabular}{llcccc}\n"
        "\\hline\nFunction & Best CRD & Mean CRD & Mean RD & Cohen's d & Effect \\\\\n\\hline\n"
        + "\n".join(rows)
        + "\n\\hline\n\\end{tabular}"
    )
    out = _wrap(body, f"Cohen's d (best CRD vs RD) on {suite_dir.name}", f"cohen-{suite_dir.name}")
    (out_dir / f"table_cohen_{suite_dir.name}.tex").write_text(out, encoding="utf-8")


def table_runtime(suite_dir: Path, out_dir: Path) -> None:
    direct = suite_dir / "CSV Data" / "raw_runs.csv"
    if direct.exists():
        paths = [direct]
    else:
        paths = [
            sub / "CSV Data" / "raw_runs.csv"
            for sub in sorted(suite_dir.iterdir())
            if (sub / "CSV Data" / "raw_runs.csv").exists()
        ]
    if not paths:
        return
    df = pd.concat([pd.read_csv(p, usecols=["Algorithm", "Time_s"]) for p in paths], ignore_index=True)
    rt = df.groupby("Algorithm")["Time_s"].agg(["mean", "std"]).reset_index()
    rt = rt.sort_values("mean")
    rows = [
        f"{_esc(r.Algorithm)} & {r['mean']:.3f} & {r['std']:.3f} \\\\"
        for _, r in rt.iterrows()
    ]
    body = (
        "\\begin{tabular}{lcc}\n"
        "\\hline\nAlgorithm & Mean runtime (s) & Std \\\\\n\\hline\n"
        + "\n".join(rows)
        + "\n\\hline\n\\end{tabular}"
    )
    out = _wrap(body, f"Computational time on {suite_dir.name}", f"runtime-{suite_dir.name}")
    (out_dir / f"table_runtime_{suite_dir.name}.tex").write_text(out, encoding="utf-8")


def run_for_suite(suite_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    table_results(suite_dir, out_dir, caption_prefix="Results")
    table_wilcoxon(suite_dir, out_dir)
    table_friedman(suite_dir, out_dir)
    table_cohen(suite_dir, out_dir)
    table_runtime(suite_dir, out_dir)
    print(f"[latex] tables for {suite_dir.name} saved to {out_dir}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default=str(PROJECT_ROOT / "results"))
    p.add_argument("--out-dir", default=str(PROJECT_ROOT / "tables"))
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out_dir = Path(args.out_dir)
    suites = discover_suite_dirs(Path(args.results_dir))
    if not suites:
        print(f"[latex] no suite dirs found under {args.results_dir}")
        sys.exit(0)
    for s in suites:
        run_for_suite(s, out_dir)
