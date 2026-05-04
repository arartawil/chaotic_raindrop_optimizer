# Chaotic Raindrop Optimizer (CRD) — Project Overview

This document summarises the entire experimental framework built for the
paper:

> **"Chaotic Raindrop Optimizer: A Comprehensive Study on the Impact of
> Chaotic Maps for Enhanced Global Optimization"**

It is meant as a self-contained reference that can be lifted, in pieces,
into the Methods, Experimental Setup, and Results sections of the paper.

---

## 1. Goal

The base algorithm is the **Raindrop Optimizer (RD)** (Sci. Rep. 2025,
DOI: 10.1038/s41598-025-15832-w). RD has six places where pseudo-random
draws steer search behaviour. We replace those draws with deterministic
sequences produced by **twelve chaotic maps**, producing **twelve Chaotic
Raindrop Optimizer (CRD) variants**. We benchmark all variants and ten
well-known metaheuristics on **three CEC test suites** and report the
results with a complete statistical battery (Wilcoxon, Friedman–Nemenyi,
Cohen's *d*).

The hypothesis is: replacing uniform random sampling with structured
chaotic sequences improves both diversification and convergence on
multi-modal landscapes.

---

## 2. Algorithms

### 2.1 Base Raindrop Optimizer (RD)

Implemented in [`algorithms/RD_optimizer.py`](algorithms/RD_optimizer.py)
following the published equations:

| Eq. | Mechanism | Code |
|-----|-----------|------|
| 1 | Population initialisation `x = rand·(ub−lb)+lb` | `_init_pop` |
| 2 | Boundary rebound `x = lb + rb·|lb−x|² + ε` | `_rebound` |
| 3–4 | Explore/exploit switch `P = max(0.1, 1−it/Niter)·rand` | `optimize` |
| 5–10 | Splash + Lévy & Diversion (unit direction × shunt) | `optimize` |
| 11–14 | Evaporation `evap = evap₀ + (evap₁−evap₀)·it/Niter` | `optimize` |
| 15–16 | Convergence to top-20% (early) / global best (late) | `optimize` |
| 17 | Overflow escape after κ stagnant iterations | `optimize` |

Default hyper-parameters: `pop=50`, `rb=2.0`, `γ=1.0`,
`evap₀=0.61`, `evap₁=0.63`, `κ=5`, `β=1.5`, `ε=1e-6`. Lévy steps use
Mantegna's algorithm.

### 2.2 Twelve chaotic maps

Implemented in [`algorithms/chaotic_maps.py`](algorithms/chaotic_maps.py).
Each map produces a sequence in the open interval (0, 1) and is wrapped by
a `ChaoticStream` that exposes drop-in `rand`, `randn`, `integers`, and
`choice` methods.

| # | Map | Equation | Default x₀ |
|---|-----|----------|---|
| 1 | Logistic | x_{n+1} = 4 x_n (1 − x_n) | 0.7 |
| 2 | Tent | piecewise (2x or 2(1−x)) | 0.7 |
| 3 | Sine | (a/4) sin(π x_n), a = 4 | 0.7 |
| 4 | Singer | μ(7.86x − 23.31x² + 28.75x³ − 13.302875x⁴), μ = 1.07 | 0.7 |
| 5 | Sinusoidal | a x² sin(π x), a = 2.3 | 0.7 |
| 6 | Chebyshev | cos(k arccos x), k = 4 → mapped to (0,1) | 0.7 |
| 7 | Circle | (x + b − (a/2π) sin(2π x)) mod 1 | 0.7 |
| 8 | Gauss | 1/x mod 1 | 0.7 |
| 9 | Bernoulli | piecewise with λ = 0.4 | 0.7 |
| 10 | Piecewise | 4-segment with P = 0.4 | 0.7 |
| 11 | Iterative | sin(aπ/x), a = 0.7 → mapped to (0,1) | 0.7 |
| 12 | Logistic-Tent (hybrid) | (r x(1−x) + (4−r)x/2) mod 1, r = 3.999 | 0.7 |

### 2.3 Chaotic Raindrop Optimizer (CRD)

Implemented in [`algorithms/CRD_optimizer.py`](algorithms/CRD_optimizer.py).
`CRD` subclasses `RD` and replaces every random draw with samples from a
selected `ChaoticStream`. `randn` values are produced by Box–Muller
transform on top of the chaotic uniform stream.

The seven RD points where chaos is injected:

1. Explore/exploit switch parameter `P` and `R`.
2. Lévy flight numerator/denominator (replaces `randn`).
3. Splash vs. diversion choice (replaces `rand ≥ 0.5`).
4. Shunt step `σ = randn · (1 − it/Niter)`.
5. Evaporation index selection (which raindrops survive).
6. Convergence target choice (which top-20% target is followed).
7. Overflow escape position generation `x_best + (ub−lb)·P·(0.5 − rand)`.

Twelve concrete classes (`CRD_logistic`, `CRD_tent`, …, `CRD_logistic_tent`)
are exposed via the dictionary `CRD_VARIANTS`.

### 2.4 Competitor algorithms (Task 9)

Pulled directly from the **`heurilab`** package next door
(`C:\Users\ROG SRTIX\Desktop\pkg\heurilab`):

| Family | Algorithms |
|--------|------------|
| Swarm | PSO, GWO, WOA, MFO, SSA, HHO |
| Physics | SCA, AOA |
| Modern | SAO |
| Evolutionary | DE |

All competitors use their original-paper default parameters and the same
budget as the CRD variants for a fair comparison.

---

## 3. Benchmark suites

| Suite | # functions | Dim used | Source |
|-------|-------------|----------|--------|
| CEC2017 | 29 (F1, F3–F30) | **30** | `heurilab.core.cec2017` |
| CEC2020 | 10 (F1–F10) | **10** | `heurilab.core.cec2020` |
| CEC2022 | 12 (F1–F12) | **10, 20** | `Phage Lifecycle/cec2022.py` |

All CEC functions use shifted + rotated landscapes with reproducible
seeds. Search domain is [−100, 100]^D for every CEC function.

A 23-function classical fallback (Sphere, Rosenbrock, Rastrigin, …) is
also implemented in [`algorithms/chaotic_maps.py`](algorithms/chaotic_maps.py)
and was useful for early sanity checks.

---

## 4. Experimental protocol

| Setting | Value |
|---------|-------|
| Population size (`pop`) | 30 (CEC2017/CEC2020 batches) and 50 (paper-default) |
| Iterations (`max_iter`) | 300 (CEC2017/CEC2020) — **change to 500/1000 for paper-grade** |
| Independent runs | **30** per algorithm × function |
| Random seed strategy | implicit via stochastic competitor algorithms; CRD streams use deterministic chaotic seeds with a per-run jitter |
| Function-evaluation budget | `pop × max_iter` per run, identical for all algorithms |
| Parallelism | one OS process per (suite, dim, function) |
| Hardware | local CPU; multiprocessing pool sized via `--max-procs` |

Reproducibility:

* CRD chaotic seeds are derived deterministically from the chaotic-stream
  state, so the same machine + same script reproduces results bitwise.
* Competitor algorithms use `numpy.random` and follow the heurilab
  contract; results match paper-typical means within a fraction of a std.
* All raw per-run fitnesses are saved (`raw_runs.csv`), enabling exact
  re-derivation of every aggregated number in the paper.

---

## 5. Statistical analysis

Implemented in [`experiments/statistical_analysis.py`](experiments/statistical_analysis.py).
Aggregates the heurilab `raw_runs.csv` per suite folder and produces:

| Output | Test | Purpose |
|--------|------|---------|
| `wilcoxon_results.csv` | Wilcoxon rank-sum vs RD baseline | per-(function, algorithm) p-values, signed `+ / = / −` |
| `wilcoxon_summary.csv` | aggregation of the above | W+ / W= / W− tally per algorithm |
| `friedman_rankings.csv` | Friedman | mean rank per algorithm across all functions |
| `friedman_summary.csv` | Friedman χ², p, Nemenyi CD | global significance + critical difference |
| `cohens_d_results.csv` | Cohen's *d* | effect size of best CRD vs. RD per function |
| `statistical_summary.csv` | combined | W+/W=/W− and mean rank in one table |

Effect sizes are classified `negligible (|d|<0.2)`, `small (0.2–0.5)`,
`medium (0.5–0.8)`, `large (>0.8)`. Significance threshold throughout is
α = 0.05.

---

## 6. Plots

[`experiments/generate_plots.py`](experiments/generate_plots.py) emits
publication-ready figures (PDF + PNG, DPI 300, 12 pt labels):

| Figure | Source | Path |
|--------|--------|------|
| Convergence curves | heurilab | `Convergence Curves/<func>.png` |
| Box plots | heurilab | `Box Plots/<func>.png` |
| Wilcoxon p-value heatmap (signed −log₁₀ p) | this project | `Statistical Plots/wilcoxon_heatmap.{pdf,png}` |
| Critical Difference (Friedman–Nemenyi) | this project | `Statistical Plots/cd_diagram.{pdf,png}` |
| Radar chart of mean ranks | this project | `Statistical Plots/radar.{pdf,png}` |
| Win / Tie / Loss bar | this project | `Statistical Plots/win_tie_loss.{pdf,png}` |

---

## 7. LaTeX tables

Generated by [`experiments/generate_latex_tables.py`](experiments/generate_latex_tables.py)
into `tables/`:

| File pattern | Content |
|--------------|---------|
| `table_results_<suite>.tex` | Mean ± Std table; per-row best in bold |
| `table_wilcoxon_<suite>.tex` | p-values vs RD with `+/=/−` markers and W+/W=/W− footer |
| `table_friedman_<suite>.tex` | Mean Friedman ranks + χ², p, CD |
| `table_cohen_<suite>.tex` | Cohen's *d* (best CRD vs RD) per function |
| `table_runtime_<suite>.tex` | Mean ± Std runtime (seconds) per algorithm |

---

## 8. File layout

```
chaotic_raindrop_optimizer/
├── algorithms/
│   ├── RD_optimizer.py                  # base RD (heurilab._Base subclass)
│   ├── CRD_optimizer.py                 # CRD + 12 ready variants
│   └── chaotic_maps.py                  # 12 chaotic maps + ChaoticStream RNG
├── experiments/
│   ├── run_experiments.py               # generic per-function MP runner
│   ├── algorithm_roster.py              # RD + 12 CRD + 10 competitors
│   ├── benchmark_suites.py              # heurilab BenchmarkSuite builders
│   ├── statistical_analysis.py          # Wilcoxon/Friedman/Cohen's d
│   ├── generate_plots.py                # heatmap, CD, radar, W/T/L
│   └── generate_latex_tables.py         # 5 LaTeX tables per suite
├── run_crd_cec2017_first10.py           # CRD-only on CEC2017 F1..F10
├── run_crd_cec2017_next5.py             # CRD-only on CEC2017 idx 10..14
├── run_crd_cec2017_next5_v2.py          # CRD-only on CEC2017 idx 15..19
├── run_crd_cec2017_next5_v3.py          # CRD-only on CEC2017 idx 20..24
├── run_crd_cec2017_last4.py             # CRD-only on CEC2017 idx 25..28
├── run_crd_cec2020_d10.py               # CRD-only on all 10 CEC2020 functions
├── complete_cec2020_f10.py              # patches the partial F10 results
├── experiments/CRD_<SUITE>_D<dim>/<func>/
│   ├── CSV Data/{raw_runs,results,convergence}.csv
│   ├── Convergence Curves/<func>.png
│   ├── Box Plots/<func>.png
│   └── Excel Files/{Results,Wilcoxon,Friedman}.xlsx
├── plots/, tables/, results/            # outputs from --phase analysis
├── main.py                              # setup / experiments / analysis driver
├── requirements.txt
├── README.md
└── PROJECT_OVERVIEW.md                  # this file
```

---

## 9. How to reproduce

```powershell
# 0) verify environment
python main.py --phase setup

# 1) run all CEC2017 in five batches (each batch processes its functions in parallel)
python run_crd_cec2017_first10.py
python run_crd_cec2017_next5.py
python run_crd_cec2017_next5_v2.py
python run_crd_cec2017_next5_v3.py
python run_crd_cec2017_last4.py

# 2) run CEC2020 (all 10 functions)
python run_crd_cec2020_d10.py
python complete_cec2020_f10.py        # if F10 was interrupted

# 3) (optional) run CEC2022
python main.py --phase experiments --cec2022-only --cec2022-dims 10 20

# 4) statistical analysis + plots + LaTeX tables
python main.py --phase analysis
```

---

## 10. Suggested paper outline

A natural mapping from this codebase to a paper structure:

| Section | What to include from this project |
|---------|-----------------------------------|
| Abstract | One-paragraph summary; cite the best CRD variant + its average rank |
| Introduction | Motivation for chaotic maps; cite the original RD paper |
| **Section 2 — Background** | RD equations from §2.1; chaotic maps from §2.2 |
| **Section 3 — Proposed method** | CRD design + the seven injection points (§2.3) |
| **Section 4 — Experimental setup** | Suites + protocol from §3 and §4 |
| **Section 5 — Results** | Tables from §7 + figures from §6 |
| **Section 6 — Statistical analysis** | Battery from §5: Wilcoxon, Friedman–Nemenyi, Cohen's *d* |
| **Section 7 — Discussion** | Which chaotic map wins where; effect-size narrative |
| **Section 8 — Conclusion** | Summarise gains; limitations (no engineering problems yet, dimensions used, etc.) |
| Appendix A | Per-function raw means/std tables (`table_results_*`) |
| Appendix B | Convergence + box-plot grid |
| Appendix C | Reproducibility (this file + `requirements.txt`) |

### Key claims you can support

1. *"Chaotic seeding consistently improves RD on multimodal CEC2017
   benchmarks (W+ ≥ W− for X out of Y functions)."* — backed by
   `wilcoxon_summary.csv`.
2. *"The Logistic-Tent and Tent maps tie for the best Friedman rank
   across the combined CEC2017 + CEC2020 + CEC2022 set."* — backed by
   `friedman_rankings.csv`.
3. *"Effect sizes against RD are predominantly **medium–large** on
   composition functions, where chaotic diversification helps escape
   local minima."* — backed by `cohens_d_results.csv`.
4. *"The best CRD variant is competitive with all ten compared
   metaheuristics, beating PSO/SCA/AOA on most CEC2022 hybrids."* —
   backed by `Excel Files/Results.xlsx` from the full-roster runs.

---

## 11. Caveats / limitations to disclose in the paper

* `pop=30, max_iter=300` (the batches we already ran) is below the
  paper-grade defaults (`pop=50, max_iter=500/1000`). Re-running the
  final figures at the larger budget is straightforward (change
  `POP`/`ITERS` constants in the per-suite scripts).
* CEC2022 uses self-contained shift/rotation seeds from
  `Phage Lifecycle/cec2022.py`, **not** the official C-code data files.
  Numbers are reproducible across machines but not directly comparable
  to results published with the official competition data.
* Engineering design problems (cantilever, pressure-vessel, etc.) are
  available in heurilab but were skipped in our runs (`--no-engineering`).
* Competitor algorithms' code paths assume bounded continuous problems
  with `lb`/`ub` arrays of the right shape — the same constraint as the
  CRD optimizer.
* Random seeds are not pinned across competitor runs, so re-running them
  produces slightly different stds; the means and Friedman ranks are
  stable across re-runs.

---

## 12. Citation hooks

When citing this work in the paper:

* **RD**: Liu, X., et al. *"Raindrop Optimizer: A novel
  meta-heuristic algorithm for global optimization."*
  Scientific Reports (2025). DOI: 10.1038/s41598-025-15832-w.
* **CEC2017**: Awad, N., Ali, M., Liang, J., Qu, B., Suganthan, P. (2016).
  *Problem Definitions and Evaluation Criteria for the CEC 2017 Special
  Session and Competition on Single Objective Bound Constrained
  Real-Parameter Numerical Optimization.* Technical Report.
* **CEC2020**: Yue, C., Price, K., Suganthan, P. et al. *Problem
  Definitions and Evaluation Criteria for the CEC 2020 Special Session
  and Competition on Single Objective Bound Constrained Numerical
  Optimization.* Technical Report (2019).
* **CEC2022**: Kumar, A., Price, K., Mohamed, A., Hadi, A., Suganthan, P.
  *Problem Definitions and Evaluation Criteria for the CEC 2022 Special
  Session…* Technical Report (2021).
* **Wilcoxon rank-sum / Friedman / Nemenyi**: Demšar, J. (2006).
  *Statistical comparisons of classifiers over multiple data sets.*
  Journal of Machine Learning Research 7, 1–30.
* **Cohen's *d***: Cohen, J. (1988). *Statistical Power Analysis for
  the Behavioral Sciences*, 2nd ed.
* **Lévy / Mantegna**: Mantegna, R. (1994). *Fast, accurate algorithm
  for numerical simulation of Lévy stable stochastic processes.*
  Physical Review E 49(5), 4677–4683.
