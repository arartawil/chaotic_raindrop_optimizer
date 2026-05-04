"""
Raindrop Optimizer (RD) — base implementation, heurilab-compatible.

Reference: Scientific Reports (2025) — DOI: 10.1038/s41598-025-15832-w.
The optimizer subclasses :class:`heurilab.algorithms.base._Base` so it can be
plugged directly into ``heurilab.run_experiment``.
"""

from __future__ import annotations

from math import gamma
from typing import List

import numpy as np

from heurilab.algorithms.base import _Base


# Default RD hyper-parameters (per the original paper)
DEFAULTS = dict(
    rb=2.0,
    gamma_coef=1.0,
    evap_initial=0.61,
    evap_final=0.63,
    kappa=5,
    beta=1.5,
    epsilon=1e-6,
)


def _levy_step(dim: int, beta: float, rng) -> np.ndarray:
    sigma = (
        gamma(1.0 + beta) * np.sin(np.pi * beta / 2.0)
        / (gamma((1.0 + beta) / 2.0) * beta * 2.0 ** ((beta - 1.0) / 2.0))
    ) ** (1.0 / beta)
    u = rng.standard_normal(dim) * sigma
    v = rng.standard_normal(dim)
    return u / (np.abs(v) ** (1.0 / beta) + 1e-12)


class RD(_Base):
    """Base Raindrop Optimizer.

    Parameters mirror the heurilab `_Base` constructor; RD-specific knobs
    have sensible defaults pulled from the original paper.
    """

    # Allow RD to receive extra hyperparameters from heurilab if desired
    _hyperparams = DEFAULTS

    def __init__(self, pop_size, dim, lb, ub, max_iter, obj_func, **rd_kwargs):
        super().__init__(pop_size, dim, lb, ub, max_iter, obj_func)
        for k, v in {**DEFAULTS, **rd_kwargs}.items():
            setattr(self, k, v)

    # ------------------------------------------------------------------ stream
    def _make_rng(self):
        """Return an object with .random/.standard_normal/.integers/.choice."""
        return _NumpyRNG()

    # ------------------------------------------------------------------ rebound
    def _rebound(self, x: np.ndarray) -> np.ndarray:
        out = np.array(x, dtype=float, copy=True)
        below = out < self.lb
        above = out > self.ub
        if np.any(below):
            diff = np.abs(self.lb[below] - out[below])
            out[below] = self.lb[below] + self.rb * diff ** 2 + self.epsilon
        if np.any(above):
            diff = np.abs(self.ub[above] - out[above])
            out[above] = self.ub[above] - self.rb * diff ** 2 + self.epsilon
        out = np.clip(out, self.lb, self.ub)
        out = np.where(np.isfinite(out), out, (self.lb + self.ub) / 2.0)
        return out

    # ------------------------------------------------------------------ optimize
    def optimize(self):
        rng = self._make_rng()
        dim = self.dim
        n_pop = self.pop_size

        # Eq. 1 — population init
        positions = rng.random((n_pop, dim)) * (self.ub - self.lb) + self.lb
        fitness = np.array([self._eval(positions[i]) for i in range(n_pop)])

        best_idx = int(np.argmin(fitness))
        best_pos = positions[best_idx].copy()
        best_fit = float(fitness[best_idx])
        prev_best = best_fit

        convergence: List[float] = [best_fit]
        kappa_counter = 0

        for it in range(self.max_iter):
            # Eqs. 3-4 — explore/exploit switch
            P_iter = max(0.1, 1.0 - it / max(1, self.max_iter)) * float(rng.random())
            R = float(rng.random())

            if P_iter > R:
                # ------------------------------------------------ exploration
                new_positions = []
                for i in range(n_pop):
                    a = int(rng.integers(2, 4))  # randi(2,3)
                    neigh_size = max(2, min(n_pop, a + 2))
                    neigh_idx = rng.choice(n_pop, size=neigh_size, replace=False)
                    local_best_idx = neigh_idx[int(np.argmin(fitness[neigh_idx]))]
                    x_local = positions[local_best_idx]
                    for _ in range(a):
                        x_last = positions[i]
                        if float(rng.random()) >= 0.5:
                            # Eq. 5 — splash + Levy
                            step = float(rng.standard_normal()) * _levy_step(dim, self.beta, rng)
                            x_new = x_last + P_iter * step
                        else:
                            # Eqs. 7-10 — diversion
                            dir_vec = x_local - x_last
                            norm = float(np.linalg.norm(dir_vec))
                            if norm < 1e-12:
                                dir_unit = rng.standard_normal(dim)
                                dir_unit /= np.linalg.norm(dir_unit) + 1e-12
                            else:
                                dir_unit = dir_vec / norm
                            shunt = float(rng.standard_normal()) * (1.0 - it / max(1, self.max_iter))
                            x_new = x_last + dir_unit * shunt * (self.ub - self.lb) * 0.1
                        new_positions.append(self._rebound(x_new))

                if not new_positions:
                    convergence.append(best_fit)
                    continue

                arr = np.array(new_positions)
                arr_fit = np.array([self._eval(p) for p in arr])

                # Eqs. 11-14 — evaporation
                evap = self.evap_initial + (self.evap_final - self.evap_initial) * (
                    it / max(1, self.max_iter)
                )
                n_evap = int(np.floor(len(arr) * evap))
                if 0 < n_evap < len(arr):
                    keep = rng.choice(len(arr), size=len(arr) - n_evap, replace=False)
                    arr = arr[keep]
                    arr_fit = arr_fit[keep]

                # μ + λ selection
                combined_pos = np.vstack([positions, arr])
                combined_fit = np.concatenate([fitness, arr_fit])
                order = np.argsort(combined_fit)[:n_pop]
                positions = combined_pos[order]
                fitness = combined_fit[order]
            else:
                # ----------------------------------------------- exploitation
                order = np.argsort(fitness)
                if it < self.max_iter / 2:
                    top_n = max(1, int(np.ceil(0.2 * n_pop)))
                    targets = positions[order[:top_n]]
                else:
                    targets = best_pos[np.newaxis, :]

                new_pos = np.empty_like(positions)
                for i in range(n_pop):
                    t_idx = int(rng.integers(0, len(targets)))
                    t = targets[t_idx]
                    new_pos[i] = self._rebound(
                        positions[i]
                        + self.gamma_coef * (t - positions[i]) * float(rng.random())
                    )
                new_fit = np.array([self._eval(new_pos[i]) for i in range(n_pop)])
                better = new_fit < fitness
                positions[better] = new_pos[better]
                fitness[better] = new_fit[better]

            cur_idx = int(np.argmin(fitness))
            if fitness[cur_idx] < best_fit:
                best_fit = float(fitness[cur_idx])
                best_pos = positions[cur_idx].copy()

            # Eq. 17 — overflow escape
            if abs(best_fit - prev_best) < 1e-12:
                kappa_counter += 1
            else:
                kappa_counter = 0
            prev_best = best_fit

            if kappa_counter >= self.kappa:
                escape = best_pos + (self.ub - self.lb) * P_iter * (
                    0.5 - rng.random((5, dim))
                )
                escape = np.array([self._rebound(p) for p in escape])
                e_fit = np.array([self._eval(p) for p in escape])
                worst = np.argsort(fitness)[-len(escape):]
                positions[worst] = escape
                fitness[worst] = e_fit
                kappa_counter = 0

            convergence.append(best_fit)

        return best_pos, best_fit, convergence


# Tiny RNG facade so RD can be subclassed by chaotic variants without
# changing call sites.
class _NumpyRNG:
    """Minimal RNG facade backed by numpy.random."""

    def random(self, size=None):
        if size is None:
            return float(np.random.random())
        return np.random.random(size)

    def standard_normal(self, size=None):
        if size is None:
            return float(np.random.standard_normal())
        return np.random.standard_normal(size)

    def integers(self, low, high=None):
        if high is None:
            low, high = 0, int(low)
        return int(np.random.randint(low, high))

    def choice(self, a, size=None, replace=True):
        if size is None:
            return np.random.choice(a, replace=replace)
        return np.random.choice(a, size=size, replace=replace)
