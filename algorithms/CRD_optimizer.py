"""
Chaotic Raindrop Optimizer (CRD).

Subclasses the base :class:`RD` and replaces every random number draw with
deterministic samples produced by one of the 12 chaotic maps.  A factory
generates one ready-to-use class per map so heurilab's ``run_experiment``
can reference them like any other algorithm.
"""

from __future__ import annotations

from typing import Dict, Type

import numpy as np

from .RD_optimizer import RD, _NumpyRNG
from .chaotic_maps import CHAOTIC_MAPS, ChaoticStream


class _ChaoticRNG(_NumpyRNG):
    """RNG facade backed by a chaotic map; falls back to numpy for ``choice``."""

    def __init__(self, stream: ChaoticStream):
        self._stream = stream

    def random(self, size=None):
        return self._stream.rand(size)

    def standard_normal(self, size=None):
        return self._stream.randn(size)

    def integers(self, low, high=None):
        if high is None:
            low, high = 0, int(low)
        return int(self._stream.randint(int(low), int(high)))

    def choice(self, a, size=None, replace=True):
        if isinstance(a, (int, np.integer)):
            n = int(a)
            arr = np.arange(n)
        else:
            arr = np.asarray(a)
            n = len(arr)
        if size is None:
            idx = int(np.floor(self._stream.rand() * n))
            return arr[min(max(idx, 0), n - 1)]
        if isinstance(size, int):
            count = size
        else:
            count = int(np.prod(size))

        if replace:
            u = np.asarray(self._stream.rand(count), dtype=float).reshape(-1)
            idx = np.clip(np.floor(u * n).astype(int), 0, n - 1)
        else:
            count = min(count, n)
            keys = np.asarray(self._stream.rand(n), dtype=float).reshape(-1)
            idx = np.argsort(keys)[:count]
        out = arr[idx]
        if isinstance(size, tuple):
            out = out.reshape(size)
        return out


class CRD(RD):
    """Chaotic Raindrop Optimizer (parameterised by a chaotic map)."""

    map_name: str = "logistic"
    x0: float = 0.7
    chunk_size: int = 4096

    def __init__(self, pop_size, dim, lb, ub, max_iter, obj_func, **rd_kwargs):
        # Allow the map to be overridden via kwargs.
        if "chaotic_map" in rd_kwargs:
            self.map_name = rd_kwargs.pop("chaotic_map")
        if "x0" in rd_kwargs:
            self.x0 = float(rd_kwargs.pop("x0"))
        if self.map_name not in CHAOTIC_MAPS:
            raise ValueError(f"Unknown chaotic map '{self.map_name}'")
        super().__init__(pop_size, dim, lb, ub, max_iter, obj_func, **rd_kwargs)

    def _make_rng(self):
        # Slight per-instance jitter so independent runs use different streams
        # (numpy's RNG state changes between calls).
        seed_jitter = float(np.random.random())
        x0 = float(np.clip(0.5 * (self.x0 + seed_jitter), 0.05, 0.95))
        stream = ChaoticStream(self.map_name, x0=x0, chunk_size=self.chunk_size)
        return _ChaoticRNG(stream)


def _make_variant(name: str) -> Type[CRD]:
    """Build a CRD subclass with a fixed map (so its ``__name__`` is unique)."""
    cls_name = f"CRD_{name}"
    return type(cls_name, (CRD,), {"map_name": name})


CRD_logistic = _make_variant("logistic")
CRD_tent = _make_variant("tent")
CRD_sine = _make_variant("sine")
CRD_singer = _make_variant("singer")
CRD_sinusoidal = _make_variant("sinusoidal")
CRD_chebyshev = _make_variant("chebyshev")
CRD_circle = _make_variant("circle")
CRD_gauss = _make_variant("gauss")
CRD_bernoulli = _make_variant("bernoulli")
CRD_piecewise = _make_variant("piecewise")
CRD_iterative = _make_variant("iterative")
CRD_logistic_tent = _make_variant("logistic_tent")


CRD_VARIANTS: Dict[str, Type[CRD]] = {
    "CRD-Logistic": CRD_logistic,
    "CRD-Tent": CRD_tent,
    "CRD-Sine": CRD_sine,
    "CRD-Singer": CRD_singer,
    "CRD-Sinusoidal": CRD_sinusoidal,
    "CRD-Chebyshev": CRD_chebyshev,
    "CRD-Circle": CRD_circle,
    "CRD-Gauss": CRD_gauss,
    "CRD-Bernoulli": CRD_bernoulli,
    "CRD-Piecewise": CRD_piecewise,
    "CRD-Iterative": CRD_iterative,
    "CRD-LogisticTent": CRD_logistic_tent,
}
