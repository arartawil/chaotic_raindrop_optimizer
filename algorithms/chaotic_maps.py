"""
Twelve chaotic maps for the Chaotic Raindrop Optimizer.

Each map produces a sequence of values in (0, 1) given an initial seed x0.
A small epsilon is added to keep values strictly inside (0, 1) so the values
can be used as drop-in replacements for ``numpy.random.rand``.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12


def _clip01(x: float) -> float:
    """Clip a value into the open interval (0, 1)."""
    if not np.isfinite(x):
        return 0.5
    if x <= 0.0:
        return EPS
    if x >= 1.0:
        return 1.0 - EPS
    return float(x)


def logistic_map(x0: float = 0.7, n: int = 1000) -> np.ndarray:
    seq = np.empty(n, dtype=float)
    x = _clip01(x0)
    for i in range(n):
        x = 4.0 * x * (1.0 - x)
        x = _clip01(x)
        seq[i] = x
    return seq


def tent_map(x0: float = 0.7, n: int = 1000) -> np.ndarray:
    seq = np.empty(n, dtype=float)
    x = _clip01(x0)
    for i in range(n):
        x = 2.0 * x if x < 0.5 else 2.0 * (1.0 - x)
        x = _clip01(x)
        seq[i] = x
    return seq


def sine_map(x0: float = 0.7, n: int = 1000, a: float = 4.0) -> np.ndarray:
    seq = np.empty(n, dtype=float)
    x = _clip01(x0)
    for i in range(n):
        x = (a / 4.0) * np.sin(np.pi * x)
        x = _clip01(x)
        seq[i] = x
    return seq


def singer_map(x0: float = 0.7, n: int = 1000, mu: float = 1.07) -> np.ndarray:
    seq = np.empty(n, dtype=float)
    x = _clip01(x0)
    for i in range(n):
        x = mu * (7.86 * x - 23.31 * x**2 + 28.75 * x**3 - 13.302875 * x**4)
        x = _clip01(x)
        seq[i] = x
    return seq


def sinusoidal_map(x0: float = 0.7, n: int = 1000, a: float = 2.3) -> np.ndarray:
    seq = np.empty(n, dtype=float)
    x = _clip01(x0)
    for i in range(n):
        x = a * (x**2) * np.sin(np.pi * x)
        x = _clip01(x)
        seq[i] = x
    return seq


def chebyshev_map(x0: float = 0.7, n: int = 1000, k: float = 4.0) -> np.ndarray:
    """Chebyshev map. Native range is (-1, 1); map to (0, 1) via (x+1)/2."""
    seq = np.empty(n, dtype=float)
    x = 2.0 * _clip01(x0) - 1.0
    for i in range(n):
        x = np.cos(k * np.arccos(np.clip(x, -1.0 + EPS, 1.0 - EPS)))
        seq[i] = _clip01((x + 1.0) / 2.0)
    return seq


def circle_map(
    x0: float = 0.7, n: int = 1000, a: float = 0.5, b: float = 0.2
) -> np.ndarray:
    seq = np.empty(n, dtype=float)
    x = _clip01(x0)
    for i in range(n):
        x = (x + b - (a / (2.0 * np.pi)) * np.sin(2.0 * np.pi * x)) % 1.0
        x = _clip01(x)
        seq[i] = x
    return seq


def gauss_map(x0: float = 0.7, n: int = 1000) -> np.ndarray:
    seq = np.empty(n, dtype=float)
    x = _clip01(x0)
    for i in range(n):
        if x == 0.0:
            x = EPS
        x = (1.0 / x) % 1.0
        x = _clip01(x)
        seq[i] = x
    return seq


def bernoulli_map(x0: float = 0.7, n: int = 1000, lam: float = 0.4) -> np.ndarray:
    seq = np.empty(n, dtype=float)
    x = _clip01(x0)
    for i in range(n):
        if x < lam:
            x = x / lam
        else:
            x = (x - lam) / (1.0 - lam)
        x = _clip01(x)
        seq[i] = x
    return seq


def piecewise_map(x0: float = 0.7, n: int = 1000, P: float = 0.4) -> np.ndarray:
    seq = np.empty(n, dtype=float)
    x = _clip01(x0)
    for i in range(n):
        if 0.0 <= x < P:
            x = x / P
        elif P <= x < 0.5:
            x = (x - P) / (0.5 - P)
        elif 0.5 <= x < 1.0 - P:
            x = (1.0 - P - x) / (0.5 - P)
        else:
            x = (1.0 - x) / P
        x = _clip01(x)
        seq[i] = x
    return seq


def iterative_map(x0: float = 0.7, n: int = 1000, a: float = 0.7) -> np.ndarray:
    """Iterative chaotic map. Native range (-1, 1), mapped to (0, 1)."""
    seq = np.empty(n, dtype=float)
    x = 2.0 * _clip01(x0) - 1.0
    for i in range(n):
        if x == 0.0:
            x = EPS
        x = np.sin((a * np.pi) / x)
        seq[i] = _clip01((x + 1.0) / 2.0)
    return seq


def logistic_tent_map(x0: float = 0.7, n: int = 1000, r: float = 3.999) -> np.ndarray:
    """Hybrid Logistic-Tent map."""
    seq = np.empty(n, dtype=float)
    x = _clip01(x0)
    for i in range(n):
        if x < 0.5:
            x = (r * x * (1.0 - x) + (4.0 - r) * x / 2.0) % 1.0
        else:
            x = (r * x * (1.0 - x) + (4.0 - r) * (1.0 - x) / 2.0) % 1.0
        x = _clip01(x)
        seq[i] = x
    return seq


CHAOTIC_MAPS = {
    "logistic": logistic_map,
    "tent": tent_map,
    "sine": sine_map,
    "singer": singer_map,
    "sinusoidal": sinusoidal_map,
    "chebyshev": chebyshev_map,
    "circle": circle_map,
    "gauss": gauss_map,
    "bernoulli": bernoulli_map,
    "piecewise": piecewise_map,
    "iterative": iterative_map,
    "logistic_tent": logistic_tent_map,
}


def generate_chaotic_sequence(
    map_name: str, x0: float = 0.7, n: int = 1000
) -> np.ndarray:
    """Generate a chaotic sequence by name, normalized to (0, 1)."""
    if map_name not in CHAOTIC_MAPS:
        raise ValueError(
            f"Unknown chaotic map '{map_name}'. Options: {list(CHAOTIC_MAPS)}"
        )
    return CHAOTIC_MAPS[map_name](x0=x0, n=n)


class ChaoticStream:
    """Stateful chaotic sequence generator that refills lazily."""

    def __init__(
        self,
        map_name: str,
        x0: float = 0.7,
        chunk_size: int = 4096,
    ) -> None:
        if map_name not in CHAOTIC_MAPS:
            raise ValueError(f"Unknown chaotic map '{map_name}'")
        self.map_name = map_name
        self.chunk_size = int(chunk_size)
        self._x0 = float(x0)
        self._buffer = generate_chaotic_sequence(map_name, x0=self._x0, n=self.chunk_size)
        self._idx = 0

    def _refill(self) -> None:
        new_seed = float(self._buffer[-1])
        if new_seed in (0.0, 1.0):
            new_seed = 0.7
        self._buffer = generate_chaotic_sequence(
            self.map_name, x0=new_seed, n=self.chunk_size
        )
        self._idx = 0
        self._x0 = new_seed

    def rand(self, size: int | tuple | None = None) -> np.ndarray | float:
        """Return chaotic value(s) in (0, 1) shaped like ``size``."""
        if size is None:
            count = 1
            shape: tuple = ()
        elif isinstance(size, int):
            count = int(size)
            shape = (count,)
        else:
            shape = tuple(int(s) for s in size)
            count = int(np.prod(shape)) if shape else 1

        out = np.empty(count, dtype=float)
        filled = 0
        while filled < count:
            available = len(self._buffer) - self._idx
            take = min(available, count - filled)
            out[filled:filled + take] = self._buffer[self._idx:self._idx + take]
            self._idx += take
            filled += take
            if self._idx >= len(self._buffer):
                self._refill()

        if shape == ():
            return float(out[0])
        return out.reshape(shape)

    def randn(self, size: int | tuple | None = None) -> np.ndarray | float:
        """Standard-normal samples derived from chaotic values via Box-Muller."""
        if size is None:
            count = 1
            shape: tuple = ()
        elif isinstance(size, int):
            count = int(size)
            shape = (count,)
        else:
            shape = tuple(int(s) for s in size)
            count = int(np.prod(shape)) if shape else 1

        n_pairs = (count + 1) // 2
        u1 = np.asarray(self.rand(n_pairs), dtype=float).reshape(-1)
        u2 = np.asarray(self.rand(n_pairs), dtype=float).reshape(-1)
        u1 = np.clip(u1, EPS, 1.0 - EPS)
        r = np.sqrt(-2.0 * np.log(u1))
        theta = 2.0 * np.pi * u2
        z = np.empty(2 * n_pairs, dtype=float)
        z[0::2] = r * np.cos(theta)
        z[1::2] = r * np.sin(theta)
        out = z[:count]
        if shape == ():
            return float(out[0])
        return out.reshape(shape)

    def randint(self, low: int, high: int | None = None) -> int:
        """Single integer in [low, high) using chaotic value."""
        if high is None:
            low, high = 0, int(low)
        u = float(self.rand())
        val = int(low + np.floor(u * (high - low)))
        return min(max(val, low), high - 1)
