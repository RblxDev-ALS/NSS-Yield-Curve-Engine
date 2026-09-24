"""Nelson-Siegel (NS) and Nelson-Siegel-Svensson (NSS) term-structure models.

Conventions used throughout the package
---------------------------------------
* Maturities ``tau`` are in **years**.
* Rates are in **percent** (``4.25`` means 4.25%).
* The NSS curve is a **continuously compounded zero-coupon** curve::

      z(τ) = β0
           + β1 · (1 - e^{-λ1 τ}) / (λ1 τ)
           + β2 · [(1 - e^{-λ1 τ}) / (λ1 τ) - e^{-λ1 τ}]
           + β3 · [(1 - e^{-λ2 τ}) / (λ2 τ) - e^{-λ2 τ}]

  ``λ1`` and ``λ2`` are *decay rates* (1/years), as in Diebold & Li (2006).
  Svensson's original paper uses time constants ``τ_i = 1/λ_i`` instead.
* Setting ``β3 = 0`` recovers the 3-factor Nelson-Siegel model.

Interpretation of the factors
-----------------------------
* ``β0`` - long-run **level**: ``z(τ) → β0`` as ``τ → ∞``.
* ``β1`` - **slope**: ``z(0) = β0 + β1``, so the long-minus-short spread is ``-β1``.
* ``β2`` - **curvature** (hump/trough centred near ``τ ≈ 1.79/λ1``).
* ``β3`` - second curvature term that lets the long end bend independently.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray: TypeAlias = NDArray[np.float64]

#: Diebold & Li (2006) decay parameter, 0.0609 per month converted to per year.
#: It places the maximum of the curvature loading at 30 months.
DIEBOLD_LI_LAMBDA = 0.0609 * 12

#: Below this |x| the loadings are evaluated with a Taylor series to avoid
#: catastrophic cancellation in ``1 - exp(-x)``.
_SERIES_CUTOFF = 1e-4

PARAM_NAMES = ("beta0", "beta1", "beta2", "beta3", "lambda1", "lambda2")


def slope_loading(x: ArrayLike) -> FloatArray:
    """Return ``(1 - e^{-x}) / x`` evaluated stably, with the limit 1 at ``x = 0``."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    small = np.abs(x) < _SERIES_CUTOFF
    xs = x[small]
    out[small] = 1.0 - xs / 2.0 + xs**2 / 6.0
    xl = x[~small]
    out[~small] = -np.expm1(-xl) / xl
    return out


def curvature_loading(x: ArrayLike) -> FloatArray:
    """Return ``(1 - e^{-x}) / x - e^{-x}`` evaluated stably, with the limit 0 at ``x = 0``."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    small = np.abs(x) < _SERIES_CUTOFF
    xs = x[small]
    out[small] = xs / 2.0 - xs**2 / 3.0 + xs**3 / 8.0
    xl = x[~small]
    out[~small] = -np.expm1(-xl) / xl - np.exp(-xl)
    return out


def ns_loadings(tau: ArrayLike, lambda1: float) -> FloatArray:
    """Nelson-Siegel factor loadings, shape ``(len(tau), 3)``: level, slope, curvature."""
    tau = np.atleast_1d(np.asarray(tau, dtype=float))
    x = lambda1 * tau
    return np.column_stack([np.ones_like(tau), slope_loading(x), curvature_loading(x)])


def nss_loadings(tau: ArrayLike, lambda1: float, lambda2: float) -> FloatArray:
    """Svensson factor loadings, shape ``(len(tau), 4)``."""
    tau = np.atleast_1d(np.asarray(tau, dtype=float))
    return np.column_stack([ns_loadings(tau, lambda1), curvature_loading(lambda2 * tau)])


def curvature_peak(lam: float) -> float:
    """Maturity (years) at which the curvature loading for decay ``lam`` peaks.

    Solves ``d/dτ [(1-e^{-λτ})/(λτ) - e^{-λτ}] = 0``; the solution is ``x*/λ``
    with ``x* ≈ 1.7933`` independent of ``λ``.
    """
    return _CURVATURE_PEAK_X / lam


def _find_curvature_peak_x() -> float:
    # Root of (x^2 + x + 1) e^{-x} = 1 (the derivative condition), solved once.
    from scipy.optimize import brentq

    return float(brentq(lambda x: (x * x + x + 1.0) * np.exp(-x) - 1.0, 0.5, 5.0))


_CURVATURE_PEAK_X = _find_curvature_peak_x()


def nss_zero(
    tau: ArrayLike,
    beta0: float,
    beta1: float,
    beta2: float,
    beta3: float,
    lambda1: float,
    lambda2: float,
) -> FloatArray:
    """Vectorised NSS zero rate (percent, continuous compounding)."""
    return nss_loadings(tau, lambda1, lambda2) @ np.array([beta0, beta1, beta2, beta3])


def nss_forward(
    tau: ArrayLike,
    beta0: float,
    beta1: float,
    beta2: float,
    beta3: float,
    lambda1: float,
    lambda2: float,
) -> FloatArray:
    """Instantaneous forward rate ``f(τ) = d[τ z(τ)]/dτ`` (percent)."""
    tau = np.atleast_1d(np.asarray(tau, dtype=float))
    e1 = np.exp(-lambda1 * tau)
    e2 = np.exp(-lambda2 * tau)
    return beta0 + beta1 * e1 + beta2 * lambda1 * tau * e1 + beta3 * lambda2 * tau * e2


@dataclass(frozen=True)
class NSSCurve:
    """A calibrated Nelson-Siegel-Svensson zero curve.

    A Nelson-Siegel curve is represented with ``beta3 = 0`` (``lambda2`` is then
    irrelevant). All methods accept scalars or arrays of maturities in years and
    return numpy arrays.
    """

    beta0: float
    beta1: float
    beta2: float
    beta3: float = 0.0
    lambda1: float = DIEBOLD_LI_LAMBDA
    lambda2: float = 0.2

    def __post_init__(self) -> None:
        if not (self.lambda1 > 0 and self.lambda2 > 0):
            raise ValueError("decay parameters lambda1 and lambda2 must be positive")

    # ------------------------------------------------------------------ construction
    @classmethod
    def nelson_siegel(
        cls, beta0: float, beta1: float, beta2: float, lambda1: float = DIEBOLD_LI_LAMBDA
    ) -> NSSCurve:
        """Build a 3-factor Nelson-Siegel curve."""
        return cls(beta0, beta1, beta2, 0.0, lambda1, lambda1)

    @classmethod
    def from_array(cls, params: ArrayLike) -> NSSCurve:
        """Build from ``[beta0, beta1, beta2, beta3, lambda1, lambda2]``."""
        p = np.asarray(params, dtype=float).ravel()
        if p.size != 6:
            raise ValueError(f"expected 6 parameters, got {p.size}")
        return cls(*(float(v) for v in p))

    @classmethod
    def from_mapping(cls, row: Any) -> NSSCurve:
        """Build from any mapping / pandas row with the :data:`PARAM_NAMES` keys."""
        return cls(*(float(row[name]) for name in PARAM_NAMES))

    def as_array(self) -> FloatArray:
        return np.array([getattr(self, n) for n in PARAM_NAMES], dtype=float)

    def as_dict(self) -> dict[str, float]:
        return asdict(self)

    @property
    def betas(self) -> FloatArray:
        return np.array([self.beta0, self.beta1, self.beta2, self.beta3])

    # ------------------------------------------------------------------ curve values
    def zero(self, tau: ArrayLike) -> FloatArray:
        """Continuously compounded zero rate in percent."""
        return nss_zero(tau, *self.as_array())

    def forward(self, tau: ArrayLike) -> FloatArray:
        """Instantaneous forward rate in percent."""
        return nss_forward(tau, *self.as_array())

    def discount(self, tau: ArrayLike) -> FloatArray:
        """Discount factor ``D(τ) = exp(-z(τ) τ / 100)``."""
        tau = np.atleast_1d(np.asarray(tau, dtype=float))
        return np.exp(-self.zero(tau) * tau / 100.0)

    def forward_rate(self, t1: ArrayLike, t2: ArrayLike) -> FloatArray:
        """Continuously compounded forward rate between ``t1`` and ``t2`` (percent)."""
        t1 = np.atleast_1d(np.asarray(t1, dtype=float))
        t2 = np.atleast_1d(np.asarray(t2, dtype=float))
        if np.any(t2 <= t1):
            raise ValueError("t2 must be strictly greater than t1")
        return (self.zero(t2) * t2 - self.zero(t1) * t1) / (t2 - t1)

    def par_yield(self, tau: ArrayLike, freq: int = 2) -> FloatArray:
        """Par yield (percent, compounded ``freq`` times a year).

        Maturities of one year or less are treated as zero-coupon bills, quoted
        as a bond-equivalent yield ``freq·(D^{-1/(freq·τ)} - 1)``. Longer
        maturities are coupon bonds paying ``c/freq`` on the schedule
        ``τ, τ - 1/freq, ...`` (> 0), whose price equals par when
        ``c = freq · (1 - D(τ)) / Σ D(t_i)``. This mirrors how the Treasury
        constant-maturity (CMT) series are quoted.
        """
        tau_arr = np.atleast_1d(np.asarray(tau, dtype=float))
        out = np.empty_like(tau_arr)
        for i, t in enumerate(tau_arr):
            if t <= 0:
                out[i] = self.zero(0.0)[0]
            elif t <= 1.0:
                d = self.discount(t)[0]
                out[i] = 100.0 * freq * (d ** (-1.0 / (freq * t)) - 1.0)
            else:
                n = int(np.floor(t * freq + 1e-9))
                times = t - np.arange(n) / freq
                annuity = self.discount(times).sum() / freq
                out[i] = 100.0 * (1.0 - self.discount(t)[0]) / annuity
        return out

    def evaluate(self, tau: ArrayLike, measure: str = "zero") -> FloatArray:
        """Evaluate the curve on a given ``measure``: ``zero``, ``par`` or ``forward``."""
        if measure == "zero":
            return self.zero(tau)
        if measure == "par":
            return self.par_yield(tau)
        if measure == "forward":
            return self.forward(tau)
        raise ValueError(f"unknown measure {measure!r}; use 'zero', 'par' or 'forward'")

    # ------------------------------------------------------------------ summaries
    @property
    def short_rate(self) -> float:
        """Instantaneous short rate ``z(0) = β0 + β1``."""
        return self.beta0 + self.beta1

    @property
    def long_rate(self) -> float:
        """Asymptotic long rate ``z(∞) = β0``."""
        return self.beta0

    def spread(self, long: float = 10.0, short: float = 2.0, measure: str = "zero") -> float:
        """Model-implied spread ``y(long) - y(short)`` in percentage points."""
        vals = self.evaluate([short, long], measure)
        return float(vals[1] - vals[0])
