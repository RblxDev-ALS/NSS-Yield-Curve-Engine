"""Nelson-Siegel (NS) and Nelson-Siegel-Svensson (NSS) curve mathematics.

Conventions used throughout the package
---------------------------------------
* Maturities (``tau``) are in years.
* Model rates are **continuously compounded zero rates in percent**.
* Market Treasury quotes (FRED ``DGS*`` constant-maturity series) are
  **semi-annual bond-equivalent yields in percent**: bills (<= 1Y) are
  zero-coupon, notes and bonds are par yields. Helpers below convert between
  the two so the model can be fitted either to approximate zero rates or to
  exact par yields.

NSS zero curve::

    y(tau) = b0
           + b1 * f1(l1 tau)
           + b2 * f2(l1 tau)
           + b3 * f2(l2 tau)

    f1(x) = (1 - exp(-x)) / x          (slope loading, 1 -> 0)
    f2(x) = f1(x) - exp(-x)            (hump loading, 0 -> max -> 0)

``b0`` is the long rate, ``b0 + b1`` the instantaneous short rate, so the
model slope (long minus short) is ``-b1``. ``f2`` peaks at ``x ~= 1.7933`` so
the hump driven by ``b2`` sits at ``1.7933 / l1`` years and the ``b3`` hump at
``1.7933 / l2`` years. The classic three-factor NS model is the special case
``b3 = 0``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

# argmax of f2(x); used to translate decay rates into hump locations.
HUMP_ARGMAX = 1.7932821329007607

PARAM_NAMES_NSS = ("Beta0", "Beta1", "Beta2", "Beta3", "Lambda1", "Lambda2")
PARAM_NAMES_NS = ("Beta0", "Beta1", "Beta2", "Lambda1")


# -----------------------------------------------------------------------------
# Loadings
# -----------------------------------------------------------------------------

def slope_loading(x: np.ndarray | float) -> np.ndarray:
    """``(1 - e^-x) / x`` with the removable singularity at 0 handled."""
    x = np.asarray(x, dtype=float)
    small = np.abs(x) < 1e-8
    safe = np.where(small, 1.0, x)
    return np.where(small, 1.0 - 0.5 * x, -np.expm1(-safe) / safe)


def hump_loading(x: np.ndarray | float) -> np.ndarray:
    """``(1 - e^-x) / x - e^-x`` with the removable singularity at 0 handled."""
    x = np.asarray(x, dtype=float)
    small = np.abs(x) < 1e-8
    return np.where(small, 0.5 * x, slope_loading(x) - np.exp(-x))


def slope_loading_deriv(x: np.ndarray | float) -> np.ndarray:
    """d/dx of :func:`slope_loading`."""
    x = np.asarray(x, dtype=float)
    small = np.abs(x) < 1e-6
    safe = np.where(small, 1.0, x)
    return np.where(small, -0.5 + x / 3.0, (np.exp(-safe) - slope_loading(safe)) / safe)


def hump_loading_deriv(x: np.ndarray | float) -> np.ndarray:
    """d/dx of :func:`hump_loading`."""
    x = np.asarray(x, dtype=float)
    return slope_loading_deriv(x) + np.exp(-x)


def loading_matrix(tau: Sequence[float] | np.ndarray,
                   lambda1: float | np.ndarray,
                   lambda2: float | np.ndarray | None = None) -> np.ndarray:
    """Factor loading matrix.

    ``tau`` has shape ``(n,)``. ``lambda1`` / ``lambda2`` may be scalars or
    arrays of identical shape ``S`` (e.g. a grid of candidate decay rates);
    the result has shape ``S + (n, k)`` where ``k`` is 3 (NS, ``lambda2`` is
    None) or 4 (NSS).
    """
    tau = np.asarray(tau, dtype=float)
    l1 = np.asarray(lambda1, dtype=float)[..., None]
    x1 = l1 * tau
    cols = [np.ones_like(x1), slope_loading(x1), hump_loading(x1)]
    if lambda2 is not None:
        l2 = np.asarray(lambda2, dtype=float)[..., None]
        cols.append(hump_loading(l2 * tau))
    return np.stack(cols, axis=-1)


# -----------------------------------------------------------------------------
# Compounding conversions
# -----------------------------------------------------------------------------

def cont_to_periodic(rate_cc: np.ndarray | float, freq: int = 2) -> np.ndarray:
    """Continuously compounded rate (%) -> periodically compounded rate (%)."""
    r = np.asarray(rate_cc, dtype=float) / 100.0
    return 100.0 * freq * np.expm1(r / freq)


def periodic_to_cont(rate: np.ndarray | float, freq: int = 2) -> np.ndarray:
    """Periodically compounded rate (%) -> continuously compounded rate (%)."""
    r = np.asarray(rate, dtype=float) / 100.0
    return 100.0 * freq * np.log1p(r / freq)


# -----------------------------------------------------------------------------
# Parameters & curve object
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class NSSParams:
    """Parameter set of an NS (``beta3 = 0``, ``lambda2 = None``) or NSS curve."""

    beta0: float
    beta1: float
    beta2: float
    beta3: float = 0.0
    lambda1: float = 1.0
    lambda2: float | None = None

    @property
    def is_svensson(self) -> bool:
        return self.lambda2 is not None

    @property
    def betas(self) -> np.ndarray:
        if self.is_svensson:
            return np.array([self.beta0, self.beta1, self.beta2, self.beta3])
        return np.array([self.beta0, self.beta1, self.beta2])

    @property
    def lambdas(self) -> tuple[float, ...]:
        return (self.lambda1,) if self.lambda2 is None else (self.lambda1, self.lambda2)

    def to_vector(self) -> np.ndarray:
        """Optimiser vector: ``[betas..., lambdas...]``."""
        return np.concatenate([self.betas, self.lambdas])

    @classmethod
    def from_vector(cls, v: Iterable[float], svensson: bool = True) -> "NSSParams":
        v = [float(x) for x in v]
        if svensson:
            b0, b1, b2, b3, l1, l2 = v
            return cls(b0, b1, b2, b3, l1, l2)
        b0, b1, b2, l1 = v
        return cls(b0, b1, b2, 0.0, l1, None)

    def to_dict(self) -> dict[str, float]:
        """Flat dict using the historic CSV column names (NS gets NaN Beta3/Lambda2)."""
        return {
            "Beta0": self.beta0, "Beta1": self.beta1, "Beta2": self.beta2,
            "Beta3": self.beta3 if self.is_svensson else np.nan,
            "Lambda1": self.lambda1,
            "Lambda2": self.lambda2 if self.is_svensson else np.nan,
        }

    @classmethod
    def from_dict(cls, d) -> "NSSParams":
        l2 = d.get("Lambda2", np.nan)
        svensson = l2 is not None and np.isfinite(l2)
        return cls(float(d["Beta0"]), float(d["Beta1"]), float(d["Beta2"]),
                   float(d["Beta3"]) if svensson else 0.0,
                   float(d["Lambda1"]), float(l2) if svensson else None)


@dataclass(frozen=True)
class NSSCurve:
    """Evaluates zero, forward, discount and par curves for a parameter set."""

    params: NSSParams
    _coef: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "_coef", self.params.betas)

    # --- zero / forward / discount -----------------------------------------
    def zero(self, tau) -> np.ndarray:
        """Continuously compounded zero rate (%)."""
        p = self.params
        t = np.asarray(tau, dtype=float)
        z = loading_matrix(np.atleast_1d(t), p.lambda1, p.lambda2) @ self._coef
        return z if t.ndim else float(z[0])

    def forward(self, tau) -> np.ndarray:
        """Instantaneous forward rate (%) ``f(tau) = d(tau * y(tau)) / d tau``."""
        p = self.params
        t = np.asarray(tau, dtype=float)
        e1 = np.exp(-p.lambda1 * t)
        f = p.beta0 + p.beta1 * e1 + p.beta2 * p.lambda1 * t * e1
        if p.is_svensson:
            f = f + p.beta3 * p.lambda2 * t * np.exp(-p.lambda2 * t)
        return f

    def zero_jacobian(self, tau) -> np.ndarray:
        """``d zero(tau) / d theta`` with ``theta = [betas..., lambdas...]``; shape ``(n, p)``."""
        p = self.params
        t = np.atleast_1d(np.asarray(tau, dtype=float))
        L = loading_matrix(t, p.lambda1, p.lambda2)
        x1 = p.lambda1 * t
        cols = [L, (t * (p.beta1 * slope_loading_deriv(x1)
                         + p.beta2 * hump_loading_deriv(x1)))[:, None]]
        if p.is_svensson:
            cols.append((t * p.beta3 * hump_loading_deriv(p.lambda2 * t))[:, None])
        return np.hstack(cols)

    def discount(self, tau) -> np.ndarray:
        t = np.asarray(tau, dtype=float)
        return np.exp(-self.zero(t) * t / 100.0)

    @property
    def short_rate(self) -> float:
        return self.params.beta0 + self.params.beta1

    @property
    def long_rate(self) -> float:
        return self.params.beta0

    @property
    def hump_locations(self) -> tuple[float, ...]:
        """Maturities (years) at which each curvature loading peaks."""
        return tuple(HUMP_ARGMAX / lam for lam in self.params.lambdas)

    # --- market-convention yields ------------------------------------------
    def par_yield(self, maturity, freq: int = 2, bill_cutoff: float = 1.0) -> np.ndarray:
        """Bond-equivalent yield (%) consistent with Treasury CMT conventions.

        Maturities ``<= bill_cutoff`` are treated as zero-coupon bills
        (zero rate expressed with ``freq`` compounding); longer maturities
        are par coupon rates of a ``freq``-pay bullet bond, with the accrued
        interest of an initial stub period handled exactly.
        """
        mats = np.asarray(maturity, dtype=float)
        out = ParYieldPricer(np.atleast_1d(mats), freq, bill_cutoff)(self)
        return out if mats.ndim else float(out[0])

    def market_yield(self, maturity, target: str = "zero", freq: int = 2) -> np.ndarray:
        """Model-implied quote in market (bond-equivalent) convention.

        ``target='zero'`` re-expresses the zero rate with ``freq`` compounding
        (the classic approximation of treating CMT quotes as zeros);
        ``target='par'`` returns exact par yields.
        """
        if target == "par":
            return self.par_yield(maturity, freq)
        return cont_to_periodic(self.zero(maturity), freq)

    # --- bonds ---------------------------------------------------------------
    def bond_price(self, coupon: float, maturity: float, freq: int = 2,
                   zero_shift=None) -> float:
        """Dirty price per 100 face of a bullet bond discounted off this curve.

        ``zero_shift`` optionally maps maturities to an additive zero-rate
        shift (in %), used for key-rate bumps.
        """
        n = max(int(np.ceil(maturity * freq - 1e-9)), 1)
        times = maturity - np.arange(n)[::-1] / freq
        times = times[times > 1e-12]
        z = self.zero(times)
        if zero_shift is not None:
            z = z + zero_shift(times)
        dfs = np.exp(-z * times / 100.0)
        cfs = np.full(times.shape, coupon / freq)
        cfs[-1] += 100.0
        return float(np.dot(cfs, dfs))

    def key_rate_durations(self, coupon: float, maturity: float,
                           key_tenors: Sequence[float] = (0.25, 2, 5, 10, 20, 30),
                           bump_bp: float = 1.0, freq: int = 2) -> dict[float, float]:
        """Key-rate durations via triangular zero-rate bumps.

        Each key tenor gets a tent-shaped bump that is 1 at the tenor and
        falls linearly to 0 at its neighbours (flat beyond the end points),
        so the key-rate durations sum to the effective (parallel) duration.
        """
        keys = np.asarray(sorted(key_tenors), dtype=float)
        base = self.bond_price(coupon, maturity, freq)
        h = bump_bp / 100.0
        out = {}
        for i, k in enumerate(keys):
            basis = np.zeros_like(keys)
            basis[i] = 1.0

            def tent(t, basis=basis):
                return np.interp(t, keys, basis)

            up = self.bond_price(coupon, maturity, freq, lambda t: tent(t) * h)
            dn = self.bond_price(coupon, maturity, freq, lambda t: -tent(t) * h)
            out[float(k)] = (dn - up) / (2.0 * base * h / 100.0)
        return out


class ParYieldPricer:
    """Vectorised par-yield evaluation for a fixed set of maturities.

    The coupon schedule of every maturity is flattened into one array of
    cash-flow times so a curve's discount factors are evaluated with a
    single call; this matters inside the optimiser, which prices the same
    maturities thousands of times.
    """

    def __init__(self, maturities: Sequence[float], freq: int = 2,
                 bill_cutoff: float = 1.0):
        self.maturities = np.asarray(maturities, dtype=float)
        self.freq = freq
        self.is_bill = self.maturities <= bill_cutoff + 1e-12
        times, starts, first_frac = [], [], []
        for T in self.maturities[~self.is_bill]:
            n = int(np.ceil(T * freq - 1e-9))
            sched = T - np.arange(n)[::-1] / freq     # ascending coupon dates
            starts.append(sum(len(t) for t in times))
            times.append(sched)
            first_frac.append(sched[0] * freq)         # 1.0 when there is no stub
        self._times = np.concatenate(times) if times else np.empty(0)
        self._starts = np.asarray(starts, dtype=int)
        self._ends = np.asarray([s + len(t) for s, t in zip(starts, times)], dtype=int) - 1
        self._accrued = 1.0 - np.asarray(first_frac)

    def __call__(self, curve: "NSSCurve") -> np.ndarray:
        out = np.empty_like(self.maturities)
        if self.is_bill.any():
            out[self.is_bill] = cont_to_periodic(curve.zero(self.maturities[self.is_bill]),
                                                 self.freq)
        if len(self._starts):
            dfs = curve.discount(self._times)
            annuity = np.add.reduceat(dfs, self._starts) - self._accrued
            out[~self.is_bill] = 100.0 * self.freq * (1.0 - dfs[self._ends]) / annuity
        return out

    def jacobian(self, curve: "NSSCurve") -> np.ndarray:
        """``d par_yield / d theta``; shape ``(n_maturities, n_params)``."""
        n_par = len(curve.params.to_vector())
        jac = np.empty((len(self.maturities), n_par))
        if self.is_bill.any():
            tb = self.maturities[self.is_bill]
            dq_dz = np.exp(curve.zero(tb) / (100.0 * self.freq))[:, None]
            jac[self.is_bill] = dq_dz * curve.zero_jacobian(tb)
        if len(self._starts):
            t = self._times
            dfs = curve.discount(t)
            d_dfs = -(dfs * t / 100.0)[:, None] * curve.zero_jacobian(t)   # (m, p)
            annuity = np.add.reduceat(dfs, self._starts) - self._accrued
            d_annuity = np.add.reduceat(d_dfs, self._starts, axis=0)
            d_T, dd_T = dfs[self._ends], d_dfs[self._ends]
            jac[~self.is_bill] = 100.0 * self.freq * (
                -dd_T * annuity[:, None] - (1.0 - d_T)[:, None] * d_annuity) / annuity[:, None] ** 2
        return jac


def nss_zero(tau, b0, b1, b2, b3, l1, l2) -> np.ndarray:
    """Functional form kept for backwards compatibility with the original script."""
    return NSSCurve(NSSParams(b0, b1, b2, b3, l1, l2)).zero(np.asarray(tau, dtype=float))
