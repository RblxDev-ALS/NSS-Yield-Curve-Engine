"""Curve analytics built on calibrated NSS curves.

* **Factor validation** - principal components of yield changes and their
  relationship to the NSS level/slope/curvature factors.
* **Bond risk** - pricing off the zero curve, DV01, duration, convexity,
  key-rate durations and *factor durations* (sensitivity to each β).
* **Carry & roll-down** - expected return of a bond if the curve does not move.
* **Relative value** - rich/cheap signals from fit residuals, with z-scores and
  mean-reversion half-lives.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from .data import maturity_label
from .models import FloatArray, NSSCurve, coupon_schedule, nss_loadings

# =============================================================================
# Factor validation: PCA and empirical proxies
# =============================================================================


@dataclass(frozen=True)
class PCAResult:
    """Principal components of a yield panel."""

    explained_variance_ratio: pd.Series  #: share of variance per component
    loadings: pd.DataFrame  #: maturities × components (unit-norm eigenvectors)
    scores: pd.DataFrame  #: dates × components

    @property
    def cumulative(self) -> pd.Series:
        return self.explained_variance_ratio.cumsum()


def pca(yields: pd.DataFrame, n_components: int = 3, changes: bool = True) -> PCAResult:
    """PCA of yield levels or (by default) changes, using complete-case rows.

    Litterman & Scheinkman (1991) found that three components - level, slope,
    curvature - explain almost all Treasury return variance; NSS hard-codes
    exactly that structure, so this is the natural empirical check.
    """
    data = yields.dropna()
    if changes:
        data = data.diff().dropna()
    if len(data) < n_components + 1:
        raise ValueError("not enough complete observations for PCA")
    centred = data - data.mean()
    cov = np.cov(centred.to_numpy(), rowvar=False)
    eigval, eigvec = np.linalg.eigh(cov)
    order = np.argsort(eigval)[::-1]
    eigval, eigvec = eigval[order], eigvec[:, order]
    # Sign convention: level loads positively on average, slope rises with
    # maturity, curvature is positive in the belly.
    names = [f"PC{i + 1}" for i in range(n_components)]
    vecs = eigvec[:, :n_components].copy()
    mats = np.asarray(data.columns, dtype=float)
    if n_components >= 1 and vecs[:, 0].sum() < 0:
        vecs[:, 0] *= -1
    if n_components >= 2 and vecs[np.argmax(mats), 1] < vecs[np.argmin(mats), 1]:
        vecs[:, 1] *= -1
    if n_components >= 3:
        mid = np.argmin(np.abs(mats - np.median(mats)))
        if vecs[mid, 2] < 0:
            vecs[:, 2] *= -1
    ratio = pd.Series(eigval[:n_components] / eigval.sum(), index=names, name="explained_variance")
    loadings = pd.DataFrame(vecs, index=data.columns, columns=names)
    scores = pd.DataFrame(centred.to_numpy() @ vecs, index=data.index, columns=names)
    return PCAResult(ratio, loadings, scores)


def factor_proxies(yields: pd.DataFrame) -> pd.DataFrame:
    """Model-free level, slope and curvature proxies (Diebold & Li, 2006).

    * level     = (3M + 2Y + 10Y) / 3
    * slope     = 10Y − 3M
    * curvature = 2·2Y − 3M − 10Y
    """
    y3m, y2, y10 = (_column(yields, t) for t in (0.25, 2.0, 10.0))
    return pd.DataFrame(
        {
            "level": (y3m + y2 + y10) / 3.0,
            "slope": y10 - y3m,
            "curvature": 2.0 * y2 - y3m - y10,
        }
    )


def factor_proxy_correlations(params: pd.DataFrame, yields: pd.DataFrame) -> pd.Series:
    """Correlation of NSS factors with their empirical proxies.

    ``β0 ↔ level``, ``−β1 ↔ slope`` and ``β2 ↔ curvature``. Values near 1 confirm
    the latent factors carry their textbook economic meaning.
    """
    prox = factor_proxies(yields).reindex(params.index)
    pairs = {
        "beta0 ~ level": (params["beta0"], prox["level"]),
        "-beta1 ~ slope": (-params["beta1"], prox["slope"]),
        "beta2 ~ curvature": (params["beta2"], prox["curvature"]),
    }
    return pd.Series({k: a.corr(b) for k, (a, b) in pairs.items()}, name="correlation")


def _column(df: pd.DataFrame, tau: float) -> pd.Series:
    cols = np.asarray(df.columns, dtype=float)
    idx = int(np.argmin(np.abs(cols - tau)))
    if abs(cols[idx] - tau) > 1e-6:
        raise KeyError(f"maturity {tau} not in panel")
    return df.iloc[:, idx]


# =============================================================================
# Bond pricing and risk
# =============================================================================


@dataclass(frozen=True)
class Bond:
    """A fixed-coupon bullet bond. :func:`price` returns the full (dirty) price;
    for maturities that are not a whole number of coupon periods the first
    coupon is a full coupon paid within one period (see :func:`coupon_schedule`).

    ``coupon`` is the annual coupon rate in percent; ``coupon = 0`` gives a
    zero-coupon bond.
    """

    maturity: float
    coupon: float
    freq: int = 2
    face: float = 100.0

    def cashflows(self) -> tuple[FloatArray, FloatArray]:
        """``(times, amounts)`` of all remaining cash flows."""
        if self.coupon == 0:
            return np.array([self.maturity]), np.array([self.face])
        times, _ = coupon_schedule(self.maturity, self.freq)
        n = times.size
        amounts = np.full(n, self.face * self.coupon / 100.0 / self.freq)
        amounts[-1] += self.face
        return times, amounts

    @classmethod
    def par(cls, curve: NSSCurve, maturity: float, freq: int = 2) -> Bond:
        """The bond whose coupon makes it price at par on ``curve``."""
        return cls(maturity, float(curve.par_yield(maturity, freq)[0]), freq)


ZeroShift = Callable[[FloatArray], FloatArray]


def price(curve: NSSCurve, bond: Bond, zero_shift: ZeroShift | None = None) -> float:
    """Price off the NSS zero curve, optionally with a shift (percent) applied to zero rates."""
    times, amounts = bond.cashflows()
    z = curve.zero(times)
    if zero_shift is not None:
        z = z + zero_shift(times)
    return float(np.sum(amounts * np.exp(-z * times / 100.0)))


@dataclass(frozen=True)
class RiskReport:
    price: float
    dv01: float  #: price change for a 1 bp parallel fall in zero rates
    duration: float  #: effective (≈ Macaulay under continuous compounding) duration, years
    convexity: float
    key_rate_durations: pd.Series
    factor_durations: pd.Series


def risk_report(
    curve: NSSCurve,
    bond: Bond,
    key_rates: Sequence[float] = (0.25, 2.0, 5.0, 10.0, 30.0),
    bump_bp: float = 1.0,
) -> RiskReport:
    """Full risk decomposition of ``bond`` on ``curve``.

    * DV01, duration and convexity from symmetric parallel bumps of the zero curve.
    * Key-rate durations (Ho, 1992) from triangular bumps centred on
      ``key_rates``; they sum to the effective duration.
    * Factor durations (Willner, 1996) ``-(1/P) ∂P/∂β_k`` - the bond's exposure to
      a 1-percentage-point move in each NSS factor. These are what a desk would
      use to hedge level/slope/curvature risk.
    """
    h = bump_bp / 100.0
    p0 = price(curve, bond)
    up = price(curve, bond, lambda t: np.full_like(t, h))
    dn = price(curve, bond, lambda t: np.full_like(t, -h))
    duration = (dn - up) / (2 * h * p0) * 100.0
    convexity = (up + dn - 2 * p0) / (p0 * h**2) * 1e4
    dv01 = (dn - up) / 2.0 / bump_bp

    krd = {}
    kr = np.asarray(sorted(key_rates), dtype=float)
    for i, k in enumerate(kr):
        left = kr[i - 1] if i > 0 else None
        right = kr[i + 1] if i < kr.size - 1 else None
        shape = _triangle(k, left, right)
        up_k = price(curve, bond, _scaled(shape, h))
        dn_k = price(curve, bond, _scaled(shape, -h))
        krd[maturity_label(k)] = (dn_k - up_k) / (2 * h * p0) * 100.0

    times, amounts = bond.cashflows()
    disc = np.exp(-curve.zero(times) * times / 100.0)
    load = nss_loadings(times, curve.lambda1, curve.lambda2)
    # ∂P/∂β_k = Σ CF_i · D_i · (−t_i/100) · L_k(t_i); duration per 1 percentage point
    dP = -(amounts * disc * times / 100.0) @ load
    fdur = pd.Series(
        -dP / p0 * 100.0, index=["level (β0)", "slope (β1)", "curvature (β2)", "curvature2 (β3)"]
    )
    return RiskReport(
        p0, dv01, duration, convexity, pd.Series(krd, name="krd"), fdur.rename("factor_duration")
    )


def _scaled(shape: ZeroShift, size: float) -> ZeroShift:
    def shift(t: FloatArray) -> FloatArray:
        return size * shape(t)

    return shift


def _triangle(center: float, left: float | None, right: float | None) -> ZeroShift:
    def shape(t: FloatArray) -> FloatArray:
        out = np.zeros_like(t)
        if left is None:
            out[t <= center] = 1.0
        else:
            m = (t > left) & (t <= center)
            out[m] = (t[m] - left) / (center - left)
        if right is None:
            out[t > center] = 1.0
        else:
            m = (t > center) & (t < right)
            out[m] = (right - t[m]) / (right - center)
        return out

    return shape


# =============================================================================
# Carry and roll-down
# =============================================================================


def carry_rolldown(curve: NSSCurve, maturities: ArrayLike, horizon: float = 0.25) -> pd.DataFrame:
    """Expected return of zero-coupon bonds over ``horizon`` years if the curve is unchanged.

    For a zero with maturity ``T`` the log return is exactly::

        z(T)·T − z(T−h)·(T−h) = z(T)·h  +  [z(T) − z(T−h)]·(T−h)
                                ───────     ─────────────────────
                                 carry            roll-down

    ``excess`` subtracts the return from holding cash (the ``h``-year zero), i.e.
    the return of a financed position. All values are in basis points over the
    horizon (not annualised).
    """
    T = np.atleast_1d(np.asarray(maturities, dtype=float))
    if np.any(horizon >= T):
        raise ValueError("every maturity must exceed the horizon")
    zT, zTh = curve.zero(T), curve.zero(T - horizon)
    carry = zT * horizon
    roll = (zT - zTh) * (T - horizon)
    cash = curve.zero(horizon)[0] * horizon
    return pd.DataFrame(
        {
            "yield_pct": zT,
            "carry_bp": carry * 100.0,
            "rolldown_bp": roll * 100.0,
            "total_bp": (carry + roll) * 100.0,
            "excess_bp": (carry + roll - cash) * 100.0,
        },
        index=pd.Index([maturity_label(t) for t in T], name="tenor"),
    )


# =============================================================================
# Relative value from residuals
# =============================================================================


def residual_zscores(
    residuals_bp: pd.DataFrame, window: int = 52, min_periods: int = 26
) -> pd.DataFrame:
    """Rolling z-score of each tenor's residual (uses only past data - no look-ahead)."""
    mean = residuals_bp.rolling(window, min_periods=min_periods).mean().shift(1)
    std = residuals_bp.rolling(window, min_periods=min_periods).std().shift(1)
    return (residuals_bp - mean) / std


def residual_half_life(residuals_bp: pd.DataFrame) -> pd.Series:
    """Mean-reversion half-life (in periods) of each tenor's residual, from an AR(1) fit.

    A short half-life means mispricings relative to the fitted curve correct
    quickly - the precondition for any curve relative-value strategy.
    """
    out = {}
    for col in residuals_bp.columns:
        s = residuals_bp[col].dropna()
        if len(s) < 10:
            out[maturity_label(float(col))] = np.nan
            continue
        x, y = s.to_numpy()[:-1], s.to_numpy()[1:]
        x_c = x - x.mean()
        phi = float(np.dot(x_c, y - y.mean()) / np.dot(x_c, x_c))
        if phi >= 1:
            half_life = np.inf  # no mean reversion
        elif phi <= 0:
            half_life = 0.0  # reverts (or overshoots) within one period
        else:
            half_life = np.log(0.5) / np.log(phi)
        out[maturity_label(float(col))] = half_life
    return pd.Series(out, name="half_life_periods")


def rich_cheap(
    residuals_bp: pd.DataFrame, window: int = 52, threshold: float = 2.0
) -> pd.DataFrame:
    """Latest rich/cheap table.

    A positive residual means the observed yield is *above* the fitted curve, so
    the bond's price is low relative to its neighbours: **cheap** (a long
    candidate). Negative residuals are **rich**.
    """
    z = residual_zscores(residuals_bp, window)
    last_res = residuals_bp.iloc[-1]
    last_z = z.iloc[-1]
    signal = np.where(last_z > threshold, "CHEAP", np.where(last_z < -threshold, "RICH", "fair"))
    return pd.DataFrame(
        {"residual_bp": last_res.to_numpy(), "zscore": last_z.to_numpy(), "signal": signal},
        index=pd.Index([maturity_label(float(c)) for c in residuals_bp.columns], name="tenor"),
    )
