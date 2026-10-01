"""A synthetic Treasury market with *known* true parameters.

Used for three things:

1. **Testing** - calibration can be checked against ground truth, which is
   impossible with real data.
2. **Benchmarking** - comparing calibration algorithms on parameter recovery
   and stability, not just in-sample fit.
3. **Offline demos** - ``nss-engine run --source synthetic`` works without
   internet access.

The factors follow a mean-reverting VAR(1) (the dynamic Nelson-Siegel model of
Diebold & Li, 2006) with slowly drifting decay rates, and quotes are the model
curve plus i.i.d. measurement noise. A stylised recession indicator is attached:
a recession begins about a year after the 10y-3m spread has been inverted for a
full quarter - the empirical regularity that motivates the regime module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import TREASURY_SERIES
from .models import PARAM_NAMES, NSSCurve


@dataclass(frozen=True)
class SyntheticMarket:
    """Output of :func:`simulate_market`."""

    yields: pd.DataFrame  #: noisy quotes, columns = maturities (years)
    true_params: pd.DataFrame  #: true NSS parameters per date
    recession: pd.Series  #: monthly 0/1 recession indicator
    noise_bp: float

    def true_curve(self, i: int) -> NSSCurve:
        return NSSCurve.from_mapping(self.true_params.iloc[i])


# Long-run means, weekly AR(1) coefficients and shock sizes (percent) for the
# betas. Calibrated by eye to resemble 1990-2024 Treasury history.
_MU = np.array([5.5, -1.5, -1.0, 0.5])
_PHI = np.array([0.998, 0.995, 0.98, 0.98])
_SIGMA = np.array([0.07, 0.12, 0.30, 0.25])


def simulate_market(
    start: str = "1990-01-05",
    periods: int = 52 * 30,
    freq: str = "W-FRI",
    seed: int = 0,
    noise_bp: float = 3.0,
    maturities: np.ndarray | None = None,
    missing_short_end_until: str | None = "2001-07-31",
    quote: str = "par",
) -> SyntheticMarket:
    """Simulate a weekly Treasury panel from a dynamic NSS model.

    Parameters
    ----------
    periods, freq, start:
        Length and spacing of the panel.
    seed:
        RNG seed (results are fully reproducible).
    noise_bp:
        Standard deviation of i.i.d. quote noise, in basis points.
    maturities:
        Tenors in years (defaults to the 11 FRED CMT tenors).
    missing_short_end_until:
        Blank out the 1-month tenor before this date, mimicking FRED history
        (exercises the calibrator's missing-data handling). ``None`` disables.
    quote:
        ``"par"`` (default) quotes semi-annual par yields, like the FRED CMT
        series; ``"zero"`` quotes continuously compounded zero rates.
    """
    rng = np.random.default_rng(seed)
    mats = np.asarray(
        maturities if maturities is not None else list(TREASURY_SERIES.values()), dtype=float
    )
    index = pd.date_range(start=start, periods=periods, freq=freq)

    betas = np.empty((periods, 4))
    betas[0] = _MU + rng.normal(0, 1, 4) * _SIGMA / np.sqrt(1 - _PHI**2) * 0.5
    for t in range(1, periods):
        betas[t] = _MU + _PHI * (betas[t - 1] - _MU) + _SIGMA * rng.normal(size=4)
        # Soft zero lower bound: the short rate β0 + β1 cannot go below 5 bp.
        betas[t, 1] = max(betas[t, 1], 0.05 - betas[t, 0])

    # Decay rates drift slowly in logs around typical values, inside the
    # calibrator's default bounds and ratio constraint.
    log_l1 = np.log(0.8) + _ar1_path(rng, periods, phi=0.995, sigma=0.02)
    log_l2 = np.log(0.18) + _ar1_path(rng, periods, phi=0.995, sigma=0.02)
    lambdas = np.exp(np.column_stack([log_l1, log_l2]))

    params = np.column_stack([betas, lambdas])
    true_params = pd.DataFrame(params, index=index, columns=list(PARAM_NAMES))

    clean = np.array(
        [
            NSSCurve.from_array(p).par_yield(mats)
            if quote == "par"
            else NSSCurve.from_array(p).zero(mats)
            for p in params
        ]
    )
    noisy = clean + rng.normal(0.0, noise_bp / 100.0, clean.shape)
    yields = pd.DataFrame(noisy, index=index, columns=mats)
    yields.index.name = "date"
    if missing_short_end_until is not None and np.isclose(mats, 1 / 12).any():
        col = mats[np.isclose(mats, 1 / 12)][0]
        yields.loc[yields.index < pd.Timestamp(missing_short_end_until), col] = np.nan

    spread = pd.Series(
        clean[:, np.argmin(np.abs(mats - 10))] - clean[:, np.argmin(np.abs(mats - 0.25))],
        index=index,
    )
    recession = _stylised_recessions(spread)
    return SyntheticMarket(
        yields=yields, true_params=true_params, recession=recession, noise_bp=noise_bp
    )


@dataclass(frozen=True)
class TipsMarket:
    """Output of :func:`simulate_tips_market`."""

    yields: pd.DataFrame  #: noisy real (TIPS) par yields, columns = maturities (years)
    true_real_params: pd.DataFrame  #: true real NSS curve per date
    true_breakeven_params: pd.DataFrame  #: true zero-coupon breakeven curve (NSS form)

    def true_breakevens(self) -> pd.DataFrame:
        """True zero-coupon breakevens at 5 and 10 years and the 5y5y forward (percent)."""
        rows = [
            NSSCurve.from_array(p).zero([5.0, 10.0]) for p in self.true_breakeven_params.to_numpy()
        ]
        out = pd.DataFrame(
            rows, index=self.true_breakeven_params.index, columns=["be_5y", "be_10y"]
        )
        out["be_5y5y"] = (10.0 * out["be_10y"] - 5.0 * out["be_5y"]) / 5.0
        return out


def simulate_tips_market(
    nominal: SyntheticMarket | pd.DataFrame,
    seed: int = 0,
    noise_bp: float = 3.0,
    start: str = "2003-01-01",
    maturities: tuple[float, ...] = (5.0, 7.0, 10.0, 20.0, 30.0),
    long_end_from: str | None = "2010-02-28",
) -> TipsMarket:
    """TIPS real par yields consistent with a nominal :func:`simulate_market`.

    ``nominal`` is the nominal market or its ``true_params``.

    The zero-coupon breakeven curve is an NSS curve with the nominal curve's
    decay rates and its own betas: a level near 2.4% and a slope and
    curvature that swing enough to reproduce episodes like the 2008 collapse
    of short breakevens. The true real curve is the nominal curve minus it,
    so it is an NSS curve that a four-parameter fit to 4-5 quotes cannot
    match exactly (as with real TIPS). Quotes start at ``start``; the 30-year
    only from ``long_end_from``, as on FRED. The quote noise (3 bp by default)
    is larger than for nominal yields because TIPS trade less.
    """
    rng = np.random.default_rng(seed + 10_000)
    params = nominal.true_params if isinstance(nominal, SyntheticMarket) else nominal
    true = params.loc[params.index >= pd.Timestamp(start)]
    n = len(true)
    if n == 0:
        raise ValueError(f"the nominal market has no dates from {start}")
    mu = np.array([2.4, -0.2, 0.0])
    phi = np.array([0.995, 0.98, 0.98])
    sigma = np.array([0.04, 0.12, 0.10])
    b = np.empty((n, 3))
    b[0] = mu
    for t in range(1, n):
        b[t] = mu + phi * (b[t - 1] - mu) + sigma * rng.normal(size=3)
    be = true.copy()
    be[["beta0", "beta1", "beta2"]] = b
    be["beta3"] = 0.0
    real = true.copy()
    real[["beta0", "beta1", "beta2", "beta3"]] = (
        true[["beta0", "beta1", "beta2", "beta3"]].to_numpy()
        - be[["beta0", "beta1", "beta2", "beta3"]].to_numpy()
    )
    mats = np.asarray(maturities, dtype=float)
    clean = np.array([NSSCurve.from_array(p).par_yield(mats) for p in real.to_numpy()])
    noisy = clean + rng.normal(0.0, noise_bp / 100.0, clean.shape)
    yields = pd.DataFrame(noisy, index=true.index, columns=mats)
    yields.index.name = "date"
    if long_end_from is not None and 30.0 in mats:
        yields.loc[yields.index < pd.Timestamp(long_end_from), 30.0] = np.nan
    return TipsMarket(yields=yields, true_real_params=real, true_breakeven_params=be)


def _ar1_path(rng: np.random.Generator, n: int, phi: float, sigma: float) -> np.ndarray:
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + sigma * rng.normal()
    return x


def _stylised_recessions(
    spread: pd.Series, min_inversion_months: int = 3, lag_months: int = 12, length_months: int = 9
) -> pd.Series:
    """Monthly recession flags: a recession starts ``lag_months`` after the spread
    has been negative for ``min_inversion_months`` consecutive months."""
    monthly = spread.resample("ME").mean()
    inverted = (monthly < 0).astype(int)
    run = inverted.groupby((inverted != inverted.shift()).cumsum()).cumsum() * inverted
    rec = pd.Series(0, index=monthly.index, name="recession")
    triggers = monthly.index[run.to_numpy() == min_inversion_months]
    for trig in triggers:
        start_pos = monthly.index.get_loc(trig) - (min_inversion_months - 1) + lag_months
        rec.iloc[start_pos : start_pos + length_months] = 1
    return rec


# =============================================================================
# An arbitrage-free affine market with a known term premium
# =============================================================================


@dataclass(frozen=True)
class AffineMarket:
    """Output of :func:`simulate_affine_market`."""

    yields: pd.DataFrame  #: zero yields (percent), columns 1/12 … max_months/12 years
    term_premium: pd.DataFrame  #: true term premium (percent), same shape
    factors: pd.DataFrame  #: true state variables
    mu: np.ndarray | None = None  #: real-world VAR intercept (percent)
    phi: np.ndarray | None = None  #: real-world VAR slope
    bill_loadings: tuple[float, np.ndarray] | None = None  #: 3-month yield = c0 + c1'X

    def expected_bill_rate(self, start: int, end: int) -> pd.Series:
        """True expectation, at each date, of the average 3-month yield ``start…end`` months ahead."""
        if self.mu is None or self.phi is None or self.bill_loadings is None:
            raise ValueError("market has no stored dynamics")
        c0, c1 = self.bill_loadings
        k = self.phi.shape[0]
        xbar = np.linalg.solve(np.eye(k) - self.phi, self.mu)
        powers = [np.linalg.matrix_power(self.phi, m) for m in range(start, end + 1)]
        W = np.mean(powers, axis=0)
        D = self.factors.to_numpy() - xbar
        return pd.Series(c0 + c1 @ xbar + D @ W.T @ c1, index=self.factors.index)

    def surveys(
        self,
        windows: dict[str, tuple[int, int]] | None = None,
        every: int = 3,
        noise_pp: float = 0.1,
        bias_pp: float = 0.0,
        seed: int = 0,
    ) -> pd.DataFrame:
        """Simulated survey forecasts of the 3-month yield in the SPF format.

        Every ``every`` months, each window gets the true expectation plus
        ``bias_pp`` and N(0, ``noise_pp``²) noise; the ten-year window
        (``"BILL10"``) is surveyed once a year, as in the SPF.
        """
        windows = windows or {"Q1": (2, 4), "Q2": (5, 7), "Q4": (11, 13), "BILL10": (1, 120)}
        rng = np.random.default_rng(seed)
        rows = []
        for name, (start, end) in windows.items():
            step = 12 if name == "BILL10" else every
            exp = self.expected_bill_rate(start, end).iloc[::step]
            vals = exp.to_numpy() + bias_pp + rng.normal(0.0, noise_pp, exp.size)
            rows.append(
                pd.DataFrame(
                    {"date": exp.index, "series": name, "start": start, "end": end, "value": vals}
                )
            )
        return pd.concat(rows, ignore_index=True).sort_values(["date", "start"], ignore_index=True)


def simulate_affine_market(
    periods: int = 600,
    seed: int = 0,
    noise_bp: float = 0.0,
    max_months: int = 120,
    start: str = "1970-01-31",
    level_persistence: float = 0.97,
) -> AffineMarket:
    """Simulate a monthly three-factor Gaussian affine term structure model.

    The factors are a level, a slope and a curvature (in percent); the short
    rate is level + slope, and curvature feeds into the slope, so all three
    move yields. Prices of risk ``λ_t = λ0 + λ1 X_t`` are set so that the
    10-year term premium averages about 0.9 points and moves with the slope, as
    in U.S. data. Samples too long for pandas timestamps (past the year 2262)
    get a plain integer index. Because the true risk-neutral yields are known, the term
    premium estimated by :func:`~nss_engine.termpremium.fit_acm` can be
    checked against the truth. ``level_persistence`` is the monthly
    autocorrelation of the level under the real-world measure (U.S. rates are
    closer to a unit root than the default 0.97, which is where small-sample
    bias matters most). The risk-neutral dynamics, and so the cross-section of
    yields, do not depend on it: the difference goes into the price of level
    risk.
    """
    from .termpremium import affine_loadings

    rng = np.random.default_rng(seed)
    phi = np.array([[level_persistence, 0.0, 0.0], [0.0, 0.93, 0.06], [0.0, 0.0, 0.85]])
    mean = np.array([5.0, -1.5, 0.0])
    mu = (np.eye(3) - phi) @ mean
    chol = np.diag([0.25, 0.30, 0.40])
    delta1 = np.array([1.0, 1.0, 0.0]) / 1200.0
    lambda0 = np.array([-0.13, 0.0, 0.0])
    lambda1 = np.zeros((3, 3))
    lambda1[0, 0] = level_persistence - 0.97  # risk-neutral level persistence stays 0.97
    lambda1[0, 1] = -0.06
    X = np.empty((periods, 3))
    X[0] = mean
    for t in range(1, periods):
        X[t] = mu + phi @ X[t - 1] + chol @ rng.standard_normal(3)
    sigma = chol @ chol.T
    A, B = affine_loadings(max_months, mu - lambda0, phi - lambda1, sigma, 0.0, 0.0, delta1)
    A_rn, B_rn = affine_loadings(max_months, mu, phi, sigma, 0.0, 0.0, delta1)
    n = np.arange(1, max_months + 1)
    fitted = -(A + X @ B.T) / n * 1200.0
    risk_neutral = -(A_rn + X @ B_rn.T) / n * 1200.0
    try:
        index: pd.Index = pd.date_range(start=start, periods=periods, freq="ME", name="date")
    except (pd.errors.OutOfBoundsDatetime, OverflowError):
        index = pd.RangeIndex(periods, name="month")  # samples longer than pandas' dates
    cols = n / 12.0
    noisy = fitted + rng.normal(0.0, noise_bp / 100.0, fitted.shape)
    return AffineMarket(
        yields=pd.DataFrame(noisy, index=index, columns=cols),
        term_premium=pd.DataFrame(fitted - risk_neutral, index=index, columns=cols),
        factors=pd.DataFrame(X, index=index, columns=["level", "slope", "curvature"]),
        mu=mu,
        phi=phi,
        bill_loadings=(float(-A[2] / 3 * 1200.0), -B[2] / 3 * 1200.0),
    )
