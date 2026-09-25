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
