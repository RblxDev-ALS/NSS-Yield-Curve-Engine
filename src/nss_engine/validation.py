"""Validation of fitted curves against an independent reference curve.

The Federal Reserve publishes its own daily Svensson curve (Gürkaynak, Sack &
Wright, 2007; :func:`nss_engine.data.load_gsw_parameters`). It is estimated
from different securities (off-the-run notes and bonds rather than on-the-run
CMT par yields) with a different loss (duration-weighted price errors), so
agreement with it is a genuine out-of-sample check on the zero curves this
engine produces. Differences have two sources worth keeping apart:

* **level offsets** that are stable over time - e.g. the on-the-run premium
  (on-the-run issues trade rich, so CMT-based curves sit a few bp lower) or
  a convexity bias in how quotes are interpreted;
* **tracking errors** - how well week-to-week *changes* agree, measured by the
  correlation of changes, which is insensitive to a constant offset.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .calibration import PanelFit
from .data import maturity_label
from .models import PARAM_NAMES, NSSCurve, nss_loadings

#: Maturities (years) compared by default: the range where the GSW curve is reliable.
REFERENCE_MATURITIES = (1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 25.0, 30.0)


@dataclass(frozen=True)
class ReferenceComparison:
    """Engine curve minus reference curve on common dates."""

    difference_bp: pd.DataFrame  #: dates × measures (zero rates by maturity, 5y5y forward)
    engine: pd.DataFrame  #: engine values (percent), same shape
    reference: pd.DataFrame  #: reference values (percent), same shape

    @property
    def n_dates(self) -> int:
        return len(self.difference_bp)

    def summary(self) -> pd.DataFrame:
        """Per measure: mean difference (bias), its std, RMSE, and correlation of changes."""
        d = self.difference_bp
        change_corr = {
            c: float(self.engine[c].diff().corr(self.reference[c].diff())) for c in d.columns
        }
        return pd.DataFrame(
            {
                "bias_bp": d.mean(),
                "std_bp": d.std(),
                "rmse_bp": np.sqrt((d**2).mean()),
                "change_corr": pd.Series(change_corr),
            }
        )

    def overall(self) -> dict[str, float]:
        """Headline numbers over the zero-rate maturities."""
        zero_cols = [c for c in self.difference_bp.columns if c.endswith("Y")]
        d = self.difference_bp[zero_cols]
        s = self.summary().loc[zero_cols]
        return {
            "n_dates": float(self.n_dates),
            "rmse_bp": float(np.sqrt(np.nanmean(d.to_numpy() ** 2))),
            "mean_abs_bias_bp": float(s["bias_bp"].abs().mean()),
            "demeaned_rmse_bp": float(np.sqrt(np.nanmean((d - d.mean()).to_numpy() ** 2))),
            "mean_change_corr": float(s["change_corr"].mean()),
        }


def _zero_matrix(params: pd.DataFrame, tau: np.ndarray) -> np.ndarray:
    return np.array(
        [
            nss_loadings(tau, r.lambda1, r.lambda2) @ np.array([r.beta0, r.beta1, r.beta2, r.beta3])
            for r in params.itertuples()
        ]
    )


def _values(params: pd.DataFrame, tau: np.ndarray) -> pd.DataFrame:
    zeros = _zero_matrix(params, tau)
    out = pd.DataFrame(zeros, index=params.index, columns=[maturity_label(t) for t in tau])
    out["5y5y fwd"] = [
        float(NSSCurve.from_mapping(r).forward_rate(5.0, 10.0)[0])
        for _, r in params[list(PARAM_NAMES)].iterrows()
    ]
    return out


def compare_to_reference(
    fit: PanelFit | pd.DataFrame,
    reference: pd.DataFrame,
    maturities: Sequence[float] = REFERENCE_MATURITIES,
) -> ReferenceComparison:
    """Compare engine zero curves with a reference parameter panel on common dates.

    ``fit`` is a :class:`PanelFit` or a parameter frame (columns ``beta0 …
    lambda2``); ``reference`` has the same columns (e.g. from
    :func:`~nss_engine.data.load_gsw_parameters`). Dates are matched exactly.
    """
    params = fit.params if isinstance(fit, PanelFit) else fit
    common = params.index.intersection(reference.index)
    if common.empty:
        raise ValueError("no dates in common between the fit and the reference curve")
    tau = np.asarray(maturities, dtype=float)
    eng = _values(params.loc[common, list(PARAM_NAMES)], tau)
    ref = _values(reference.loc[common, list(PARAM_NAMES)], tau)
    return ReferenceComparison((eng - ref) * 100.0, eng, ref)
