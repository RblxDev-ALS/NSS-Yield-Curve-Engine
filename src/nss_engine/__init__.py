"""NSS Yield Curve Engine.

Robust Nelson-Siegel-Svensson calibration, curve analytics, macro regime
detection and factor forecasting for the U.S. Treasury curve.

Quick start::

    >>> from nss_engine import NSSCurve, calibrate
    >>> import numpy as np
    >>> tau = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
    >>> true = NSSCurve(4.5, -1.5, -2.0, 1.0, 0.9, 0.15)
    >>> fit = calibrate(tau, true.par_yield(tau))  # quotes are par yields, like FRED CMT
    >>> round(fit.rmse_bp, 3) < 0.1
    True
"""

from .calibration import CalibrationConfig, FitResult, PanelFit, calibrate, calibrate_panel
from .models import DIEBOLD_LI_LAMBDA, NSSCurve, ns_loadings, nss_loadings
from .pipeline import PipelineConfig, PipelineResult, run_pipeline, write_outputs
from .statespace import DNSResult, fit_dns
from .validation import ReferenceComparison, compare_to_reference

__version__ = "2.0.0"

__all__ = [
    "DIEBOLD_LI_LAMBDA",
    "CalibrationConfig",
    "DNSResult",
    "FitResult",
    "NSSCurve",
    "PanelFit",
    "PipelineConfig",
    "PipelineResult",
    "ReferenceComparison",
    "__version__",
    "calibrate",
    "calibrate_panel",
    "compare_to_reference",
    "fit_dns",
    "ns_loadings",
    "nss_loadings",
    "run_pipeline",
    "write_outputs",
]
