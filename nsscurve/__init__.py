"""nsscurve - Nelson-Siegel-Svensson yield curve engine & macro regime tracker."""

from .calibration import CalibrationConfig, CurveHistory, FitResult, NSSCalibrator
from .forecasting import DieboldLiForecaster
from .model import NSSCurve, NSSParams, cont_to_periodic, periodic_to_cont
from .pipeline import PipelineConfig, PipelineResult, export_results, run_pipeline

__version__ = "2.0.0"

__all__ = [
    "CalibrationConfig", "CurveHistory", "FitResult", "NSSCalibrator",
    "DieboldLiForecaster", "NSSCurve", "NSSParams", "cont_to_periodic", "periodic_to_cont",
    "PipelineConfig", "PipelineResult", "export_results", "run_pipeline", "__version__",
]
