"""Tenor label helpers (``'3M'`` <-> 0.25 years) and the FRED CMT universe."""

from __future__ import annotations

import re
from typing import Iterable

import numpy as np

# FRED constant-maturity Treasury series (percent, semi-annual bond-equivalent).
FRED_CMT_SERIES: dict[str, str] = {
    "1M": "DGS1MO", "3M": "DGS3MO", "6M": "DGS6MO",
    "1Y": "DGS1", "2Y": "DGS2", "3Y": "DGS3", "5Y": "DGS5",
    "7Y": "DGS7", "10Y": "DGS10", "20Y": "DGS20", "30Y": "DGS30",
}
SERIES_TO_TENOR = {v: k for k, v in FRED_CMT_SERIES.items()}

_TENOR_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([DWMY])\s*$", re.IGNORECASE)
_UNIT_YEARS = {"D": 1 / 365.0, "W": 7 / 365.0, "M": 1 / 12.0, "Y": 1.0}


def parse_tenor(label: str | float) -> float:
    """``'3M'`` -> 0.25, ``'10Y'`` -> 10.0, FRED ids (``'DGS3MO'``) and numbers pass through."""
    if isinstance(label, (int, float, np.floating, np.integer)):
        return float(label)
    label = SERIES_TO_TENOR.get(str(label).upper(), str(label))
    m = _TENOR_RE.match(label)
    if not m:
        raise ValueError(f"Unrecognised tenor label: {label!r}")
    return float(m.group(1)) * _UNIT_YEARS[m.group(2).upper()]


def format_tenor(years: float) -> str:
    """0.25 -> ``'3M'``, 10 -> ``'10Y'``."""
    months = years * 12.0
    if years < 1.0 - 1e-9 and abs(months - round(months)) < 1e-6:
        return f"{int(round(months))}M"
    if abs(years - round(years)) < 1e-9:
        return f"{int(round(years))}Y"
    return f"{years:g}Y"


def maturities_of(labels: Iterable) -> np.ndarray:
    return np.array([parse_tenor(c) for c in labels], dtype=float)
