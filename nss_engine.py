"""Entry point for the NSS Yield Curve Engine.

    python nss_engine.py                 # live FRED data, weekly, 5 years
    python nss_engine.py --offline       # synthetic data, no network needed
    python nss_engine.py --help          # all options

The implementation lives in the ``nsscurve`` package.
"""

import sys

from nsscurve.cli import main

if __name__ == "__main__":
    sys.exit(main())
