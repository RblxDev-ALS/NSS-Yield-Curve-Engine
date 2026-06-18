# Institutional NSS Yield Curve Engine & Macro Regime Tracker



## Overview
A quantitative finance engine built in Python to parameterize the U.S. Treasury Term Structure. 

Most open-source yield curve models rely on static optimization algorithms that fail or produce violent parameter swings during volatile macro regimes. This pipeline solves that instability by expanding the standard Nelson-Siegel model to the **Nelson-Siegel-Svensson (NSS) 6-factor model**, utilizing sequential warm starting and L2 Ridge Regularization to stabilize the latent curvature factors.

The result is a production-ready tool that tracks macroeconomic cycles and generates actionable relative value (RV) trading signals.

## Core Features
* **Automated Data Pipeline:** Fetches and cleans weekly constant maturity Treasury rates directly from the St. Louis FED (FRED) API.
* **Algorithmic Stability:** Implements the L-BFGS-B non-linear optimization solver with strict economic bounding, warm-started initial guesses, and L2 regularization to prevent extreme, unrealistic curve humps.
* **Macro Regime Detection:** Mathematically isolates the implied curve slope ($-\beta_1$) and automatically classifies the market state. It triggers a "Recession Warning" when the curve deeply inverts, mirroring the physical 10Y-2Y benchmark spread.
* **Interactive 2x2 Dashboard:** Built with Plotly to provide a multi-dimensional view of the term structure, including a live 2D curve comparison, macro factor time-series tracking with regime shading, and a 3D volatility surface.
* **Downstream Data Export:** Automatically writes cleaned latent macro factors (Beta0-Beta3, Lambda1-Lambda2) to a CSV for immediate use in machine learning backtests and predictive modeling.

## Why This Matters (The Math)
In the NSS framework, the long-term rate is represented by $\beta_0$, and the short-term rate is $\beta_0 + \beta_1$. Therefore, the actual slope of the yield curve (Long minus Short) is mathematically $-\beta_1$. 

This engine corrects standard programmatic assumptions by accurately tracking $-\beta_1$ against the physical 10Y-2Y spread, proving that the mathematical optimization is perfectly translating the term structure into a reliable, single-factor macro signal.

## How to Run the Engine

1. Clone the repository:
pip install -r requirements.txt
