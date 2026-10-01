# Show HN

**Title:** Show HN: The U.S. Treasury yield curve, refitted every week since 1990

**URL:** https://rblxdev-als.github.io/NSS-Yield-Curve-Engine/

**Text (first comment):**

This started as a sophomore-year script that fitted a Nelson–Siegel curve to
FRED data. It now refits the Treasury curve every weekday and splits the
10-year yield into expected Fed policy and a term premium, plus breakeven
inflation and recession odds.

Things I learned along the way:

* FRED's Treasury yields are par yields, not zero rates. Treating them as
  zero rates, as I did until version 2.0, put my curve 16.3 bp from the
  Fed's; fitting them correctly, 10.0 bp.
* The textbook term premium model is unstable if you re-estimate it each
  month. Anchoring it to the Philadelphia Fed's survey of forecasters fixed
  most of that.
* None of the three forecasting models I tried beat "yields don't change"
  one month ahead.
* An earlier version claimed one model had better-calibrated forecast
  intervals. A proper test said the opposite, so I withdrew it.

A GitHub Action reruns everything each weekday and republishes the site.
Code (MIT): https://github.com/RblxDev-ALS/NSS-Yield-Curve-Engine

---

*Notes: the HN guidelines ask for human-written comments; answer questions
yourself. Post on a weekday morning (U.S. time) and stay around for a few
hours.*
