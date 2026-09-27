# Show HN

**Title:** Show HN: What the U.S. yield curve says, measured weekly since 1990 and tested out of sample

**URL:** https://rblxdev-als.github.io/NSS-Yield-Curve-Engine/

**Text (first comment):**

This started as a sophomore-year script that fitted a Nelson–Siegel curve to
FRED data. It is now an open-source Python engine that fits the Treasury curve
every week since 1990 and splits it into things people argue about: expected
Fed policy vs the term premium, breakeven inflation, recession odds.

The part I think is interesting is the testing. Every estimate is checked
twice: against a simulated market where the truth is known, and against
independent published estimates (the Fed's own curves, the Fed Board's
Kim–Wright term premium, FRED's breakevens). Everything that can be done "in
real time" is re-estimated each month using only data published by then.

Some results, including the ones that went against me:

* FRED's Treasury yields are par yields, not zero rates. Treating them as
  zero rates, as I did until version 2.0, puts the curve 16.3 bp from the Fed's;
  fitting them as par yields, 10.0 bp.
* The textbook term premium model (ACM) is unstable in real time. Anchoring
  it to the Philadelphia Fed's survey of forecasters cuts its error against a
  known truth from 82 to 13 bp.
* No model beats "yields don't change" at forecasting. I tested three.
* A claim I made in an earlier version (better-calibrated forecast intervals)
  was reversed by a proper test.

A GitHub Action reruns everything each weekday and republishes the site.
Code: https://github.com/RblxDev-ALS/NSS-Yield-Curve-Engine (MIT).
