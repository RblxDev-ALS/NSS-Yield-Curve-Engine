# Email to a professor

Pick someone who teaches fixed income, macro or econometrics, ideally one
whose class you took. Keep it short enough to read on a phone, ask for one
specific thing, and make it easy to say no. Rewrite it in your own words.

---

**Subject:** Feedback on a yield curve project (15 minutes?)

Dear Professor [name],

I took your [course] in [term]. Since then I have been working on an
open-source project that fits the U.S. Treasury yield curve every week since
1990 and estimates the term premium, breakeven inflation and recession odds
from it. It reruns on Federal Reserve data every weekday:
https://rblxdev-als.github.io/NSS-Yield-Curve-Engine/

The part I would most like an expert's view on is the term premium. The
standard ACM model is unstable when re-estimated in real time, so I anchored
its expected short rates to the Survey of Professional Forecasters, as Kim
and Orphanides do. That brings it within 28 basis points of the Fed Board's
Kim-Wright estimate, against 123 for the plain model. But when I scored both
against the bond returns that followed, the anchored premium did worse,
because the surveys kept forecasting rate rises that never came. I'd like to
know whether I am reading that tension correctly. The method and tests are on
the results page:
https://rblxdev-als.github.io/NSS-Yield-Curve-Engine/results.html

Would you have 15 minutes in office hours in the next few weeks to tell me
whether the approach is sound, or what you would check next? No worries at
all if not.

Thank you,
[Your name]
[Year, major]

---

Notes:

* Name the course and term so they can place you.
* If they ask how you built it, be straightforward about any AI tools you
  used and what you did yourself; professors care about that more than
  most readers.
* Bring the dashboard on a laptop to office hours, and a list of the three
  questions you most want answered.
