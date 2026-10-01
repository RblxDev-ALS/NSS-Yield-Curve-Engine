# Launch kit (drafts, not published)

| file | where | angle |
|---|---|---|
| [reddit-quant.md](reddit-quant.md) | r/quant | methods; asks for critique |
| [reddit-econometrics.md](reddit-econometrics.md) | r/econometrics | how much of the 10-year yield is term premium |
| [reddit-python.md](reddit-python.md) | r/Python (Showcase flair) | engineering and testing |
| [reddit-algotrading.md](reddit-algotrading.md) | r/algotrading | the random walk still wins |
| [show-hn.md](show-hn.md) | Hacker News | short, links the website |
| [professor-email.md](professor-email.md) | email to a professor | asks for 15 minutes of feedback |
| [friends.md](friends.md) | group chat, Discord, LinkedIn | two or three lines |
| [faq.md](faq.md) | any thread | likely questions, with the numbers to answer them |

## Before posting

1. Merge to `main`, wait for the live workflow to deploy the site, and check
   that the README badges, charts and the site's link preview work (paste
   the site URL into a Discord or Slack message to see the preview card).
2. Release 2.5.0 to PyPI so `pip install nss-engine` works: once the PyPI
   publisher and the `pypi` environment are set up (docs/releasing.md), it is
   *Actions → Release → Run workflow* on `main` with *publish* ticked.
3. Check before posting: run `python scripts/preflight.py` (`--offline` skips
   the network checks). It checks PyPI, the site and its link preview, data
   freshness, the README links, the author name and the GitHub About box, and
   prints a fix hint for every WARN or FAIL. Fix every FAIL.
4. Re-read each draft against the latest live run and update any number
   that moved. Every number must be in the README, docs/results.md or the
   run's job summary.

## Write them in your own words

These drafts are a starting point. They were written with AI help, and
people on Reddit and Hacker News notice the style quickly: bold lead-ins on
every paragraph, lists of exactly three, "it's not X, it's Y", and a tidy
paragraph for every point. Hacker News's guidelines also ask people not to
post AI-generated or AI-edited comments. So:

* Rewrite each post the way you would explain the project to a friend in
  your year. Keep it short; the README and results page hold the details.
* Only include claims you can defend in the comments without looking them
  up. Cut the rest; nobody will miss them.
* Reply to comments yourself, in your own words. The FAQ is there to give
  you the numbers, not the sentences.
* If someone asks whether you used AI tools, say so plainly and say what you
  did yourself (the idea, the design decisions, the checks you ran, what you
  read). A straight answer goes down much better than being found out.

## Posting

* One place a day, on weekdays (morning U.S. Eastern time does best), and
  stay in the thread for the first few hours.
* Check each subreddit's rules on the day. Several limit self-promotion, and
  r/Python's Showcase posts need the "What my project does / Target audience
  / Comparison" sections.
* Suggested order: friends and a professor first (they catch mistakes
  privately), then r/econometrics, r/quant, r/Python, r/algotrading, and
  Show HN last, once the site has survived some traffic.
* Never ask anyone to upvote. Linking a friend to a post with "vote for
  this" can get the post removed and the account banned.
