# NBA Edge

NBA research + phone board, built the Tennis Edge way: free public data, walk-forward models, tests against real
Kalshi prices after the fee, and a GitHub Pages board refreshed by GitHub Actions. **Read [VERDICT.md](VERDICT.md)
first** — the short version is that Kalshi's NBA prices are sharp and only threes props show a (paper-tracked) lead.

## What runs where

| Piece | Where | What |
|---|---|---|
| `tools/espn_collect.py` | local, once | ESPN box scores + pickcenter lines since Oct 2022 → `work/espn_games.jsonl.gz` |
| `tools/kalshi_markets.py`, `tools/kalshi_candles.py` | local | Every settled Kalshi NBA market (live + /historical) and hourly bid/ask candles. **No final-volume filter** (it leaks outcomes); `--sample` picks whole games at random |
| `edge/features.py`, `edge/gamemodel.py`, `edge/props.py`, `edge/dist.py` | both | Leak-free features, team ratings, per-stat gradient boosting, P(stat ≥ line) calibration |
| `tools/proj_wf.py`, `tools/game_test.py`, `tools/line_test.py`, `tools/prop_test.py`, `tools/band_test.py`, `tools/phase.py`, `tools/bench_check.py` | local | Walk-forward accuracy and edge tests (halves, luck runs, game-clustered bootstrap) |
| `edge/refresh.py` | cloud, every 20 min | ESPN results → history; daily retrain; today's projections (ESPN schedule + injury report); open Kalshi props; threes priced with the model+Kalshi blend; paper picks with would-be units; pick log, grading, record → `board.json` |
| `site/index.html` | GitHub Pages | Picks, Parlays (correlated hit chance, break-even payout, round robin), Games (projections), Record, How |

Rolling state (`hist_*.csv.gz`, `models.pkl`, `picklog.json`, `board.json`) lives on the `gh-pages` branch, which is
force-pushed each run. No API keys are needed (Kalshi and ESPN are public).

## Local commands

```
..\tennis-edge\.venv\Scripts\python.exe -m edge.refresh --data out      # one refresh into ./out
..\tennis-edge\.venv\Scripts\python.exe tools\prop_test.py fg3m          # re-run the threes test
```

Not financial advice.
