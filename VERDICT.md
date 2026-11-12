# NBA Edge — research verdict (2026-11-11)

**Short answer: nothing beats Kalshi's NBA prices after the spread and fee on fresh data.** Kalshi's pre-tip prices
for moneylines, spreads, totals and player props are well calibrated; our projections add nothing for game lines,
points, rebounds, assists or PRA. **Threes props are the one exception worth watching**: the model is as sharp as
Kalshi there and a model+Kalshi blend beats Kalshi's price on log-loss even on this season's fresh games, but betting
it was +10% in 2025-26 and only about break-even (90% CI −1% to +11%) in 2026-27. So the board ships **paper picks
only** (threes, with would-be units) and tracks them until a live record proves or kills the lead.
"Safe" near-lock legs lose 2–4% each and parlays compound that. Details and numbers below.

## Data (all free, public)

| Source | What | Coverage |
|---|---|---|
| ESPN site API | Box scores (minutes, pts, reb, ast, 3PM …, starters, DNP reasons), schedule, injuries | 5,726 games, Oct 2022 – Nov 2026 |
| ESPN pickcenter | DraftKings / consensus lines | 2022-23, 2025-26, 2026-27 only; behaves like an *earlier* line, not the close |
| Kalshi public API | Pre-tip bid/ask (hourly candles) for every settled market | Game lines since Apr 2025; props (PTS/REB/AST/3PT) since Nov 2025; PRA since Oct 2026 |

No free source has historical sportsbook **player-prop** odds or Pinnacle closes (The Odds API history is paid), so
sportsbook prop line-shopping cannot be backtested; Kalshi is the only real-price history.

Price used = Kalshi bid/ask at the last hourly candle before tip (0–60 min pre-game). Bets pay the ask (YES) or
1 − bid (NO) plus Kalshi's taker fee 0.07·p·(1−p). Props where the player never plays settle at a "fair price" on
Kalshi, so every test conditions on the player playing (no lineup leak for the prop's own player).

## Models (walk-forward: every prediction trained only on earlier games)

* **Game model** — decayed ridge team ratings (offence/defence, home, back-to-backs, injured-rotation scoring),
  refit before each game day. Margin RMSE 14.6 vs DraftKings line 14.3 on the same games.
* **Player projections** — gradient boosting (Poisson) per stat on recent minutes/role, per-minute rates (10-game
  EWMA + season), teammates out, opponent pace/defence/allowed rates, rest, home, game-model margin/total; monthly
  expanding-window refits. Mean-absolute error vs a last-5 EWMA baseline: points 4.45 vs 4.62, rebounds 1.86 vs 1.91,
  assists 1.32 vs 1.35, threes 0.88 vs 0.89, PRA 5.80 vs 6.10. A calibrated distribution (variance function + logistic
  calibration fitted before Nov 2025) turns the mean into P(stat ≥ line).

## Results

### Game lines
| Test | Result |
|---|---|
| Moneyline log-loss (1,545 games) | Kalshi mid **0.571**, DraftKings 0.571, our model 0.593 |
| Model ML bets on Kalshi (edge > 0/2/4%) | −7% to −8%; halves **+7% / −21%** (flip) |
| ATS / totals vs DraftKings lines | −3% to −9% |
| Kalshi spreads, this season (5,114 mkts) | log-loss mid 0.471 vs model 0.500; model bets **−15.5%** |
| Kalshi totals, this season (4,407 mkts) | log-loss mid 0.492 vs model 0.503; model bets +2.0% on ~155 games, but over 967 games (9,810 mkts): **−8.4%** (−17.4% / +0.9%) |

### Player props (Kalshi, all markets, no volume filter)
Bets = model edge > 3% after fee (thr .03); luck = same rule with outcomes redrawn from Kalshi's mid (≈ cost of spread + fee).
Log-loss "blend" = model+Kalshi blend fitted on one half, scored on the other.

| Stat | Markets (games) | Log-loss Kalshi / model | Model weight in blend | Model bets ROI (H1 / H2) | Luck baseline |
|---|---|---|---|---|---|
| Points | 24,579 (1,188) | 0.494 / 0.506 | 0.06–0.09 | −6.3% (−8.4 / −4.8) | −8.1% |
| Rebounds* | 13,321 (564) | 0.487 / 0.495 | 0.13–0.33, no gain | −4.9% (−2.2 / −7.3) | −8.9% |
| Assists* | 10,769 (562) | 0.444 / 0.456 | 0.11–0.18, no gain | −7.3% (−9.4 / −5.6) | −8.4% |
| PRA (2026-27 only) | 2,000 (154) | 0.589 / 0.608 | ~0 | −7.6% (−2.7 / −14.2) | −8.6% |
| **Threes** | 17,631 (1,080) | **0.4611 / 0.4610** | **~0.5, gains both ways** | **+4.0% (−0.6 / +7.5)** | −9.8% |

\*Rebounds and assists use every market of a random 40% of historical games (outcome-blind) plus all 2026-27 games.

Threes in detail:
* Unders the model likes: +7.3% (+6.7 / +7.9), game-clustered 90% CI +1.4% to +13.5% — but by phase: Nov–Jan +20%,
  Feb–Apr +6.6%, playoffs +12.5%, **2026-27 −0.4%** (695 bets).
* Strict forward test — blend fitted only on pre-season data, applied to all 155 games of 2026-27: log-loss blend
  0.4510 vs Kalshi 0.4521 (still better), bets at edge > 0/1/2/3%: +0.9% / −0.8% / +5.2% / +0.7%, every CI includes 0.
* Rookies/newcomers produce the largest "edges" live; the blend shrinks them toward Kalshi's price.

Other things tested and rejected: price bands without a model (long-shot overs are the worst bet on the board: points
0–10¢ overs hit 4.7% vs 6.9% paid; no band positive in both halves), model-unders on points (−0.6%), role players
(+4%/+3% on points/rebounds, −10% on assists → noise), blowout risk for starters (priced), teammates-out segments
(priced), early entry 3 h before tip (closing-line value negative for every stat: the price moves *away* from the
model before tip).

### "Safe" legs and parlays
Kalshi legs priced 85–97¢ (both sides, all four prop stats) won slightly less often than their price:

| Stat | Overs 85–97¢ ROI | Unders 85–97¢ ROI |
|---|---|---|
| Points | −4.2% | −1.8% |
| Rebounds | −2.3% | −2.4% |
| Assists | −3.1% | −2.6% |
| Threes | −4.4% | −3.2% |

Average −3% per leg. A "safe" 2× parlay needs about seven 91% legs (0.91⁷ ≈ 52% hit) and would lose ~20%; 2–5×
parlays from 2–3 legs need 55–70% legs, which lose 5–8% each here, so ~10–25% per parlay. Kalshi's NBA combos
(game-line legs only — no player props) cost buyers −12% at traded prices (555 combos, hit 29.7% vs 35.3% paid).
Parlays only make sense from legs that are individually +EV; the board's builder only offers those (threes, paper).

## Leaks caught (and why the first results were wrong)
1. **Final-volume filter.** Kalshi props close early when the over hits, so final traded volume depends on the
   outcome. Downloading only markets with ≥ 2,000 contracts made overs look +2.4 pts (points) and +6.2 pts (rebounds)
   better than reality and produced a fake "+4% model edge" and "overs win in Feb-Apr 2026". Fixed by using every
   market (or an outcome-blind random 40% of historical games for rebounds/assists/threes).
2. **Teammate share leak.** A "share of team scoring" feature summed over teammates who actually played — garbage-time
   players appear in blowouts — leaked the game script. Replaced with a share over the pre-game rotation minus listed
   absences.
3. **ESPN lines aren't closes.** "DraftKings no-vig > Kalshi ask" bets lost 45%: the ESPN line is earlier/stale.
4. Checked: rolling features recompute exactly from prior games only (0/300 mismatches); no duplicate player-tips.

## Same-game correlations (for parlay maths)
Walk-forward residual correlations: same player points–rebounds 0.25, points–assists 0.14, rebounds–assists 0.18;
teammates' points −0.06; opponents ≈ 0; a player's points vs the game-total residual +0.17, vs own-team margin +0.15.
So a player's over and the game over are positively correlated; teammates' overs slightly negatively.

## What to do with this
1. **Board (built, not published):** `edge/refresh.py` + `site/index.html` + `.github/workflows/refresh.yml`, the
   Tennis Edge pattern. Paper threes picks with would-be ¼-Kelly units, a parlay / round-robin builder from those legs
   (correlated hit chance, break-even payout), projections for every player, paper record with closing-line value,
   ledger. No secrets needed. Publishing needs a new public GitHub repo — your call.
2. **Promotion rule:** stake threes only after ~150 graded live bets with ROI and CLV above zero; otherwise drop it.
3. **Untested lead:** sportsbook line shopping vs Pinnacle / Kalshi for NBA props (the one thing that worked in tennis).
   There's no free history to test it; it needs the Odds API key going forward. `trading_bot_V2/bets/scan.py` already
   line-shops NBA main lines.
4. Not done: double-double / triple-double cross-market consistency (download stopped), quarter/half markets.

## Round 2 (2026-11-12): correlation, defence vs position, game script, ~50¢ picks

`tools/value_hunt.py` ran 139 rules over every prop (points, rebounds, assists, threes), each judged at the real
pre-tip price with fee, on both halves, on the fresh 2026-27 season alone, and with a game-clustered bootstrap.
A rule "passes" if both halves > 0, the fresh season > 0 and P(ROI ≤ 0) < 10%. With outcomes redrawn from Kalshi's
price (luck runs) **0 rules pass**; on the real outcomes **5 pass — all of them the model+Kalshi blend**.

| Idea | Result (ROI after fee) |
|---|---|
| Favourite team (win ≥ 70%) starters OVER | −17.8% (hit 39.5% at 42.3¢) |
| Favourite team bench OVER | −26.6% |
| Big underdog starters UNDER / OVER | −4.6% / −32% (assists) |
| Soft defence vs position (top 20%) OVER | −15.8% — the effect is real in box scores (+0.36 pts, +0.17 ast vs the stingiest fifth) but priced |
| Tough defence vs position UNDER | −6.1% |
| High implied team total OVER / low UNDER | −16.1% / −2.2% |
| Teammates out → starters OVER | −16.3% |
| Back-to-back starters UNDER; rested OVER; home OVER | −4.4%; −16.9%; −16.6% |
| Game total / win-prob moved before tip → props | −4% to −6% (props aren't lagging the game lines) |
| Follow or fade prop moves (5¢+ in last 3h) | negative |
| Kalshi ladder (line rich vs the player's other lines) | −1% to −4% |
| **Blend edge > 2%, ~50¢ (35–65¢)** | **+5.3%, 1,093 bets, hit 56.2% at 51.9¢; H1 +11.3%, H2 +4.2%, 2026-27 +3.9% (690)** |
| Blend edge > 2%, any price | +7.1% (H1 +17.3, H2 +5.3, 2026-27 +1.9%) |

The ~50¢ blend edge grows with the threshold (edge > 1/2/3/5%: +1.9/+5.3/+6.5/+7.9%) and is positive this season
at every threshold. It fires mostly on threes unders (+6.8%) and rebounds (both sides +7.6–10.5%); for points and
assists the blend is ~90% Kalshi's price, so it rarely disagrees. Board strategy `value_50c` stakes it small
(0.1 Kelly, max 0.25u, one pick per player per stat, games within 12 h); `value_any_price` is paper.

## Round 3 (2026-11-12): trying to improve it — and a robustness warning

* **Model features** (defence vs position inside the model, shrunk 3P%, short-term shot volume): accuracy barely moved
  (points MAE 4.447 → 4.445, threes 0.879 → 0.877) but the ~50¢ blend rule fell from **+5.3% to +1.3%** (this season
  +0.3%); 1 of 139 rules passed vs 1 in a luck run. A near-identical model halving-to-quartering the return means the
  +5.3% is fragile. The model stays on the original feature set and `value_50c` goes back to **paper**.
* On the original model: threes overs lose in every slice (−3.6%); threes unders (+6.7%) and rebounds (both sides)
  carry the result; edge-tiered stakes beat flat stakes (+5.9% vs +4.9% per unit); entering ~3h before tip is not
  clearly better.
* Next: price disagreements between venues (Kalshi vs Polymarket, which lists full NBA props with free price history)
  — the only kind of edge that held up in tennis, and it doesn't depend on a model.

## Round 4 (2026-11-12): other venues and the other side of the trade

**Kalshi vs Polymarket** (`tools/pm_match.py`, `pm_prices.py`, `pm_test.py`; 28,428 matched markets, 1,464 games,
Polymarket price taken at the same moment as the Kalshi quote):

| Market | Log-loss Kalshi / Polymarket | Buy the cheaper venue |
|---|---|---|
| Moneyline (1,456) | 0.5698 / 0.5698 | buy Polymarket when Kalshi says cheap: −2.3% |
| Totals (4,608) | 0.5964 / 0.5962 | ~100 real gaps, noise |
| Props (22,364) | 0.538 / 0.583 (threes 0.498 / 0.598) | buy Kalshi when Polymarket disagrees: −12% to −17% |

Polymarket NBA props look "wrong" only because their books are empty: live median bid–ask spread 23¢ (points) to
53¢ (assists) with $20–50 at the best price, so the historical prices are midpoints you can't trade. Moneylines (1¢
spread, ~$1,200 deep) and totals (4¢) are as sharp as Kalshi. No cross-venue edge.

**Resting (maker) orders on Kalshi props** (`tools/kalshi_trades.py`, `maker_test.py`; 330k pre-game fills, $10.3M
risked, 155 games of 2026-27; maker fee assumed 0): maker ROI **−0.6%** (H1 +2.0%, H2 −3.6%); takers −2.6% after fee.
No slice holds in both halves except fills in the last 15 minutes (+5.7%, one of ~30 slices — not trusted); makers
on the side our blend agrees with +1.0% (+3.5% / −2.3%). Same adverse-selection story as tennis: no edge.

## Bottom line (so far)

NBA prices on Kalshi are efficient. Nothing tested — model picks, favourite/defence correlations, price bands,
line moves, cross-venue gaps, or making markets — beats them robustly after costs. The model+Kalshi blend on threes
and rebounds is the best lead (+5.3% in backtest, fragile) and is paper-tracked on the board. The only approach with
real evidence of profit in this family of projects is **sportsbook line shopping against a sharp price** (tennis:
+3–8% vs Pinnacle no-vig), which needs the Odds API key going forward.

## Round 5 (2026-11-12): maker rewards

* **Kalshi Liquidity Incentive Program** (API `/incentive_programs`): 6,565 active programs, ~$100 per market per
  week (period_reward 1,000,000 in hundredths of a cent), target 1,000 contracts, score halves per cent from the best
  price. Almost all are slow markets — politics, awards, culture and sports futures (UCL top scorer, NHL season goals,
  MLB next team, NFL trades, Heisman). **No NBA game or prop markets.** Depth near the top of those books: median
  ~14k contracts (1–2¢ longshots 170k+), so 1,000 contracts would earn a few % of a $100 pool.
* **Kalshi Sports Prop Combo legs program**: 25% of parlay fees, paid monthly as bonus credit to makers on prop legs.
  Open NBA prop parlays: ~258k contracts, mostly 1–3¢ → fees of a few hundred dollars → a pool of tens to low hundreds
  of dollars a month vs millions of dollars of NBA prop maker volume. Negligible.
* **Polymarket**: maker rebates in sports = 15% of a 0.05·p(1−p) taker fee (≈0.4% of notional at 50¢). Liquidity
  rewards ≈ $127k/day over 11,313 markets, mostly 2028 election markets ($100–510/day each) plus some NFL games
  ($400–500/day, ~$800k already quoting); **$0 on NBA game markets**.
* Combined with the −0.6% maker result: rewards don't rescue NBA market-making. The only reward money of size is on
  slow futures/political markets, where it is a "paid to quote" yield play, not sports prediction.

## Round 6 (2026-11-12): opening prices (`tools/open_candles.py`, `tools/open_test.py`)

23,221 prop markets of 2026-27 (155 games) with the full price path from listing to tip; first sane quote a median
13.5 h before tip. Opening mid log-loss 0.4841 vs tip mid 0.4801 (openers less informed), but the opening spread is
6.7¢ vs 3.8¢ at tip, so every "buy at the open" price-band rule did worse than the same bet at tip. Drift open→tip
averages −0.1¢ (|drift| 2.6¢), no usable direction. Model+Kalshi blend at the open: −3.2% on all games (+0.3¢ CLV
— that's the absence leak: the model "knows" who sat out), −8% to −11% on games with no absences on either team
(CLV −1.1¢). Opening prices are not an edge; injury news is what moves them.
