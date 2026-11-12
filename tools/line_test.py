"""This season's Kalshi spread / total ladders vs the team model (all markets, no volume filter)."""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from scipy.stats import norm
from edge.kalshi import load_markets, load_candles, parse_event, game_index, quote_at, fee, K2E
X = pd.read_pickle("work/gm_all.pkl"); gi = game_index(X); X = X.set_index("gid")
def ll(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4); return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
for ser in ("KXNBASPREAD", "KXNBATOTAL"):
    C = load_candles(ser); rows = []
    for m in load_markets(ser):
        if m.get("result") not in ("yes", "no") or m["ticker"] not in C: continue
        if m.get("hist") and ser != "KXNBATOTAL": continue
        pe = parse_event(m["event_ticker"]); gid = gi.get(pe) if pe else None
        if gid is None: continue
        g = X.loc[gid]; q = quote_at(C[m["ticker"]], int(g.tip.timestamp()))
        if not q or q[1] - q[0] > 0.10: continue
        k = float(m["floor_strike"])
        if ser == "KXNBASPREAD":
            team = K2E.get(m["ticker"].split("-")[-1].rstrip("0123456789"), m["ticker"].split("-")[-1].rstrip("0123456789"))
            mu = g.xmargin if team == g.home else -g.xmargin; sd = 14.2
        else:
            mu, sd = g.xtotal, 18.5
            if abs(k - mu) > 10: continue          # same pre-game strike filter as the historical download
        rows.append({"tip": g.tip, "bid": q[0], "ask": q[1], "y": int(m["result"] == "yes"), "p": 1 - norm.cdf((k - mu) / sd)})
    R = pd.DataFrame(rows).sort_values("tip", kind="stable").reset_index(drop=True); R["mid"] = (R.bid + R.ask) / 2
    H1 = (R.tip <= R.tip.quantile(.5)).values
    cy = R.ask + fee(R.ask); cn = 1 - R.bid + fee(1 - R.bid)
    ey, en = R.p - cy, (1 - R.p) - cn
    pnl = np.where((ey > .03) & (ey >= en), (R.y - cy) / cy, np.where(en > .03, ((1 - R.y) - cn) / cn, np.nan))
    print(f"{ser}: {len(R)} markets, spread {np.mean(R.ask-R.bid):.3f}; log-loss mid {ll(R.mid,R.y):.4f} model {ll(R.p,R.y):.4f}; "
          f"model bets (edge>3%) n={np.sum(~np.isnan(pnl))} ROI {np.nanmean(pnl):+.3f} | H1 {np.nanmean(pnl[H1]):+.3f} H2 {np.nanmean(pnl[~H1]):+.3f} | games {R.tip.nunique()}")
