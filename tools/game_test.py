"""Game lines: model vs DraftKings close (ESPN) and vs Kalshi pre-tip quotes."""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from scipy.stats import norm
from edge.kalshi import load_markets, load_candles, parse_event, game_index, quote_at, fee
from edge.data import american_to_prob

X = pd.read_pickle("work/gm_preds.pkl")
SIG = 14.2
X["p_mod"] = norm.cdf(X["xmargin"] / SIG)
ph, pa = X["ml_h"].map(american_to_prob), X["ml_a"].map(american_to_prob)
X["p_dk"] = ph / (ph + pa)
X["y"] = (X["m"] > 0).astype(int)

def ll(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4); return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

# --- 1. ATS vs DK close (-110-ish real odds) ---
Y = X[X.spread.notna() & ~X.spread_sign_odd].copy()
Y["edge"] = Y["xmargin"] + Y["spread"]           # >0: model likes home vs the line
Y["cover_h"] = np.sign(Y["m"] + Y["spread"])
def dec(a): return np.where(a > 0, 1 + a / 100, 1 + 100 / -a)
for thr in (0, 2, 4, 6):
    s = Y[Y.edge.abs() >= thr]
    side_h = s.edge > 0
    win = np.where(side_h, s.cover_h > 0, s.cover_h < 0); push = s.cover_h == 0
    odds = np.where(side_h, dec(s.sp_odds_h.fillna(-110)), dec(s.sp_odds_a.fillna(-110)))
    pnl = np.where(push, 0, np.where(win, odds - 1, -1))
    print(f"ATS edge>={thr}: n={len(s)} win%={win[~push].mean():.3f} roi={pnl.mean():+.3f}")
Y["t_edge"] = Y["xtotal"] - Y["total"]; Y["over"] = np.sign(Y["t"] - Y["total"])
for thr in (0, 3, 6):
    s = Y[Y.t_edge.abs() >= thr]; o = s.t_edge > 0
    win = np.where(o, s.over > 0, s.over < 0); push = s.over == 0
    odds = np.where(o, dec(s.o_odds.fillna(-110)), dec(s.u_odds.fillna(-110)))
    pnl = np.where(push, 0, np.where(win, odds - 1, -1))
    print(f"Totals edge>={thr}: n={len(s)} win%={win[~push].mean():.3f} roi={pnl.mean():+.3f}")
print("ML log-loss on DK games: model", round(ll(Y.p_mod, Y.y), 4), "DK no-vig", round(ll(Y.p_dk.dropna(), Y.y[Y.p_dk.notna()]), 4))

# --- 2. Kalshi moneyline ---
gi = game_index(X); C = load_candles("KXNBAGAME")
rows = []
for m in load_markets("KXNBAGAME"):
    pe = parse_event(m["event_ticker"]); gid = gi.get(pe) if pe else None
    if gid is None or m.get("result") not in ("yes", "no") or m["ticker"] not in C: continue
    team = m["ticker"].split("-")[-1]; team = {"GSW": "GS", "NOP": "NO", "NYK": "NY", "SAS": "SA", "UTA": "UTAH", "WAS": "WSH"}.get(team, team)
    rows.append({"gid": gid, "team": team, "ticker": m["ticker"], "y": int(m["result"] == "yes"), "c": C[m["ticker"]]})
K = pd.DataFrame(rows).merge(X[["gid", "home", "away", "tip", "p_mod", "p_dk", "xmargin", "spread"]], on="gid")
K["is_home"] = K.team == K.home
K["q"] = [quote_at(c, int(t.timestamp())) for c, t in zip(K.c, K.tip)]
K = K[K.q.notna()].copy()
K["bid"] = K.q.map(lambda q: q[0]); K["ask"] = K.q.map(lambda q: q[1]); K["mid"] = (K.bid + K.ask) / 2
K["pm"] = np.where(K.is_home, K.p_mod, 1 - K.p_mod); K["pd"] = np.where(K.is_home, K.p_dk, 1 - K.p_dk)
print("Kalshi ML markets with pre-tip quote:", len(K), "games", K.gid.nunique(), "spread", round((K.ask - K.bid).mean(), 4))
H = K[K.is_home]
print("log-loss  Kalshi mid", round(ll(H.mid, H.y), 4), " model", round(ll(H.pm, H.y), 4), " DK(n=%d)" % H.pd.notna().sum(),
      round(ll(H.pd[H.pd.notna()], H.y[H.pd.notna()]), 4), " Kalshi on same", round(ll(H.mid[H.pd.notna()], H.y[H.pd.notna()]), 4))
K = K.sort_values("tip", kind="stable")
half = K.tip.quantile(0.5)
for src in ("pm", "pd"):
    for thr in (0.0, 0.02, 0.04):
        s = K[K[src].notna()]
        ev = s[src] - s.ask - fee(s.ask)
        b = s[ev > thr]
        pnl = (b.y - b.ask - fee(b.ask)) / (b.ask + fee(b.ask))
        h1 = pnl[b.tip <= half]; h2 = pnl[b.tip > half]
        print(f"Kalshi ML buy when {src}-ask-fee>{thr}: n={len(b)} roi={pnl.mean():+.3f} | H1 {len(h1)} {h1.mean():+.3f} H2 {len(h2)} {h2.mean():+.3f}")
K.drop(columns=["c", "q"]).to_pickle("work/kalshi_ml.pkl")
