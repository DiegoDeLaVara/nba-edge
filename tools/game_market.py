"""Per-game Kalshi market view just before tip (and 3h before): home win prob (ML mid), market total (strike where the
total ladder crosses 50%), implied margin and team totals. Pre-game information only. -> work/game_market.pkl"""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from scipy.stats import norm
from edge.kalshi import load_markets, load_candles, parse_event, game_index, quote_at

G = pd.read_pickle("work/G.pkl"); gi = game_index(G); tip = dict(zip(G.gid, G.tip))
SIG = 13.0          # margin sd used to turn a win prob into an expected margin


def ladder_line(pts):
    """pts: [(strike, p_over)] -> strike where p_over = 0.5 (linear interpolation on the monotone part)."""
    pts = sorted(pts)
    for (k1, p1), (k2, p2) in zip(pts, pts[1:]):
        if p1 >= 0.5 >= p2 and p1 != p2:
            return k1 + (p1 - 0.5) / (p1 - p2) * (k2 - k1)
    return np.nan


rows = {}
# moneyline
C = load_candles("KXNBAGAME")
for m in load_markets("KXNBAGAME"):
    pe = parse_event(m["event_ticker"]); gid = gi.get(pe) if pe else None
    if gid is None or m["ticker"] not in C:
        continue
    team = pe[2]   # home
    code = m["ticker"].split("-")[-1]
    from edge.kalshi import K2E
    team_m = K2E.get(code, code)
    for lab, off in (("", 0), ("_3h", 3 * 3600)):
        q = quote_at(C[m["ticker"]], int(tip[gid].timestamp()) - off)
        if q:
            mid = (q[0] + q[1]) / 2
            r = rows.setdefault(gid, {"gid": gid})
            r.setdefault(f"ph{lab}_list", []).append(mid if team_m == pe[2] else 1 - mid)
# totals ladder
C = load_candles("KXNBATOTAL")
lad = {}
for m in load_markets("KXNBATOTAL"):
    pe = parse_event(m["event_ticker"]); gid = gi.get(pe) if pe else None
    if gid is None or m["ticker"] not in C:
        continue
    for lab, off in (("", 0), ("_3h", 3 * 3600)):
        q = quote_at(C[m["ticker"]], int(tip[gid].timestamp()) - off)
        if q and q[1] - q[0] <= 0.10:
            lad.setdefault((gid, lab), []).append((float(m["floor_strike"]), (q[0] + q[1]) / 2))
for (gid, lab), pts in lad.items():
    rows.setdefault(gid, {"gid": gid})[f"tot{lab}"] = ladder_line(pts)
out = []
for gid, r in rows.items():
    d = {"gid": gid}
    for lab in ("", "_3h"):
        l = r.get(f"ph{lab}_list")
        d[f"ph{lab}"] = float(np.mean(l)) if l else np.nan
        d[f"tot{lab}"] = r.get(f"tot{lab}", np.nan)
    out.append(d)
M = pd.DataFrame(out).merge(G[["gid", "home", "away", "tip"]], on="gid")
M["marg"] = SIG * norm.ppf(M.ph.clip(0.01, 0.99))          # expected home margin
M["tt_h"] = (M.tot + M.marg) / 2; M["tt_a"] = (M.tot - M.marg) / 2
M["d_tot"] = M.tot - M.tot_3h; M["d_ph"] = M.ph - M.ph_3h
M.to_pickle("work/game_market.pkl")
print(len(M), "games;", M[["ph", "tot", "marg", "tt_h", "d_tot", "d_ph"]].describe().round(3).to_string())
