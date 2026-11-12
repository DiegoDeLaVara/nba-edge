"""Did resting (maker) orders on Kalshi NBA props make money before tip? For every pre-game trade, the maker took the
opposite side of the taker at the trade price. Maker fee assumed 0 (series fee_type 'quadratic' = taker fees only).
Returns are per dollar risked by the maker."""
import sys, json; sys.path.insert(0, ".")
import numpy as np, pandas as pd

rows = []
for l in open("work/kt_trades.jsonl"):
    t, tr = json.loads(l)
    for ts, side, p, c in tr:
        rows.append((t, ts, side, p, c))
T = pd.DataFrame(rows, columns=["ticker", "ts", "taker", "p", "c"])
R = pd.concat([pd.read_pickle(f"work/pt_{s}.pkl").assign(stat=s) for s in ("pts", "reb", "ast", "fg3m")])
R = R[["ticker", "gid", "stat", "y", "tip", "mid", "p"]].rename(columns={"p": "p_model"})
V = pd.read_pickle("work/vh_rows.pkl")[["ticker", "pb"]].drop_duplicates("ticker")
T = T.merge(R, on="ticker").merge(V, on="ticker", how="left")
T = T[(T.ts < T.tip) & (T.p > 0.01) & (T.p < 0.99)].copy()
T["mins"] = (T.tip - T.ts) / 60
# maker: if taker bought YES at p, maker is short YES (holds NO, cost 1-p); if taker bought NO, maker is long YES (cost p)
T["maker"] = np.where(T.taker == "yes", "sells YES (holds NO)", "buys YES")
T["cost"] = np.where(T.taker == "yes", 1 - T.p, T.p)
T["pnl"] = np.where(T.taker == "yes", T.p - T.y, T.y - T.p)          # per contract
T["w"] = T.c * T.cost                                                  # dollars risked
T["day"] = pd.to_datetime(T.tip, unit="s").dt.date
half = T.tip.quantile(0.5); T["H1"] = T.tip <= half
rng = np.random.default_rng(3)


def summ(S):
    if S.w.sum() <= 0 or len(S) < 30:
        return None
    roi = (S.pnl * S.c).sum() / S.w.sum()
    g = S.gid.values; ug, inv = np.unique(g, return_inverse=True)
    num = np.bincount(inv, weights=(S.pnl * S.c).values); den = np.bincount(inv, weights=S.w.values)
    bs = [(num * k).sum() / (den * k).sum() for k in (np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug)) for _ in range(400))]
    h1, h2 = S[S.H1], S[~S.H1]
    f = lambda d: (d.pnl * d.c).sum() / d.w.sum() if d.w.sum() > 0 else np.nan
    return {"trades": len(S), "contracts": round(S.c.sum()), "risked$": round(S.w.sum()), "maker_roi": round(roi, 4),
            "H1": round(f(h1), 4), "H2": round(f(h2), 4), "P(roi<=0)": round(float(np.mean(np.array(bs) <= 0)), 3), "games": len(ug)}


out = []
def add(name, S):
    r = summ(S)
    if r:
        out.append({"slice": name, **r})

add("ALL pre-game fills", T)
for m, S in T.groupby("maker"):
    add(f"maker {m}", S)
for st, S in T.groupby("stat"):
    add(f"{st}", S)
    for m, S2 in S.groupby("maker"):
        add(f"{st} maker {m}", S2)
for lo, hi in ((0, 15), (15, 60), (60, 180), (180, 1e9)):
    add(f"fills {lo}-{hi if hi < 1e9 else '+'} min before tip", T[(T.mins >= lo) & (T.mins < hi)])
for lo, hi in ((0.01, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 0.99)):
    for m, S in T[(T.p >= lo) & (T.p < hi)].groupby("maker"):
        add(f"price {lo:.2f}-{hi:.2f} maker {m}", S)
# model-guided making: only keep fills where our blend agrees the maker side is the right one
agree = np.where(T.taker == "yes", T.pb < T.p, T.pb > T.p)
add("fills where blend agrees with maker side", T[agree & T.pb.notna()])
add("fills where blend disagrees", T[~agree & T.pb.notna()])
# taker side for comparison (pays the fee): taker ROI = -maker pnl - fee on the taker's cost
fee = 0.07 * T.p * (1 - T.p)
tk_cost = np.where(T.taker == "yes", T.p, 1 - T.p) + fee
print(f"Takers before tip: ROI {(-(T.pnl) * T.c - fee * T.c).sum() / (tk_cost * T.c).sum():+.4f} (after fee)")
pd.set_option("display.width", 220)
print(pd.DataFrame(out).to_string(index=False))
