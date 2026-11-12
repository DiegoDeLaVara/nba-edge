"""Edge test: walk-forward projections vs Kalshi pre-tip prop quotes.

  python tools/prop_test.py pts reb ast fg3m pra
For each stat: log-loss (Kalshi mid vs model vs blend fitted on the first half), betting at the real ask/bid with
Kalshi's fee, split into two time halves, and luck runs where outcomes are re-drawn from the Kalshi mid (one uniform
per player-game so a player's ladder of lines stays consistent).
"""
import sys, json; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from edge.kalshi import prop_rows, fee
from edge.dist import Dist, GRID

SERIES = {"pts": "KXNBAPTS", "reb": "KXNBAREB", "ast": "KXNBAAST", "fg3m": "KXNBA3PT", "pra": "KXNBAPRA"}
CUT = "2025-11-01"


def ll(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4); return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))


def bets(R, p, thr):
    """Back yes when p - ask - fee > thr, back no when (1-p) - (1-bid) - fee > thr. Returns (mask_yes, mask_no, pnl)."""
    cy = R["ask"] + fee(R["ask"]); cn = (1 - R["bid"]) + fee(1 - R["bid"])
    ey = p - cy; en = (1 - p) - cn
    by = (ey > thr) & (ey >= en); bn = (en > thr) & (en > ey)
    pnl = np.where(by, (R["y"] - cy) / cy, np.where(bn, ((1 - R["y"]) - cn) / cn, np.nan))
    return by, bn, pnl


def run(stat):
    G = pd.read_pickle("work/G.pkl"); P = pd.read_pickle("work/P.pkl")
    if stat == "pra":
        P = P.copy(); P["pra"] = P["pts"] + P["reb"] + P["ast"]
    R, miss = prop_rows(SERIES[stat], G, P, stat)
    W = pd.read_pickle(f"work/wf_{stat}.pkl")
    pre = W[W.tip < CUT]
    dist = Dist().fit(pre.mu.values, pre[stat].values.astype(float), GRID[stat])
    R = R.merge(W[["gid", "pid", "mu"]], on=["gid", "pid"], how="inner")
    R = R[(R.ask - R.bid) <= 0.10].copy()                       # drop junk books
    import os
    if os.environ.get("SAMPLE"):        # keep only outcome-blind sampled historical games (+ all live games)
        import zlib
        f = float(os.environ["SAMPLE"])
        ks = R["event"].map(lambda et: (zlib.crc32(et.split("-", 1)[1].encode()) % 1000) / 1000 < f)
        R = R[~R["hist"] | ks].copy()
    if os.environ.get("SINCE"):
        R = R[pd.to_datetime(R.tip, unit="s") >= os.environ["SINCE"]].copy()
    R["mid"] = (R.bid + R.ask) / 2
    R["p"] = dist.p_over(R.mu.values, R.k.values)
    R = R.sort_values("tip", kind="stable").reset_index(drop=True)
    half = R.tip.quantile(0.5); H1 = R.tip <= half
    print(f"\n=== {stat}: {len(R)} Kalshi markets ({R.gid.nunique()} games, {R.groupby(['gid','pid']).ngroups} player-games), "
          f"{pd.to_datetime(R.tip.min(), unit='s').date()}..{pd.to_datetime(R.tip.max(), unit='s').date()}, miss {miss}")
    print(f"mean spread {np.mean(R.ask - R.bid):.3f}; mean mid {R.mid.mean():.3f}; hit rate {R.y.mean():.3f}")
    print(f"log-loss  Kalshi mid {ll(R.mid, R.y):.4f}   model {ll(R.p, R.y):.4f}")
    # calibration of mid by bucket
    R["mb"] = pd.cut(R.mid, [0, .1, .2, .35, .5, .65, .8, .9, 1])
    cal = R.groupby("mb", observed=True).agg(n=("y", "size"), mid=("mid", "mean"), model=("p", "mean"), hit=("y", "mean"))
    print(cal.round(3).to_string())
    # blend fitted on H1, evaluated on H2 (and vice versa)
    for a, b in ((H1, ~H1), (~H1, H1)):
        Xa = np.c_[logit(R.mid[a]), logit(R.p[a])]; Xb = np.c_[logit(R.mid[b]), logit(R.p[b])]
        lr = LogisticRegression(C=100).fit(Xa, R.y[a])
        pb = lr.predict_proba(Xb)[:, 1]
        print(f"blend fit on {'H1' if a is H1 else 'H2'}: coefs mid {lr.coef_[0][0]:.2f} model {lr.coef_[0][1]:.2f}; "
              f"out-of-half ll blend {ll(pb, R.y[b]):.4f} vs mid {ll(R.mid[b], R.y[b]):.4f}")
    # betting
    rng = np.random.default_rng(7)
    key = R.groupby(["gid", "pid"]).ngroup().values
    for thr in (0.0, 0.03, 0.06, 0.10):
        by, bn, pnl = bets(R, R.p, thr)
        m = ~np.isnan(pnl)
        h1 = pnl[m & H1.values]; h2 = pnl[m & ~H1.values]
        luck = []
        for _ in range(200):
            u = rng.random(key.max() + 1)[key]
            Rs = R.assign(y=(u < R.mid).astype(int))
            _, _, pl = bets(Rs, R.p, thr)
            luck.append(np.nanmean(pl))
        luck = np.array(luck)
        print(f"thr {thr:.2f}: bets {m.sum()} (yes {by.sum()}, no {bn.sum()}) ROI {np.nanmean(pnl):+.3f} | H1 {len(h1)} {np.mean(h1) if len(h1) else np.nan:+.3f}"
              f" H2 {len(h2)} {np.mean(h2) if len(h2) else np.nan:+.3f} | luck mean {luck.mean():+.3f} p(luck>=obs) {(luck >= np.nanmean(pnl)).mean():.2f}")
    # baselines: always back NO / YES at the quote
    cy = R["ask"] + fee(R["ask"]); cn = (1 - R["bid"]) + fee(1 - R["bid"])
    pn = ((1 - R.y) - cn) / cn; py = (R.y - cy) / cy
    print(f"baseline always NO {pn.mean():+.3f} (H1 {pn[H1].mean():+.3f} H2 {pn[~H1].mean():+.3f}); always YES {py.mean():+.3f}")
    by, bn, pnl = bets(R, R.p, 0.03)
    for nm, mk in (("model YES", by), ("model NO", bn)):
        x = pnl[mk.values]
        print(f"  {nm} thr .03: n={mk.sum()} ROI {np.nanmean(x) if len(x) else np.nan:+.3f} | H1 {np.nanmean(pnl[mk.values & H1.values]):+.3f} H2 {np.nanmean(pnl[mk.values & ~H1.values]):+.3f}")
    # early entry + closing-line value: decide on the quote ~3h before tip, compare with the tip mid
    E = R[R.ask3.notna() & ((R.ask3 - R.bid3) <= 0.10)].copy()
    if len(E) > 200:
        Ee = E.rename(columns={"ask": "ask_t", "bid": "bid_t"}).rename(columns={"ask3": "ask", "bid3": "bid"})
        by3, bn3, pnl3 = bets(Ee, Ee.p, 0.03)
        clv = np.where(by3, E.mid - Ee.ask, np.where(bn3, Ee.bid - E.mid, np.nan))
        m3 = ~np.isnan(pnl3)
        print(f"early (tip-3h) bets thr .03: n={m3.sum()} ROI {np.nanmean(pnl3):+.3f}; CLV vs tip mid {np.nanmean(clv):+.4f}"
              f" (share positive {np.mean(clv[m3] > 0):.2f})")
    R.to_pickle(f"work/pt_{stat}{os.environ.get('TAG', '')}.pkl")
    json.dump(dist.to_json(), open(f"work/dist_{stat}.json", "w"))
    return R


if __name__ == "__main__":
    for s in sys.argv[1:]:
        run(s)
