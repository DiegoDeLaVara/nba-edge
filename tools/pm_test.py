"""Kalshi vs Polymarket, same market, same moment (last Kalshi hourly quote before tip vs last Polymarket price before
tip). Which is sharper, and does buying on Kalshi when Polymarket says it's cheap make money after Kalshi's fee?"""
import sys, json; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from edge.kalshi import fee


def ll(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4); return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))


def load(max_age_h=3.0):
    M = pd.read_pickle("work/pm_matched.pkl")
    Px = pd.DataFrame([json.loads(l) for l in open("work/pm_prices2.jsonl")])
    M = M.merge(Px, on="token", how="inner")
    # Polymarket price at the SAME moment as the Kalshi quote (no later information)
    def at(h, t):
        pre = [x for x in (h if isinstance(h, list) else []) if x[0] <= t]
        return (pre[-1][1], pre[-1][0]) if pre else (np.nan, np.nan)
    pt = [at(h, t) for h, t in zip(M.h, M.qts)]
    M["p_tip"] = [a for a, b in pt]; M["t_tip"] = [b for a, b in pt]
    ok = M.p_tip.notna() & (M.p_tip != 0.5) & ((M.qts - M.t_tip) <= 1.0 * 3600) & M.p_tip.between(0.02, 0.98)
    ok &= (M.ask - M.bid) <= 0.10
    M = M[ok].copy()
    M["mid"] = (M.bid + M.ask) / 2; M["pm"] = M.p_tip.astype(float)
    M = M.sort_values("tip", kind="stable").reset_index(drop=True)
    M["H1"] = M.tip <= M.tip.quantile(0.5)
    M["fresh"] = pd.to_datetime(M.tip, unit="s") >= "2026-10-01"
    M["grp"] = np.where(M.stat.isin(["pts", "reb", "ast", "fg3m"]), "props", M.stat)
    return M


def gap_bets(M, ref, thr):
    cy = M.ask + fee(M.ask); cn = (1 - M.bid) + fee(1 - M.bid)
    ey = ref - cy; en = (1 - ref) - cn
    side = np.where(ey >= en, "YES", "NO"); e = np.maximum(ey, en)
    x = np.where(side == "YES", (M.y - cy) / cy, ((1 - M.y) - cn) / cn)
    return e > thr, side, x


def clustered_p0(x, g, rng, nb=500):
    ug, inv = np.unique(g, return_inverse=True)
    s = np.bincount(inv, weights=x); c = np.bincount(inv)
    bs = [(s * w).sum() / (c * w).sum() for w in (np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug)) for _ in range(nb))]
    return float(np.mean(np.array(bs) <= 0)), np.quantile(bs, [.05, .95])


if __name__ == "__main__":
    M = load(); rng = np.random.default_rng(9)
    print(f"usable pairs {len(M)} ({M.gid.nunique()} games); by group {M.grp.value_counts().to_dict()}; this season {int(M.fresh.sum())}")
    print(f"mean |Kalshi mid - Polymarket| = {np.mean(np.abs(M.mid - M.pm)):.4f}; share with gap > 3c: {np.mean(np.abs(M.mid - M.pm) > .03):.3f}")
    for grp, S in [("ALL", M)] + list(M.groupby("stat")):
        if len(S) < 200:
            continue
        avg = (S.mid + S.pm) / 2
        line = f"{grp:6s} n={len(S):6d} log-loss Kalshi {ll(S.mid, S.y):.4f}  Polymarket {ll(S.pm, S.y):.4f}  average {ll(avg, S.y):.4f}"
        # blend fitted on the other half
        pb = np.zeros(len(S))
        for a, b in ((S.H1.values, ~S.H1.values), (~S.H1.values, S.H1.values)):
            lr = LogisticRegression(C=100).fit(np.c_[logit(S.mid[a]), logit(S.pm[a])], S.y[a])
            pb[b] = lr.predict_proba(np.c_[logit(S.mid[b]), logit(S.pm[b])])[:, 1]
        line += f"  blend(out-of-half) {ll(pb, S.y):.4f}"
        print(line)
    print("\nBuy on Kalshi when Polymarket's price says the Kalshi side is cheap (edge after Kalshi fee):")
    key = M.groupby(["gid", "stat"]).ngroup().values
    for grp in ["ALL", "props", "ml", "total"]:
        S = M if grp == "ALL" else M[M.grp == grp]
        for thr in (0.0, 0.02, 0.04, 0.06):
            m, side, x = gap_bets(S, S.pm, thr)
            if m.sum() < 50:
                continue
            p0, ci = clustered_p0(x[m], S.gid.values[m], rng)
            h1 = x[m & S.H1.values]; h2 = x[m & ~S.H1.values]; fr = x[m & S.fresh.values]
            # luck: outcomes redrawn from the Kalshi mid (i.e. if Kalshi were right)
            kk = S.groupby(["gid", "stat"]).ngroup().values
            luck = []
            for _ in range(200):
                u = rng.random(kk.max() + 1)[kk]
                y2 = (u < S.mid.values).astype(int)
                cy = S.ask + fee(S.ask); cn = (1 - S.bid) + fee(1 - S.bid)
                xl = np.where(side == "YES", (y2 - cy) / cy, ((1 - y2) - cn) / cn)
                luck.append(xl[m].mean())
            print(f"  {grp:5s} gap>{thr:.2f}: n={m.sum():5d} hit {np.where(side=='YES',S.y,1-S.y)[m].mean():.3f} price {np.where(side=='YES',S.ask,1-S.bid)[m].mean():.3f} "
                  f"ROI {x[m].mean():+.3f} | H1 {h1.mean():+.3f} H2 {h2.mean():+.3f} | season {fr.mean() if len(fr) else np.nan:+.3f} (n {len(fr)}) | "
                  f"P(ROI<=0) {p0:.3f} 90%CI {ci.round(3)} | if Kalshi were right: {np.mean(luck):+.3f}, p {np.mean(np.array(luck) >= x[m].mean()):.3f}")
    M.to_pickle("work/pm_test_rows.pkl")

    print("\nReverse: buy on POLYMARKET when Kalshi's mid says the Polymarket side is cheap (pay Polymarket price + half its typical spread, no Polymarket fee assumed):")
    SLIP = {"ml": 0.005, "total": 0.02}
    for grp in ("ml", "total"):
        S = M[M.grp == grp]
        if len(S) < 100:
            continue
        slip = SLIP[grp]
        cy = S.pm + slip; cn = (1 - S.pm) + slip
        ey = S.mid - cy; en = (1 - S.mid) - cn
        side = np.where(ey >= en, "YES", "NO"); e = np.maximum(ey, en)
        x = np.where(side == "YES", (S.y - cy) / cy, ((1 - S.y) - cn) / cn)
        for thr in (0.0, 0.02, 0.03, 0.05):
            m = (e > thr).values
            if m.sum() < 30:
                continue
            p0, ci = clustered_p0(x[m], S.gid.values[m], rng)
            print(f"  {grp:5s} gap>{thr:.2f}: n={m.sum():5d} ROI {x[m].mean():+.3f} | H1 {x[m & S.H1.values].mean():+.3f} H2 {x[m & ~S.H1.values].mean():+.3f} | "
                  f"season {x[m & S.fresh.values].mean() if (m & S.fresh.values).any() else np.nan:+.3f} (n {(m & S.fresh.values).sum()}) | P(ROI<=0) {p0:.3f} 90%CI {ci.round(3)}")
