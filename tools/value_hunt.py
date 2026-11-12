"""Hypothesis battery on every Kalshi prop (pts/reb/ast/3PM; complete or outcome-blind-sampled games), priced at the
real pre-tip ask/bid with Kalshi's fee. Each rule is judged on: ROI, both time halves, the fresh 2026-27 season alone,
and a game-clustered bootstrap P(ROI <= 0). Luck runs (outcomes redrawn from the Kalshi mid) give how many rules pass by
chance. -> work/vh_rows.pkl, prints a table."""
import sys, json; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from edge.kalshi import fee
from edge.dist import Dist

STATS = ["pts", "reb", "ast", "fg3m"]


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))


def frame():
    F = pd.read_pickle("work/D.pkl")[["gid", "pid", "team", "home", "pos", "min_e10", "n_prev", "start_e", "miss_min",
                                      "days_off", "rest", "b2b", "opp_b2b", "share_pts", "xmargin_t"]]
    M = pd.read_pickle("work/game_market.pkl")
    V = pd.read_pickle("work/dvp.pkl")
    out = []
    for st in STATS:
        R = pd.read_pickle(f"work/pt_{st}.pkl")[["ticker", "gid", "pid", "stat", "k", "bid", "ask", "mid", "y", "tip", "p", "bid3", "ask3"]].copy()
        R = R.merge(F, on=["gid", "pid"], how="left").merge(M.drop(columns=["tip"]), on="gid", how="left")
        R = R.merge(V[["gid", "pid", f"dvp_{st}"]].rename(columns={f"dvp_{st}": "dvp"}), on=["gid", "pid"], how="left")
        out.append(R)
    R = pd.concat(out, ignore_index=True)
    R = R.rename(columns={"home_x": "is_home", "home_y": "home_team"})
    ih = R.team == R.home_team
    R["pw"] = np.where(ih, R.ph, 1 - R.ph); R["pw3"] = np.where(ih, R.ph_3h, 1 - R.ph_3h)
    R["tt"] = np.where(ih, R.tt_h, R.tt_a); R["d_pw"] = R.pw - R.pw3
    R["mid3"] = (R.bid3 + R.ask3) / 2
    R = R.sort_values("tip", kind="stable").reset_index(drop=True)
    R["H1"] = R.tip <= R.tip.quantile(0.5)
    R["fresh"] = pd.to_datetime(R.tip, unit="s") >= "2026-10-01"
    # dvp quintiles within stat (thresholds from the first half only)
    R["dvp_z"] = np.nan
    for st in STATS:
        m = R.stat == st
        mu, sd = R.loc[m & R.H1, "dvp"].mean(), R.loc[m & R.H1, "dvp"].std()
        R.loc[m, "dvp_z"] = (R.loc[m, "dvp"] - mu) / sd
    # blends fitted on the other half (per stat): market+model, and market+model+dvp+game context
    R["pb"] = np.nan; R["pb2"] = np.nan
    for st in STATS:
        for a, b in ((R.H1, ~R.H1), (~R.H1, R.H1)):
            ma, mb = (R.stat == st) & a, (R.stat == st) & b
            X1 = lambda d: np.c_[logit(d.mid), logit(d.p)]
            lr = LogisticRegression(C=100).fit(X1(R[ma]), R.y[ma]); R.loc[mb, "pb"] = lr.predict_proba(X1(R[mb]))[:, 1]
            X2 = lambda d: np.c_[logit(d.mid), logit(d.p), d.dvp_z.fillna(0), (d.pw.fillna(.5) - .5), (d.tt.fillna(115) - 115) / 5]
            lr2 = LogisticRegression(C=100).fit(X2(R[ma]), R.y[ma]); R.loc[mb, "pb2"] = lr2.predict_proba(X2(R[mb]))[:, 1]
    # Kalshi ladder fit (leave-one-strike-out): what the player's other lines imply for this one
    R["lad"] = np.nan
    grid = np.linspace(0.2, 45, 400)
    for st in STATS:
        dist = Dist.from_json(json.load(open(f"work/dist_{st}.json")))
        S = R[R.stat == st]
        for (g, p), d in S.groupby(["gid", "pid"]):
            if len(d) < 3:
                continue
            ks = d.k.values; mids = np.clip(d.mid.values, .02, .98)
            P = dist.p_over(np.repeat(grid, len(ks)), np.tile(ks, len(grid))).reshape(len(grid), len(ks))
            E = (logit(P) - logit(mids)[None, :]) ** 2
            tot = E.sum(1)
            for j, idx in enumerate(d.index):
                best = np.argmin(tot - E[:, j])
                R.at[idx, "lad"] = P[best, j]
    return R


def pnl(R, side):
    cy = R.ask + fee(R.ask); cn = (1 - R.bid) + fee(1 - R.bid)
    return np.where(side == "YES", (R.y - cy) / cy, ((1 - R.y) - cn) / cn), np.where(side == "YES", R.p - cy, (1 - R.p) - cn)


def judge(R, mask, side, rng, nb=400):
    S = R[mask]
    if len(S) < 80:
        return None
    x, _ = pnl(S, side)
    g = S.gid.values; ug, inv = np.unique(g, return_inverse=True)
    sums = np.bincount(inv, weights=x); cnt = np.bincount(inv)
    bs = []
    for _ in range(nb):
        w = np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug))
        bs.append((sums * w).sum() / (cnt * w).sum())
    h1 = x[S.H1.values]; h2 = x[~S.H1.values]; fr = x[S.fresh.values]
    return {"n": len(S), "games": len(ug), "roi": x.mean(), "H1": h1.mean() if len(h1) else np.nan, "H2": h2.mean() if len(h2) else np.nan,
            "fresh": fr.mean() if len(fr) else np.nan, "n_fresh": len(fr), "p0": float(np.mean(np.array(bs) <= 0)),
            "price": (S.ask if side == "YES" else 1 - S.bid).mean(), "hit": (S.y if side == "YES" else 1 - S.y).mean()}


def rules(R):
    starter = R.min_e10 >= 28; bench = R.min_e10 < 22
    mid50 = (R.mid >= 0.35) & (R.mid <= 0.65)
    cy = R.ask + fee(R.ask); cn = (1 - R.bid) + fee(1 - R.bid)
    ey = lambda q: q - cy; en = lambda q: (1 - q) - cn
    L = []
    for st in STATS + ["all"]:
        s = (R.stat == st) if st != "all" else pd.Series(True, index=R.index)
        L += [
            (st, "fav team (win>=70%) starters OVER", s & starter & (R.pw >= .7), "YES"),
            (st, "fav team (win>=70%) bench OVER", s & bench & (R.pw >= .7), "YES"),
            (st, "big dog (win<=30%) starters UNDER", s & starter & (R.pw <= .3), "NO"),
            (st, "big dog (win<=30%) starters OVER", s & starter & (R.pw <= .3), "YES"),
            (st, "close game (45-55%) starters OVER", s & starter & R.pw.between(.45, .55), "YES"),
            (st, "high team total (>=120) OVER", s & (R.tt >= 120), "YES"),
            (st, "low team total (<=110) UNDER", s & (R.tt <= 110), "NO"),
            (st, "game total rose 1.5+ last 3h OVER", s & (R.d_tot >= 1.5), "YES"),
            (st, "game total fell 1.5+ last 3h UNDER", s & (R.d_tot <= -1.5), "NO"),
            (st, "team win% rose 3+ pts last 3h OVER", s & (R.d_pw >= .03), "YES"),
            (st, "team win% fell 3+ pts last 3h UNDER", s & (R.d_pw <= -.03), "NO"),
            (st, "soft defence vs position (top 20%) OVER", s & (R.dvp_z >= .84), "YES"),
            (st, "tough defence vs position (bottom 20%) UNDER", s & (R.dvp_z <= -.84), "NO"),
            (st, "soft DvP + ~50c OVER", s & (R.dvp_z >= .84) & mid50, "YES"),
            (st, "tough DvP + ~50c UNDER", s & (R.dvp_z <= -.84) & mid50, "NO"),
            (st, "back-to-back starters UNDER", s & starter & (R.b2b == 1), "NO"),
            (st, "opponent on back-to-back OVER", s & (R.opp_b2b == 1), "YES"),
            (st, "rested 3+ days OVER", s & (R.rest >= 3), "YES"),
            (st, "home OVER", s & (R.is_home == True), "YES"),
            (st, "teammates out (40+ min) starters OVER", s & starter & (R.miss_min >= 40), "YES"),
            (st, "model edge>3% ~50c", s & mid50 & ((ey(R.p) > .03) | (en(R.p) > .03)), "MODEL"),
            (st, "blend edge>2% ~50c", s & mid50 & ((ey(R.pb) > .02) | (en(R.pb) > .02)), "BLEND"),
            (st, "blend edge>2% any price", s & ((ey(R.pb) > .02) | (en(R.pb) > .02)), "BLEND"),
            (st, "blend+DvP+game edge>2%", s & ((ey(R.pb2) > .02) | (en(R.pb2) > .02)), "BLEND2"),
            (st, "ladder: line rich vs player's other lines -> UNDER", s & (R.mid - R.lad >= .04), "NO"),
            (st, "ladder: line cheap vs player's other lines -> OVER", s & (R.lad - R.mid >= .04), "YES"),
            (st, "prop moved down 5c+ last 3h -> OVER (fade move)", s & (R.mid - R.mid3 <= -.05), "YES"),
            (st, "prop moved up 5c+ last 3h -> UNDER (fade move)", s & (R.mid - R.mid3 >= .05), "NO"),
            (st, "prop moved up 5c+ last 3h -> OVER (follow)", s & (R.mid - R.mid3 >= .05), "YES"),
        ]
    return L


def side_for(R, mask, side):
    if side in ("YES", "NO"):
        return np.full(mask.sum(), side)
    q = {"MODEL": R.p, "BLEND": R.pb, "BLEND2": R.pb2}[side][mask]
    S = R[mask]
    cy = S.ask + fee(S.ask); cn = (1 - S.bid) + fee(1 - S.bid)
    return np.where(q - cy >= (1 - q) - cn, "YES", "NO")


def evaluate(R, rng):
    res = []
    for st, name, mask, side in rules(R):
        mask = mask.fillna(False).values if hasattr(mask, "fillna") else mask
        if mask.sum() < 80:
            continue
        sd = side_for(R, mask, side)
        S = R[mask].copy()
        cy = S.ask + fee(S.ask); cn = (1 - S.bid) + fee(1 - S.bid)
        x = np.where(sd == "YES", (S.y - cy) / cy, ((1 - S.y) - cn) / cn)
        g = S.gid.values; ug, inv = np.unique(g, return_inverse=True)
        sums = np.bincount(inv, weights=x); cnt = np.bincount(inv)
        bs = [(sums * w).sum() / (cnt * w).sum() for w in (np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug)) for _ in range(300))]
        H1 = S.H1.values; fr = S.fresh.values
        res.append({"stat": st, "rule": name, "n": len(S), "games": len(ug), "price": float(np.where(sd == "YES", S.ask, 1 - S.bid).mean()),
                    "hit": float(np.where(sd == "YES", S.y, 1 - S.y).mean()), "roi": x.mean(), "H1": x[H1].mean() if H1.any() else np.nan,
                    "H2": x[~H1].mean() if (~H1).any() else np.nan, "fresh": x[fr].mean() if fr.any() else np.nan, "n_fresh": int(fr.sum()),
                    "p0": float(np.mean(np.array(bs) <= 0))})
    T = pd.DataFrame(res)
    T["pass"] = (T.H1 > 0) & (T.H2 > 0) & (T.fresh > 0) & (T.p0 < 0.10) & (T.n_fresh >= 50)
    return T


if __name__ == "__main__":
    R = frame(); R.to_pickle("work/vh_rows.pkl")
    rng = np.random.default_rng(42)
    T = evaluate(R, rng); T.to_pickle("work/vh_table.pkl")
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 400)
    print(T.round(3).sort_values("roi", ascending=False).to_string(index=False))
    print("PASSING (both halves >0, fresh season >0, P(ROI<=0)<10%):", int(T["pass"].sum()), "of", len(T))
    # luck: redraw outcomes from the Kalshi mid (per player-game uniform), rerun the battery
    key = R.groupby(["gid", "pid"]).ngroup().values
    for i in range(3):
        u = rng.random(key.max() + 1)[key]
        Rl = R.assign(y=(u < R.mid).astype(int))
        Tl = evaluate(Rl, rng)
        print(f"luck run {i}: rules passing {int(Tl['pass'].sum())}, best ROI {Tl.roi.max():+.3f}")
