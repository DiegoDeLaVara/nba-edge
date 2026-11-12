"""Defence vs position: how much players at each position (G / F / C) beat their own projection against a given opponent,
from earlier games only (EWMA, shifted). Also: how much the opponent allows to the player's ROLE (starter vs bench).
Checks whether it predicts the current game's residual. -> work/dvp.pkl (gid, pid, dvp_<stat>, dvp_n)"""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd

D = pd.read_pickle("work/D.pkl")[["gid", "pid", "team", "opp", "tip", "pos", "starter", "min_e10"]]
D["pg"] = np.where(D.pos.fillna("").str.contains("C"), "C", np.where(D.pos.fillna("").str.contains("G"), "G", "F"))
out = None
for st in ("pts", "reb", "ast", "fg3m"):
    W = pd.read_pickle(f"work/wf_{st}.pkl")[["gid", "pid", "mu", st]]
    X = D.merge(W, on=["gid", "pid"])
    X["res"] = X[st] - X["mu"]
    X = X[X.min_e10 >= 15]                                   # rotation players define the allowance
    # per opponent-game-position mean residual
    g = X.groupby(["opp", "pg", "gid", "tip"], as_index=False).agg(r=("res", "mean"), n=("res", "size")).sort_values("tip", kind="stable")
    g[f"dvp_{st}"] = g.groupby(["opp", "pg"])["r"].transform(lambda s: s.shift(1).ewm(halflife=12, min_periods=5).mean())
    feat = g[["opp", "pg", "gid", f"dvp_{st}"]]
    Y = D.merge(W, on=["gid", "pid"]).merge(feat, on=["opp", "pg", "gid"], how="left")
    Y["res"] = Y[st] - Y["mu"]
    ok = Y[f"dvp_{st}"].notna() & (Y.min_e10 >= 15)
    c = np.corrcoef(Y.loc[ok, f"dvp_{st}"], Y.loc[ok, "res"])[0, 1]
    # slope: residual per unit of dvp
    b = np.polyfit(Y.loc[ok, f"dvp_{st}"], Y.loc[ok, "res"], 1)[0]
    q = pd.qcut(Y.loc[ok, f"dvp_{st}"], 5)
    print(f"{st}: corr(dvp, residual) {c:+.4f}, slope {b:+.3f}; residual by dvp quintile:",
          Y.loc[ok].groupby(q, observed=True)["res"].mean().round(3).tolist())
    Y = Y[["gid", "pid", f"dvp_{st}"]]
    out = Y if out is None else out.merge(Y, on=["gid", "pid"], how="outer")
out.to_pickle("work/dvp.pkl")
