"""Team ratings game model: decayed ridge regression on team points (offence / defence / home / rest / absences),
refit before each game day using only earlier days. Gives expected margin and total for every game."""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from edge.features import TEAMS

TL = sorted(TEAMS)
IX = {t: i for i, t in enumerate(TL)}
NT = len(TL)


def gday(tip):
    return (tip - pd.Timedelta(hours=10)).dt.date


def design(R):
    n = len(R)
    X = np.zeros((n, 2 * NT + 6))
    X[np.arange(n), R["team"].map(IX).values] = 1
    X[np.arange(n), NT + R["opp"].map(IX).values] = -1
    X[:, 2 * NT] = np.where(R["neutral"], 0, np.where(R["home"], 1, -1))
    X[:, 2 * NT + 1] = R["b2b"].values
    X[:, 2 * NT + 2] = R["opp_b2b"].values
    X[:, 2 * NT + 3] = R["miss_pts"].values / 10
    X[:, 2 * NT + 4] = R["opp_miss_pts"].values / 10
    X[:, 2 * NT + 5] = R["playoff"].values
    return X


def team_rows(G, absn):
    G = G[G["home"].isin(TEAMS) & G["away"].isin(TEAMS)].copy()
    rows = []
    for side, opp, pf in (("home", "away", "hs"), ("away", "home", "as")):
        r = pd.DataFrame({"gid": G["gid"], "tip": G["tip"], "season": G["season"], "team": G[side], "opp": G[opp],
                          "home": side == "home", "neutral": G["neutral"], "pts": G[pf], "playoff": (G["stype"] == 3).astype(float)})
        rows.append(r)
    R = pd.concat(rows).sort_values(["tip", "gid"], kind="stable").reset_index(drop=True)
    R["rest"] = R.groupby("team")["tip"].diff().dt.total_seconds() / 86400
    R["b2b"] = (R["rest"] < 1.6).astype(float)
    a = absn.rename(columns={"miss_pts": "miss_pts"})[["gid", "team", "miss_pts", "miss_min"]]
    R = R.merge(a, on=["gid", "team"], how="left").fillna({"miss_pts": 0, "miss_min": 0})
    o = R[["gid", "team", "b2b", "miss_pts"]].rename(columns={"team": "opp", "b2b": "opp_b2b", "miss_pts": "opp_miss_pts"})
    R = R.merge(o, on=["gid", "opp"], how="left")
    R["day"] = gday(R["tip"])
    return R


def walk_forward(R, hl_days=45, alpha=2.0, start=None):
    days = sorted(R["day"].unique())
    preds = []
    for d in days:
        if start and d < start:
            continue
        tr = R[(R["day"] < d) & (R["day"] >= d - pd.Timedelta(days=400))]
        te = R[R["day"] == d]
        if len(tr) < 400:
            continue
        age = np.array([(d - x).days for x in tr["day"]], dtype=float)
        w = 0.5 ** (age / hl_days)
        # season change: last season's games count less
        w *= np.where(tr["season"].values != te["season"].iloc[0], 0.5, 1.0)
        m = Ridge(alpha=alpha).fit(design(tr), tr["pts"].values, sample_weight=w)
        p = te[["gid", "team"]].copy(); p["xpts"] = m.predict(design(te))
        preds.append(p)
    return pd.concat(preds)


def game_preds(G, R, P):
    """Merge team point predictions into game rows: xmargin (home - away), xtotal."""
    h = P.rename(columns={"team": "home", "xpts": "xh"}); a = P.rename(columns={"team": "away", "xpts": "xa"})
    out = G.merge(h, on=["gid", "home"]).merge(a, on=["gid", "away"])
    out["xmargin"] = out["xh"] - out["xa"]; out["xtotal"] = out["xh"] + out["xa"]
    return out
