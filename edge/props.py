"""Player projections: walk-forward gradient boosting (Poisson loss) on leak-free features, plus a negative-binomial
spread around the projected mean, giving P(stat >= k) for any prop line."""
import numpy as np
import pandas as pd
from scipy.stats import nbinom
from sklearn.ensemble import HistGradientBoostingRegressor

STATS = ["pts", "reb", "ast", "fg3m", "pra"]
BASE = ["min_e3", "min_e10", "min_last", "start_e", "start_last", "n_prev", "n_season", "days_off", "home",
        "fga_pm", "fta_pm", "fg3a_pm", "tov_pm", "ast_pm", "reb_pm", "oreb_pm", "pts_pm", "fg3m_pm", "stl_pm", "blk_pm",
        "pts_e5", "reb_e5", "ast_e5", "fg3m_e5", "pts_spm", "reb_spm", "ast_spm", "fg3m_spm",
        "miss_min", "miss_pts", "miss_ast", "miss_reb", "miss_n", "opp_miss_pts", "opp_miss_min",
        "b2b", "opp_b2b", "rest", "xmargin_t", "xtotal", "xpts_t", "xpts_o", "pace_e", "opp_pace_e", "opp_drtg_e",
        "opp_a_reb_e", "opp_a_ast_e", "opp_a_fg3a_e", "playoff", "pos_g", "pos_c", "share_min", "share_pts", "avail_min"]


def build(Q, A, T, GM):
    """Q: played player-games with features; A: absences; T: team context; GM: game model preds (per gid)."""
    D = Q.copy()
    D["pra"] = D["pts"] + D["reb"] + D["ast"]
    D["pra_e5"] = D["pts_e5"] + D["reb_e5"] + D["ast_e5"]
    a = A[["gid", "team", "miss_min", "miss_pts", "miss_ast", "miss_reb", "miss_n", "roster_pts", "roster_min"]]
    D = D.merge(a, on=["gid", "team"], how="left")
    ao = A[["gid", "team", "miss_pts", "miss_min"]].rename(columns={"team": "opp", "miss_pts": "opp_miss_pts", "miss_min": "opp_miss_min"})
    D = D.merge(ao, on=["gid", "opp"], how="left")
    for c in ("miss_min", "miss_pts", "miss_ast", "miss_reb", "miss_n", "opp_miss_pts", "opp_miss_min", "roster_pts", "roster_min"):
        D[c] = D[c].fillna(0.0)
    t = T[["gid", "team", "rest", "pace_e", "drtg_e", "a_reb_e", "a_ast_e", "a_fg3a_e"]]
    D = D.merge(t[["gid", "team", "rest", "pace_e"]], on=["gid", "team"], how="left")
    to = t.rename(columns={"team": "opp", "rest": "opp_rest", "pace_e": "opp_pace_e", "drtg_e": "opp_drtg_e",
                           "a_reb_e": "opp_a_reb_e", "a_ast_e": "opp_a_ast_e", "a_fg3a_e": "opp_a_fg3a_e"})
    D = D.merge(to, on=["gid", "opp"], how="left")
    D["b2b"] = (D["rest"] < 1.6).astype(float); D["opp_b2b"] = (D["opp_rest"] < 1.6).astype(float)
    g = GM[["gid", "home", "away", "xh", "xa", "xmargin", "xtotal"]].rename(columns={"home": "h", "away": "a"})
    D = D.merge(g, on="gid", how="left")
    ih = D["team"] == D["h"]
    D["xmargin_t"] = np.where(ih, D["xmargin"], -D["xmargin"])
    D["xpts_t"] = np.where(ih, D["xh"], D["xa"]); D["xpts_o"] = np.where(ih, D["xa"], D["xh"])
    D["home"] = D["home"].astype(float)
    D["playoff"] = (D["stype"] == 3).astype(float)
    D["pos_g"] = D["pos"].fillna("").str.contains("G").astype(float)
    D["pos_c"] = D["pos"].fillna("").str.contains("C").astype(float)
    # player's share of team scoring/minutes (prior form), so the model can hand missing usage to the right players
    D["exp_pts"] = D["pts_pm"] * D["min_e10"]
    # share of the team's expected scoring among players available before tip (roster minus known absences)
    known = D["roster_pts"] > 0            # no team game in the last 21 days (season openers): unknown
    D["share_pts"] = np.where(known, D["exp_pts"] / (D["roster_pts"] - D["miss_pts"]).clip(lower=20), np.nan)
    D["avail_min"] = np.where(known, (D["roster_min"] - D["miss_min"]).clip(lower=100), np.nan)
    D["share_min"] = D["min_e10"] / 240
    return D


def feats(stat):
    f = list(BASE)
    if stat == "pra":
        f.append("pra_e5")
    return f


def walk_forward(D, stat, start="2023-11-01", min_prev=3, params=None):
    """Monthly expanding-window refits; predicts every played game from `start` on with a model trained on earlier
    games only (training rows must have tipped before the first day of the predicted month)."""
    params = params or dict(loss="poisson", learning_rate=0.05, max_iter=400, max_leaf_nodes=31, min_samples_leaf=80,
                            l2_regularization=1.0)
    D = D[D["n_prev"] >= min_prev]
    f = feats(stat)
    months = pd.date_range(pd.Timestamp(start, tz="UTC"), D["tip"].max() + pd.offsets.MonthBegin(1), freq="MS")
    out = []
    for m0, m1 in zip(months[:-1], months[1:]):
        tr = D[D["tip"] < m0]; te = D[(D["tip"] >= m0) & (D["tip"] < m1)]
        if te.empty:
            continue
        mdl = HistGradientBoostingRegressor(**params).fit(tr[f], tr[stat])
        p = te[["gid", "pid", "tip", stat]].copy(); p["mu"] = np.clip(mdl.predict(te[f]), 0.05, None)
        out.append(p)
    return pd.concat(out)


def fit_r(mu, y):
    """NB size r maximising likelihood (Var = mu + mu^2 / r)."""
    best = None
    for r in np.exp(np.linspace(np.log(0.5), np.log(500), 120)):
        ll = nbinom.logpmf(y, r, r / (r + mu)).sum()
        if best is None or ll > best[0]:
            best = (ll, r)
    return best[1]


def p_over(mu, k, r):
    """P(X >= k) for NB(mean mu, size r)."""
    return nbinom.sf(np.ceil(k) - 1, r, r / (r + mu))
