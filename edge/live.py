"""Projections for upcoming games: append 'pending' rows for today's expected players to the history, run the same
leak-free feature code as the research, and predict with the trained models."""
import numpy as np
import pandas as pd

from edge.features import TEAMS, player_features, team_absence, team_context
from edge.data import team_games
from edge.gamemodel import team_rows, design
from edge.props import build, feats, STATS

OUT_STATUS = {"out", "doubtful", "suspension", "suspended", "inactive"}


def pending_rows(G, P, games, inj, recent_days=21):
    """Expected participants of upcoming games: rotation players who played for the team in the last few weeks,
    minus players listed Out/Doubtful (those become known absences)."""
    rows, grow = [], []
    Q = P[P["played"]]
    for g in games:
        if g["home"] not in TEAMS or g["away"] not in TEAMS:
            continue
        tip = pd.Timestamp(g["tip"])
        grow.append({"gid": g["gid"], "tip": tip, "season": int(G["season"].max()), "stype": 2 if (g.get("season_type") or 2) == 2 else g["season_type"],
                     "neutral": False, "home": g["home"], "away": g["away"], "hs": np.nan, "as": np.nan, "ot": 0})
        for team, opp, home in ((g["home"], g["away"], True), (g["away"], g["home"], False)):
            recent = Q[(Q["team"] == team) & (Q["tip"] >= tip - pd.Timedelta(days=recent_days)) & (Q["tip"] < tip)]
            if recent.empty:   # season opener: last season's roster isn't reliable; take the last 6 team games
                recent = Q[(Q["team"] == team) & (Q["tip"] < tip)].sort_values("tip").groupby("pid").tail(1).tail(15)
            last = recent.sort_values("tip").groupby("pid").tail(1)
            ti = inj.get(team, {})
            for r in last.itertuples(index=False):
                st = (ti.get(r.name) or {}).get("status", "")
                out = st.lower() in OUT_STATUS
                rows.append({"gid": g["gid"], "team": team, "opp": opp, "home": home, "pid": r.pid, "name": r.name, "pos": r.pos,
                             "starter": False, "dnp": out, "reason": f"INJURY {st}" if out else "", "ejected": False,
                             "min": 0.0 if out else np.nan, "tip": tip, "season": int(G["season"].max()), "stype": 2,
                             "played": not out, "inj": st})
    return pd.DataFrame(grow), pd.DataFrame(rows)


def project(G, P, games, inj, models, days=420):
    """Return per-player projections (mu per stat) for upcoming games plus team point predictions."""
    t0 = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
    Gh = G[G["tip"] >= t0]; Ph = P[P["tip"] >= t0]
    gp, pp = pending_rows(Gh, Ph, games, inj)
    if gp.empty or pp.empty:
        return pd.DataFrame(), pd.DataFrame()
    pend_g = set(gp["gid"])
    G2 = pd.concat([Gh, gp], ignore_index=True).sort_values(["tip", "gid"], kind="stable").reset_index(drop=True)
    P2 = pd.concat([Ph, pp.drop(columns=["inj"])], ignore_index=True)
    for c in ("pts", "reb", "ast", "fg3m", "fg3a", "fga", "fta", "tov", "oreb", "stl", "blk", "fgm", "ftm", "dreb", "pf", "pm"):
        if c not in P2.columns:
            P2[c] = np.nan
    P2["min"] = P2["min"].astype(float)
    Pf, Q = player_features(P2)
    A = team_absence(Pf, Q)
    T = team_games(G2[~G2["gid"].isin(pend_g)], Pf[~Pf["gid"].isin(pend_g)])
    # pending team rows so rest / pace / defence carry forward
    tp = []
    for g in gp.itertuples(index=False):
        for team, opp, home in ((g.home, g.away, True), (g.away, g.home, False)):
            tp.append({"gid": g.gid, "tip": g.tip, "season": g.season, "team": team, "opp": opp, "home": home})
    T = pd.concat([T, pd.DataFrame(tp)], ignore_index=True)
    T2 = team_context(G2, T)
    R = team_rows(G2, A)
    gm = models["game"]
    te = R[R["gid"].isin(pend_g)].copy()
    te["xpts"] = gm.predict(design(te))
    h = te[te["home"]][["gid", "xpts"]].rename(columns={"xpts": "xh"}); a = te[~te["home"]][["gid", "xpts"]].rename(columns={"xpts": "xa"})
    X = gp.merge(h, on="gid").merge(a, on="gid")
    X["xmargin"] = X["xh"] - X["xa"]; X["xtotal"] = X["xh"] + X["xa"]
    D = build(Q[Q["gid"].isin(pend_g)], A, T2, X)
    for st in STATS:
        D[f"mu_{st}"] = np.clip(models[st].predict(D[feats(st)]), 0.05, None) if len(D) else []
    inj_map = dict(zip(zip(pp["gid"], pp["pid"]), pp["inj"]))
    D["inj"] = [inj_map.get((g, p), "") for g, p in zip(D["gid"], D["pid"])]
    return D, X
