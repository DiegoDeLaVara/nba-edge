"""Leak-free pre-game features. Every per-player / per-team number uses only games that tipped before this one.

Known-before-tip assumption: players listed out with an injury/illness/personal reason (ESPN DNP reason that is not
"COACH'S DECISION") are treated as known absences (injury reports); coach's-decision DNPs are not used as inputs.
"""
import re
import numpy as np
import pandas as pd

TEAMS = {"ATL", "BKN", "BOS", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GS", "HOU", "IND", "LAC", "LAL", "MEM", "MIA",
         "MIL", "MIN", "NO", "NY", "OKC", "ORL", "PHI", "PHX", "POR", "SA", "SAC", "TOR", "UTAH", "WSH"}
RATE = ["pts", "reb", "ast", "fg3m", "fg3a", "fga", "fta", "tov", "oreb", "stl", "blk"]


def known_out(reason):
    r = (reason or "").upper()
    return bool(r) and "COACH" not in r


def ewm_prev(s, hl):
    """EWMA of previous values within a group (shifted: excludes the current row)."""
    return s.shift(1).ewm(halflife=hl, min_periods=1).mean()


def player_features(P):
    P = P[P["team"].isin(TEAMS)].copy()
    P = P.sort_values(["tip", "gid", "team", "pid"], kind="stable").reset_index(drop=True)
    P["out_known"] = (~P["played"]) & P["reason"].map(known_out)
    Q = P[P["played"]].copy()
    g = Q.groupby("pid", sort=False)
    Q["n_prev"] = g.cumcount()
    Q["n_season"] = Q.groupby(["pid", "season"], sort=False).cumcount()
    for hl in (3, 10):
        Q[f"min_e{hl}"] = g["min"].transform(lambda s: ewm_prev(s, hl))
    Q["min_last"] = g["min"].shift(1)
    Q["start_e"] = g["starter"].transform(lambda s: ewm_prev(s.astype(float), 3))
    Q["start_last"] = g["starter"].shift(1).astype(float)
    Q["prev_tip"] = g["tip"].shift(1)
    Q["days_off"] = (Q["tip"] - Q["prev_tip"]).dt.total_seconds() / 86400
    for st in RATE:
        num = g[st].transform(lambda s: ewm_prev(s, 10))
        den = Q["min_e10"]
        Q[f"{st}_pm"] = num / den.replace(0, np.nan)
        Q[f"{st}_e5"] = g[st].transform(lambda s: ewm_prev(s, 5))
    # season-to-date per-minute (slower, steadier)
    cs = Q.groupby(["pid", "season"], sort=False)
    cm = cs["min"].cumsum() - Q["min"]
    for st in ("pts", "reb", "ast", "fg3m"):
        Q[f"{st}_spm"] = (cs[st].cumsum() - Q[st]) / cm.replace(0, np.nan)
    # attach back the per-player state for absent players: last known min_e10 / pts_pm before this game
    return P, Q


def team_absence(P, Q):
    """Per team-game: minutes and scoring of rotation players known out (injury list) going into the game."""
    st = Q[["pid", "tip", "min_e10", "pts_pm", "ast_pm", "reb_pm", "min"]].copy()
    st["val_min"] = st["min_e10"]
    # state AFTER each played game: ewm including that game, approximated by blending
    st["after_min"] = 0.75 * st["min_e10"].fillna(st["min"]) + 0.25 * st["min"]
    st["team"] = Q["team"]
    st = st.sort_values("tip", kind="stable")
    # ESPN box scores often omit inactive (injured) players, so an absence = a rotation player who last played for
    # this team within 21 days and is not playing now, unless his next game is for another team (traded) or he got a
    # coach's-decision DNP (not known before tip).
    st["next_team"] = st.groupby("pid")["team"].shift(-1)
    tg = P[["gid", "tip", "team"]].drop_duplicates(["gid", "team"])
    cand, roster = [], []
    played = set(zip(Q["gid"], Q["pid"]))
    coach = set(zip(P.loc[(~P["played"]) & ~P["out_known"], "gid"], P.loc[(~P["played"]) & ~P["out_known"], "pid"]))
    for team, tgt in tg.groupby("team"):
        s = st[st["team"] == team]
        if s.empty:
            continue
        recent = s[["tip", "pid", "after_min", "pts_pm", "ast_pm", "reb_pm", "next_team"]]
        for r in tgt.itertuples(index=False):
            w = recent[(recent["tip"] < r.tip) & (recent["tip"] >= r.tip - pd.Timedelta(days=21))]
            if w.empty:
                continue
            last = w.groupby("pid").tail(1)
            last = last[(last["after_min"] >= 12)]
            for x in last.itertuples(index=False):
                if isinstance(x.next_team, str) and x.next_team != team and (r.gid, x.pid) not in played:
                    continue
                roster.append((r.gid, team, x.after_min, x.after_min * (x.pts_pm if x.pts_pm == x.pts_pm else 0)))
                if (r.gid, x.pid) in played or (r.gid, x.pid) in coach:
                    continue
                cand.append((r.gid, team, x.pid, x.after_min, x.pts_pm, x.ast_pm, x.reb_pm))
    m = pd.DataFrame(cand, columns=["gid", "team", "pid", "after_min", "pts_pm", "ast_pm", "reb_pm"])
    m["miss_pts"] = m["after_min"] * m["pts_pm"].fillna(0)
    m["miss_ast"] = m["after_min"] * m["ast_pm"].fillna(0)
    m["miss_reb"] = m["after_min"] * m["reb_pm"].fillna(0)
    a = m.groupby(["gid", "team"]).agg(miss_min=("after_min", "sum"), miss_pts=("miss_pts", "sum"),
                                       miss_ast=("miss_ast", "sum"), miss_reb=("miss_reb", "sum"),
                                       miss_n=("pid", "size")).reset_index()
    ro = pd.DataFrame(roster, columns=["gid", "team", "rmin", "rpts"]).groupby(["gid", "team"]).agg(
        roster_min=("rmin", "sum"), roster_pts=("rpts", "sum")).reset_index()
    return ro.merge(a, on=["gid", "team"], how="left").fillna({"miss_min": 0, "miss_pts": 0, "miss_ast": 0, "miss_reb": 0, "miss_n": 0})


def team_context(G, T):
    """Rest days, back-to-backs, and decayed team pace / defensive rates, all from previous games."""
    T = T[T["team"].isin(TEAMS)].sort_values(["tip", "gid"], kind="stable").copy()
    g = T.groupby("team", sort=False)
    T["rest"] = (T["tip"] - g["tip"].shift(1)).dt.total_seconds() / 86400
    T["pace_e"] = g["pace"].transform(lambda s: ewm_prev(s, 15))
    T["ppp"] = T["pf"] / T["poss"]
    T["oppp"] = T["pa"] / T["opp_poss"]
    T["ortg_e"] = g["ppp"].transform(lambda s: ewm_prev(s, 15))
    T["drtg_e"] = g["oppp"].transform(lambda s: ewm_prev(s, 15))
    # allowed rates: what opponents did against this team
    o = T[["gid", "team", "reb", "ast", "fg3a", "poss"]].rename(columns={"team": "opp", "reb": "a_reb", "ast": "a_ast", "fg3a": "a_fg3a", "poss": "a_poss"})
    T = T.merge(o, on=["gid", "opp"], how="left").sort_values(["tip", "gid"], kind="stable")
    g = T.groupby("team", sort=False)
    for c in ("a_reb", "a_ast", "a_fg3a"):
        T[c + "_e"] = g[c].transform(lambda s: ewm_prev(s, 15)) / g["a_poss"].transform(lambda s: ewm_prev(s, 15))
    return T
