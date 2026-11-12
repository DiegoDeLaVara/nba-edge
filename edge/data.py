"""Turn ESPN game rows into flat game and player-game tables (pandas)."""
import gzip, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

STAT = {"minutes": "min", "points": "pts", "rebounds": "reb", "assists": "ast", "threePointFieldGoalsMade": "fg3m",
        "threePointFieldGoalsAttempted": "fg3a", "fieldGoalsMade": "fgm", "fieldGoalsAttempted": "fga",
        "freeThrowsMade": "ftm", "freeThrowsAttempted": "fta", "turnovers": "tov", "steals": "stl", "blocks": "blk",
        "offensiveRebounds": "oreb", "defensiveRebounds": "dreb", "fouls": "pf", "plusMinus": "pm"}


def american_to_prob(a):
    if a is None or (isinstance(a, float) and np.isnan(a)):
        return np.nan
    a = float(a)
    return 100 / (a + 100) if a > 0 else -a / (-a + 100)


def _f(x):
    try:
        if isinstance(x, str):
            x = x.replace("o", "").replace("u", "").replace("+", "")
            if x.upper() in ("EVEN", "PK"):
                return 100.0 if x.upper() == "EVEN" else 0.0
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def lines(pick):
    """DraftKings (or first provider) close + open lines from ESPN pickcenter. Spread is the HOME line."""
    p = next((x for x in pick if (x.get("provider") or "").lower().startswith("draft")), pick[0] if pick else None)
    if not p:
        return {}
    h, a = p.get("homeTeamOdds") or {}, p.get("awayTeamOdds") or {}
    out = {"book": p.get("provider"), "spread": _f(p.get("spread")), "total": _f(p.get("overUnder")),
           "ml_h": _f(h.get("moneyLine")), "ml_a": _f(a.get("moneyLine")), "sp_odds_h": _f(h.get("spreadOdds")),
           "sp_odds_a": _f(a.get("spreadOdds")), "o_odds": _f(p.get("overOdds")), "u_odds": _f(p.get("underOdds"))}
    ps, tt, ml = p.get("pointSpread") or {}, p.get("total") or {}, p.get("moneyline") or {}
    try:
        out["spread_open"] = _f(((ps.get("home") or {}).get("open") or {}).get("line"))
        out["total_open"] = _f(((tt.get("over") or {}).get("open") or {}).get("line"))
        out["ml_h_open"] = _f(((ml.get("home") or {}).get("open") or {}).get("odds"))
        out["ml_a_open"] = _f(((ml.get("away") or {}).get("open") or {}).get("odds"))
    except AttributeError:
        pass
    # ESPN sometimes reports the spread from the favourite's side; fix sign from the details string ("ATL -2.5")
    det = p.get("details") or ""
    return out


def load(path=None):
    path = Path(path or ROOT / "work" / "espn_games.jsonl.gz")
    op = gzip.open if str(path).endswith(".gz") else open
    G, P = [], []
    with op(path, "rt") as fh:
        for line in fh:
            g = json.loads(line)
            if g.get("status") != "STATUS_FINAL" or not g.get("home") or not g.get("away"):
                continue
            if g.get("stype") not in (2, 3, 5):     # regular, playoffs, play-in
                continue
            h, a = g["home"]["abbr"], g["away"]["abbr"]
            row = {"gid": g["id"], "tip": pd.Timestamp(g["date"]), "season": g["season"], "stype": g["stype"],
                   "neutral": bool(g.get("neutral")), "home": h, "away": a, "hs": g["home"]["score"], "as": g["away"]["score"],
                   "ot": max(0, len(g["home"].get("ls") or []) - 4), "notes": "|".join(g.get("notes") or [])}
            row.update(lines(g.get("pick") or []))
            tpos = {}
            for team, pl in (g.get("players") or {}).items():
                opp = a if team == h else h
                for r in pl:
                    d = {"gid": g["id"], "team": team, "opp": opp, "home": team == h, "pid": r.get("pid"), "name": r.get("name"),
                         "pos": r.get("pos"), "starter": r.get("starter"), "dnp": r.get("dnp"), "reason": r.get("reason"),
                         "ejected": r.get("ejected")}
                    for k, v in STAT.items():
                        d[v] = r.get(k)
                    P.append(d)
            G.append(row)
    G = pd.DataFrame(G).sort_values(["tip", "gid"], kind="stable").reset_index(drop=True)
    P = pd.DataFrame(P)
    P["min"] = P["min"].fillna(0.0)
    P["played"] = (~P["dnp"].astype(bool)) & (P["min"] > 0)
    P = P.merge(G[["gid", "tip", "season", "stype"]], on="gid")
    # sanity: spread sign. ESPN 'spread' is the home line in pickcenter; check vs ML favourite
    fav_h = G["ml_h"] < G["ml_a"]
    bad = (fav_h & (G["spread"] > 0)) | (~fav_h & (G["spread"] < 0) & G["ml_h"].notna())
    G["spread_sign_odd"] = bad
    return G, P


def team_games(G, P):
    """One row per team-game with possessions and points for/against."""
    tp = P[P["played"]].groupby(["gid", "team"])[["fga", "fta", "oreb", "tov", "pts", "fg3a", "fg3m", "reb", "ast", "min"]].sum().reset_index()
    tp["poss"] = tp["fga"] - tp["oreb"] + tp["tov"] + 0.44 * tp["fta"]
    out = []
    Gi = G.set_index("gid")
    for (gid, team), r in tp.set_index(["gid", "team"]).iterrows():
        gg = Gi.loc[gid]
        home = team == gg["home"]
        opp = gg["away"] if home else gg["home"]
        out.append({"gid": gid, "tip": gg["tip"], "season": gg["season"], "team": team, "opp": opp, "home": home,
                    "pf": gg["hs"] if home else gg["as"], "pa": gg["as"] if home else gg["hs"], "poss": r["poss"],
                    "fg3a": r["fg3a"], "reb": r["reb"], "ast": r["ast"], "ot": gg["ot"]})
    T = pd.DataFrame(out)
    o = T[["gid", "team", "poss"]].rename(columns={"team": "opp", "poss": "opp_poss"})
    T = T.merge(o, on=["gid", "opp"], how="left")
    T["pace"] = (T["poss"] + T["opp_poss"]) / 2 * 48 / (48 + 5 * T["ot"])
    return T.sort_values(["tip", "gid", "team"], kind="stable").reset_index(drop=True)
