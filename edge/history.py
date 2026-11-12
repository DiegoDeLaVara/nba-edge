"""Compact game / player-game history (state/hist_games.csv.gz, state/hist_players.csv.gz) and daily ESPN updates."""
import gzip, io, json, time, datetime as dt
from pathlib import Path
import numpy as np
import pandas as pd
import requests

from edge.data import STAT, lines

SITE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
S = requests.Session(); S.headers.update({"User-Agent": "nba-edge/1.0 (github actions)"})
GCOLS = ["gid", "tip", "season", "stype", "neutral", "home", "away", "hs", "as", "ot", "spread", "total", "ml_h", "ml_a"]
PCOLS = ["gid", "team", "opp", "home", "pid", "name", "pos", "starter", "dnp", "reason", "ejected"] + list(STAT.values())


def get(url, params=None, tries=4):
    for i in range(tries):
        try:
            r = S.get(url, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
        except requests.RequestException:
            pass
        time.sleep(2 + 2 * i)
    return None


def scoreboard(day):
    d = get(f"{SITE}/scoreboard", {"dates": day.strftime("%Y%m%d"), "limit": 50}) or {}
    out = []
    for e in d.get("events") or []:
        c = (e.get("competitions") or [{}])[0]
        t = {x.get("homeAway"): (x.get("team") or {}).get("abbreviation") for x in c.get("competitors") or []}
        st = (e.get("status") or {}).get("type") or {}
        out.append({"gid": e["id"], "tip": e.get("date"), "home": t.get("home"), "away": t.get("away"),
                    "state": st.get("state"), "completed": bool(st.get("completed")), "name": e.get("shortName"),
                    "season_type": (e.get("season") or {}).get("type")})
    return out


def injuries():
    """{team_abbr: {player name: status}} from ESPN's league injury feed."""
    d = get(f"{SITE}/injuries") or {}
    out = {}
    for t in d.get("injuries") or []:
        for i in t.get("injuries") or []:
            a = i.get("athlete") or {}
            team = ((a.get("team") or {}).get("abbreviation")) or None
            out.setdefault(team or t.get("displayName"), {})[a.get("displayName")] = {
                "status": i.get("status"), "short": i.get("shortComment"), "date": i.get("date"),
                "team_name": t.get("displayName")}
    return out


def summary_rows(eid):
    from tools.espn_collect import game  # same parser as the research download
    g = game(eid)
    if not g or g.get("status") != "STATUS_FINAL":
        return None, None
    h, a = g["home"]["abbr"], g["away"]["abbr"]
    gr = {"gid": g["id"], "tip": g["date"], "season": g["season"], "stype": g["stype"], "neutral": bool(g.get("neutral")),
          "home": h, "away": a, "hs": g["home"]["score"], "as": g["away"]["score"],
          "ot": max(0, len(g["home"].get("ls") or []) - 4)}
    gr.update({k: v for k, v in lines(g.get("pick") or []).items() if k in GCOLS})
    pr = []
    for team, pl in (g.get("players") or {}).items():
        for r in pl:
            d = {"gid": g["id"], "team": team, "opp": a if team == h else h, "home": team == h, "pid": r.get("pid"),
                 "name": r.get("name"), "pos": r.get("pos"), "starter": r.get("starter"), "dnp": r.get("dnp"),
                 "reason": r.get("reason"), "ejected": r.get("ejected")}
            for k, v in STAT.items():
                d[v] = r.get(k)
            pr.append(d)
    return gr, pr


def load(state):
    G = pd.read_csv(state / "hist_games.csv.gz", dtype={"gid": str})
    P = pd.read_csv(state / "hist_players.csv.gz", dtype={"gid": str, "pid": str})
    G["tip"] = pd.to_datetime(G["tip"], utc=True)
    return G, P


def save(state, G, P):
    G = G.copy(); G["tip"] = G["tip"].dt.strftime("%Y-%m-%dT%H:%MZ")
    G[GCOLS].to_csv(state / "hist_games.csv.gz", index=False, compression="gzip")
    P[PCOLS].to_csv(state / "hist_players.csv.gz", index=False, compression="gzip")


def update(state, days_back=4, log=print):
    """Add finished games from the last few days that are not in the history yet."""
    G, P = load(state)
    have = set(G["gid"])
    newg, newp = [], []
    today = dt.datetime.now(dt.timezone.utc).date()
    for i in range(days_back, -1, -1):
        for e in scoreboard(today - dt.timedelta(days=i)):
            if e["completed"] and e["gid"] not in have:
                gr, pr = summary_rows(e["gid"])
                if gr:
                    newg.append(gr); newp += pr; have.add(e["gid"])
    if newg:
        g2 = pd.DataFrame(newg); g2["tip"] = pd.to_datetime(g2["tip"], utc=True)
        G = pd.concat([G, g2], ignore_index=True).sort_values(["tip", "gid"], kind="stable").reset_index(drop=True)
        P = pd.concat([P, pd.DataFrame(newp)], ignore_index=True)
        save(state, G, P)
    log(f"history: +{len(newg)} games, {len(G)} total")
    return G, P, len(newg)


def frames(G, P):
    """Same shape as edge.data.load(): played flag, tip/season/stype on player rows."""
    P = P.copy()
    for c in ("starter", "dnp", "ejected", "home"):
        P[c] = P[c].astype(str).str.lower().isin(["true", "1"])
    P["reason"] = P["reason"].fillna("")
    P["min"] = pd.to_numeric(P["min"], errors="coerce").fillna(0.0)
    P["played"] = (~P["dnp"]) & (P["min"] > 0)
    P = P.drop(columns=[c for c in ("tip", "season", "stype") if c in P.columns]).merge(G[["gid", "tip", "season", "stype"]], on="gid")
    G = G.copy(); G["neutral"] = G["neutral"].astype(str).str.lower().isin(["true", "1"])
    return G, P
