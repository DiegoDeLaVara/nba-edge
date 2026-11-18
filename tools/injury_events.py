"""Status-change events from the official injury reports (work/ir_rows.jsonl):
  OUT   - first report listing a player Out after an earlier report listed him Questionable/Doubtful/Probable,
          or first listing as Out after the team had already submitted reports without him (late scratch)
  PLAYS - first report listing him Available after Questionable/Doubtful
Joined to games (gid) and the player's role (season-to-date minutes/points before that game).
-> work/ir_events.pkl"""
import sys, json; sys.path.insert(0, ".")
import pandas as pd, numpy as np
from edge.kalshi import norm_name, K2E

E2K = {v: k for k, v in K2E.items()}
R = pd.DataFrame([json.loads(l) for l in open("work/ir_rows.jsonl")])
R["ts"] = pd.to_datetime(R.ts)
R = R[R.matchup.str.contains("@", na=False) & R.date.notna()].copy()
R["gdate"] = pd.to_datetime(R.date, format="%m/%d/%Y", errors="coerce")
R = R[R.gdate.notna()]
R["name"] = R.player.map(lambda s: " ".join(reversed([x.strip() for x in s.split(",", 1)])) if "," in s else s)
R["nk"] = R.name.map(norm_name)
R = R.sort_values("ts")

G = pd.read_pickle("work/G.pkl"); P = pd.read_pickle("work/P.pkl")
G = G[G.season == 2027].copy()
G["gdate"] = (G.tip - pd.Timedelta(hours=10)).dt.tz_localize(None).dt.normalize()
# the PDF's merged team/matchup cells mislabel rows, so take each player's team from box scores (latest team before
# the game date) and find the game by (date, team)
P = P[P.season == 2027].sort_values("tip").copy()
P["nk"] = P.name.map(norm_name)
P["min_avg"] = P.groupby("pid")["min"].transform(lambda s: s.where(P.loc[s.index, "played"]).shift().expanding().mean())
P["pts_avg"] = P.groupby("pid")["pts"].transform(lambda s: s.where(P.loc[s.index, "played"]).shift().expanding().mean())
P["gdate"] = (P.tip - pd.Timedelta(hours=10)).dt.tz_localize(None).dt.normalize()
R = R.drop(columns=["team"]).sort_values("gdate")
R = pd.merge_asof(R, P[["gdate", "nk", "team", "pid"]].sort_values("gdate"), on="gdate", by="nk",
                  allow_exact_matches=False).dropna(subset=["team"])
R = R.sort_values("ts")
gk = {}
for g in G.itertuples():
    gk[(g.gdate, g.home)] = g; gk[(g.gdate, g.away)] = g
role = {(r.gid, r.nk): r for r in P.itertuples()}
last_role = P[P.played].groupby("nk")[["min_avg", "pts_avg"]].last()
team_first = R.groupby(["gdate", "team"]).ts.min()

ev = []
for (gd, tm, nk), d in R.groupby(["gdate", "team", "nk"], sort=False):
    g = gk.get((gd, tm))
    if g is None:
        continue
    st = d.status.tolist(); ts = d.ts.tolist()
    kind = t = prev = None
    for i, s in enumerate(st):
        if s == "Out" and i > 0 and st[i - 1] in ("Questionable", "Doubtful", "Probable"):
            kind, t, prev = "OUT", ts[i], st[i - 1]; break
        if s == "Available" and i > 0 and st[i - 1] in ("Questionable", "Doubtful"):
            kind, t, prev = "PLAYS", ts[i], st[i - 1]; break
    if kind is None and st[0] == "Out":
        tf = team_first.get((gd, tm))
        if tf is not None and ts[0] - tf >= pd.Timedelta(minutes=45):
            kind, t, prev = "OUT", ts[0], "new"
    if kind is None:
        continue
    rr = role.get((g.gid, nk))
    mn = rr.min_avg if rr is not None else (last_role.min_avg.get(nk) if nk in last_role.index else np.nan)
    pa = rr.pts_avg if rr is not None else (last_role.pts_avg.get(nk) if nk in last_role.index else np.nan)
    ev.append({"gid": g.gid, "tip": g.tip, "ts": t, "mins_to_tip": (g.tip - t).total_seconds() / 60, "kind": kind,
               "prev": prev, "player": d.name.iloc[0], "nk": nk, "team": tm, "opp": g.home if tm == g.away else g.away,
               "gcode": gd.strftime("%y%b%d").upper() + E2K.get(g.away, g.away) + E2K.get(g.home, g.home), "min_avg": mn, "pts_avg": pa,
               "reason": d.reason.iloc[-1]})
E = pd.DataFrame(ev)
E = E[E.mins_to_tip > -5]
E.to_pickle("work/ir_events.pkl")
print(len(E), E.kind.value_counts().to_dict())
print("key players (>=24 min):", E[E.min_avg >= 24].kind.value_counts().to_dict())
print(E[E.min_avg >= 24].mins_to_tip.describe().round(0).to_dict())
print(E[E.min_avg >= 24].sort_values("pts_avg", ascending=False).head(15)[["ts", "mins_to_tip", "kind", "prev", "player", "team", "opp", "pts_avg"]].to_string())
