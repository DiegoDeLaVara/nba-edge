"""Match Polymarket NBA markets to the Kalshi markets we have pre-tip quotes for.
Props: PM "Player: Points O/U 16.5" (Yes = over) <-> Kalshi "Player: 17+". Moneyline: PM outcome team <-> KXNBAGAME team.
Totals: PM "O/U 229.5" (outcome Over) <-> KXNBATOTAL floor 229.5. -> work/pm_matched.pkl"""
import sys, json; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from edge.kalshi import norm_name, load_markets, load_candles, parse_event, game_index, quote_at

G = pd.read_pickle("work/G.pkl")
PM = json.load(open("work/pm_events.json"))
STAT = {"points": "pts", "rebounds": "reb", "assists": "ast", "threes": "fg3m"}
NICK = {}   # team nickname -> ESPN abbr, learnt from moneyline outcomes + slug order
SLUG = {"gsw": "GS", "sas": "SA", "nop": "NO", "nyk": "NY", "uta": "UTAH", "was": "WSH", "wsh": "WSH", "pho": "PHX", "phx": "PHX",
        "bkn": "BKN", "brk": "BKN", "cha": "CHA", "cho": "CHA", "noh": "NO", "gs": "GS", "sa": "SA", "no": "NO", "ny": "NY", "utah": "UTAH"}

# ESPN games by (away, home, ET date)
Gk = {}
for r in G.itertuples(index=False):
    d = r.tip.tz_convert("America/New_York").date()
    Gk[(r.away, r.home, d)] = (r.gid, r.tip)


def gid_of(slug):
    _, a, h, y, m, d = slug.split("-")
    a = SLUG.get(a, a.upper()); h = SLUG.get(h, h.upper())
    day = pd.Timestamp(f"{y}-{m}-{d}").date()
    for dd in (day, day - pd.Timedelta(days=1), day + pd.Timedelta(days=1)):
        for key in ((a, h, dd), (h, a, dd)):
            if key in Gk:
                return Gk[key][0], key[0], key[1]
    return None, a, h


rows = []
miss = 0
for e in PM:
    gid, away, home = gid_of(e["slug"])
    if gid is None:
        miss += 1; continue
    for m in e["markets"]:
        t = m.get("sportsMarketType"); toks = json.loads(m["clobTokenIds"]) if m.get("clobTokenIds") else None
        outs = json.loads(m["outcomes"]) if m.get("outcomes") else None
        if not toks or not outs:
            continue
        r = {"gid": gid, "type": t, "q": m["question"], "line": m.get("line"), "token": toks[0], "out0": outs[0],
             "res": json.loads(m["outcomePrices"])[0] if m.get("outcomePrices") else None, "pm_vol": float(m.get("volume") or 0)}
        if t in STAT:
            name = m["question"].split(":")[0]
            r.update(stat=STAT[t], nn=norm_name(name), k=float(m["line"]) + 0.5)
        elif t == "moneyline":
            r.update(team_nick=outs[0])
        elif t != "totals":
            continue
        rows.append(r)
P = pd.DataFrame(rows)
print("PM events not matched to ESPN games:", miss, "| PM markets kept:", len(P), P.type.value_counts().to_dict())

# ---- Kalshi side: props
out = []
pl = pd.read_pickle("work/P.pkl"); pl = pl[pl.played]; pl["nn"] = pl.name.map(norm_name)
for st in ("pts", "reb", "ast", "fg3m"):
    K = pd.read_pickle(f"work/pt_{st}.pkl")[["ticker", "gid", "pid", "name", "k", "bid", "ask", "y", "tip", "p", "qts"]]
    K["nn"] = K.name.map(norm_name); K["stat"] = st
    M = K.merge(P[P.stat == st][["gid", "nn", "k", "token", "res", "pm_vol", "q"]], on=["gid", "nn", "k"])
    out.append(M)
PR = pd.concat(out, ignore_index=True)
PR["pm_res_ok"] = (PR.res.astype(float).round() == PR.y)
print("props matched:", len(PR), PR.groupby("stat").size().to_dict(), "| PM 'Yes'=over check:", PR.pm_res_ok.mean().round(4))
# ---- moneyline
K = pd.read_pickle("work/kalshi_ml.pkl")[["gid", "team", "ticker", "y", "bid", "ask", "tip", "home", "away"]]
ml = P[P.type == "moneyline"][["gid", "token", "team_nick", "res", "pm_vol", "q"]]
# PM outcome 0 is the first team in "A vs. B"; slug order is away-home and the question lists the away team first
ML = []
for r in ml.itertuples(index=False):
    k = K[K.gid == r.gid]
    if len(k) != 2:
        continue
    away = k.iloc[0].away
    kk = k[k.team == away]
    if len(kk) == 1:
        x = kk.iloc[0].to_dict(); x.update(token=r.token, res=r.res, pm_vol=r.pm_vol, q=r.q, stat="ml", k=np.nan, tip=int(pd.Timestamp(x["tip"]).timestamp())); ML.append(x)
ML = pd.DataFrame(ML)
ML["pm_res_ok"] = ML.res.astype(float).round() == ML.y
print("moneylines matched:", len(ML), "| PM outcome0 = away team check:", ML.pm_res_ok.mean().round(4))
# ---- totals (Kalshi quotes from candles)
C = load_candles("KXNBATOTAL"); gi = game_index(G); tip = dict(zip(G.gid, G.tip))
KT = []
for m in load_markets("KXNBATOTAL"):
    if m.get("result") not in ("yes", "no") or m["ticker"] not in C:
        continue
    pe = parse_event(m["event_ticker"]); gid = gi.get(pe) if pe else None
    if gid is None:
        continue
    q = quote_at(C[m["ticker"]], int(tip[gid].timestamp()))
    if q:
        KT.append({"gid": gid, "ticker": m["ticker"], "line": float(m["floor_strike"]), "bid": q[0], "ask": q[1], "y": int(m["result"] == "yes"), "tip": int(tip[gid].timestamp())})
KT = pd.DataFrame(KT)
TT = KT.merge(P[P.type == "totals"][["gid", "line", "token", "res", "pm_vol", "q"]], on=["gid", "line"])
TT["stat"] = "total"; TT["pm_res_ok"] = TT.res.astype(float).round() == TT.y
print("totals matched:", len(TT), "| PM Over check:", TT.pm_res_ok.mean().round(4))
ALL = pd.concat([PR, ML, TT], ignore_index=True)
ALL["tip"] = ALL["tip"].astype(int)
# Kalshi quote time: props carry it; game lines use the hourly candle at or before tip
ALL["qts"] = ALL["qts"].fillna(ALL["tip"] - ALL["tip"] % 3600).astype(int)
ALL.to_pickle("work/pm_matched.pkl")
print("total matched markets:", len(ALL), "games", ALL.gid.nunique())
