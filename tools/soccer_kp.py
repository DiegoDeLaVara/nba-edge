"""Kalshi soccer 1X2 prices vs Pinnacle's closing no-vig line (football-data.co.uk), same games.
Kalshi quote = last hourly candle at or before kickoff. Tests: which is sharper, and does buying on Kalshi when it is
cheaper than Pinnacle's fair price make money after Kalshi's fee?"""
import sys, json, re, difflib; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from edge.kalshi import load_markets, load_candles, quote_at, fee

LG = {"KXEPLGAME": "E0", "KXEFLCHAMPIONSHIPGAME": "E1", "KXEFLL1GAME": "E2", "KXLALIGAGAME": "SP1", "KXLALIGA2GAME": "SP2",
      "KXSERIEAGAME": "I1", "KXBUNDESLIGAGAME": "D1", "KXBUNDESLIGA2GAME": "D2", "KXLIGUE1GAME": "F1",
      "KXEREDIVISIEGAME": "N1", "KXSUPERLIGGAME": "T1", "KXSCOTTISHPREMGAME": "SC0"}
ALIAS = {"man united": "manchester united", "man city": "manchester city", "nott'm forest": "nottingham forest",
         "wolves": "wolverhampton", "spurs": "tottenham", "sheffield weds": "sheffield wednesday", "sheffield united": "sheffield utd",
         "ath madrid": "atletico madrid", "ath bilbao": "athletic bilbao", "betis": "real betis", "sociedad": "real sociedad",
         "m'gladbach": "monchengladbach", "ein frankfurt": "eintracht frankfurt", "fc koln": "koln", "paris sg": "paris saint germain",
         "inter": "inter milan", "milan": "ac milan", "qpr": "queens park rangers", "west brom": "west bromwich"}
MON = {m: i for i, m in enumerate(["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], 1)}


def norm(s):
    s = (s or "").lower().replace("&", "and")
    s = ALIAS.get(s, s)
    s = re.sub(r"\b(fc|afc|cf|sc|ac|as|ss|us|calcio|club|de|cd|ud|rc|sv|vfb|vfl|tsg|1\.)\b", " ", s)
    return re.sub(r"[^a-z]", "", s)


def sim(a, b):
    a, b = norm(a), norm(b)
    if not a or not b:
        return 0
    if a in b or b in a:
        return 1.0
    return difflib.SequenceMatcher(None, a, b).ratio()


F = pd.read_pickle("C:/TENNIS/research/fd/fd_all.pkl")
for c in ("PSCH", "PSCD", "PSCA", "MaxCH", "MaxCD", "MaxCA", "FTHG", "FTAG"):
    F[c] = pd.to_numeric(F[c], errors="coerce")
F = F[F.season.isin(["2425", "2526", "2627"]) & F.FTR.isin(["H", "D", "A"])].copy()
F["ko"] = pd.to_datetime(F.Date + " " + F.Time.fillna("15:00"), dayfirst=True, errors="coerce").dt.tz_localize(
    "Europe/London", ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
F["d"] = F.ko.dt.date

rows, miss = [], 0
for ser, lg in LG.items():
    try:
        ms = load_markets(ser)
    except FileNotFoundError:
        continue
    C = load_candles(ser)
    ev = {}
    for m in ms:
        if m.get("result") in ("yes", "no") and m["ticker"] in C:
            ev.setdefault(m["event_ticker"], []).append(m)
    Fl = F[F.lg == lg]
    for et, mm in ev.items():
        if len(mm) != 3:
            continue
        code = et.split("-")[1]; mt = re.match(r"(\d\d)([A-Z]{3})(\d\d)", code)
        day = pd.Timestamp(2000 + int(mt.group(1)), MON[mt.group(2)], int(mt.group(3))).date()
        teams = [m for m in mm if m["yes_sub_title"] != "Tie"]; tie = [m for m in mm if m["yes_sub_title"] == "Tie"]
        if len(teams) != 2 or len(tie) != 1:
            continue
        cand = Fl[(Fl.d >= day - pd.Timedelta(days=1)) & (Fl.d <= day + pd.Timedelta(days=1))]
        best, bs = None, 0
        for r in cand.itertuples(index=False):
            for h, a in ((teams[0], teams[1]), (teams[1], teams[0])):
                s = min(sim(h["yes_sub_title"], r.HomeTeam), sim(a["yes_sub_title"], r.AwayTeam))
                if s > bs:
                    bs, best = s, (r, h, a)
        if bs < 0.6:
            miss += 1; continue
        r, h, a = best
        ko = int(r.ko.timestamp()) if pd.notna(r.ko) else None
        if ko is None:
            continue
        for side, m, pc, mc, win in (("H", h, "PSCH", "MaxCH", r.FTR == "H"), ("D", tie[0], "PSCD", "MaxCD", r.FTR == "D"),
                                     ("A", a, "PSCA", "MaxCA", r.FTR == "A")):
            q = quote_at(C[m["ticker"]], ko - int(__import__("os").environ.get("OFFSET_H", "0")) * 3600)
            if not q:
                continue
            if (m["result"] == "yes") != bool(win):      # sanity: same match outcome
                continue
            rows.append({"lg": lg, "ko": ko, "game": f"{r.HomeTeam}-{r.AwayTeam}", "side": side, "bid": q[0], "ask": q[1],
                         "y": int(win), "ps": getattr(r, pc), "mx": getattr(r, mc),
                         "psh": r.PSCH, "psd": r.PSCD, "psa": r.PSCA, "match_score": bs})
X = pd.DataFrame(rows).dropna(subset=["psh", "psd", "psa"])
inv = 1 / X[["psh", "psd", "psa"]]
X["fair"] = (1 / X.ps) / inv.sum(axis=1)
X = X[(X.ask - X.bid) <= 0.10].sort_values("ko", kind="stable").reset_index(drop=True)
X["mid"] = (X.bid + X.ask) / 2; X["H1"] = X.ko <= X.ko.quantile(0.5)


def ll(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4); return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


print(f"matched {X.game.nunique()} games ({len(X)} outcomes), unmatched events {miss}; Kalshi spread {np.mean(X.ask - X.bid):.3f}")
print(f"log-loss  Kalshi mid {ll(X.mid, X.y):.4f}   Pinnacle close no-vig {ll(X.fair, X.y):.4f}   mean |gap| {np.mean(np.abs(X.mid - X.fair)):.4f}")
cy = X.ask + fee(X.ask); cn = (1 - X.bid) + fee(1 - X.bid)
for thr in (0.0, 0.02, 0.04):
    for nm, m, cost, win in (("buy YES", X.fair - cy > thr, cy, X.y), ("buy NO", (1 - X.fair) - cn > thr, cn, 1 - X.y)):
        x = ((win - cost) / cost)[m]
        if len(x) < 30:
            continue
        print(f"  Kalshi {nm} when cheaper than Pinnacle by >{thr:.2f}: n={len(x)} ROI {x.mean():+.3f} | H1 {x[X.H1[m]].mean():+.3f} H2 {x[~X.H1[m]].mean():+.3f} | "
              f"by side {pd.Series(x.values, index=X.side[m].values).groupby(level=0).mean().round(3).to_dict()}")
# for reference: best closing price at the books vs Pinnacle on the same games
X["ev_mx"] = X.fair * X.mx - 1
m = X.ev_mx >= 0.02
print(f"  (same games) best book price >= 2% over Pinnacle: n={m.sum()} ROI {np.where(X.y[m] == 1, X.mx[m] - 1, -1).mean():+.3f}")
X.to_pickle("work/soccer_kp.pkl")
