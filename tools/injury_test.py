"""Does Kalshi reprice slowly after the official NBA injury report? For each status change of a rotation player
(work/ir_events.pkl), follow 1-minute Kalshi prices (work/mc_*.jsonl) of
  - his team's moneyline (OUT -> team less likely to win; PLAYS -> more likely)
  - his teammates' points overs (OUT -> more shots for them; PLAYS -> fewer)
around the report time T, and test buying the expected side at the ask N minutes after T (hold to settlement, fee
0.07*p*(1-p)). If the move happens before T (reporters break the news first), there is nothing left to take."""
import sys, json; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from edge.kalshi import K2E

E2K = {v: k for k, v in K2E.items()}
MINR = float(sys.argv[1]) if len(sys.argv) > 1 else 20      # rotation player: season minutes avg >= this
E = pd.read_pickle("work/ir_events.pkl")
E = E[(E.min_avg >= MINR) & (E.mins_to_tip >= 10)].copy()

C, res = {}, {}
for s in ("KXNBAGAME", "KXNBAPTS"):
    for l in open(f"work/mc_{s}.jsonl"):
        t, c = json.loads(l)
        C[t] = np.array([[e, b if b is not None else np.nan, a if a is not None else np.nan] for e, b, a, v in c], float)
    for l in open(f"work/kmk_{s}.jsonl"):
        m = json.loads(l)
        if m.get("result") in ("yes", "no"):
            res[m["ticker"]] = int(m["result"] == "yes")
by_event = {}
for t in C:
    by_event.setdefault(t.rsplit("-", 1)[0] if t.startswith("KXNBAGAME") else "-".join(t.split("-")[:2]), []).append(t)


def quote(t, ts):
    c = C.get(t)
    if c is None or not len(c):
        return None
    i = np.searchsorted(c[:, 0], ts, side="right") - 1      # last candle ending at or before ts
    while i >= 0 and (np.isnan(c[i, 1]) or np.isnan(c[i, 2]) or not 0 < c[i, 2] - c[i, 1] <= 0.10):
        i -= 1
    return (c[i, 1], c[i, 2]) if i >= 0 else None


OFF = [-60, -30, -15, -5, 0, 2, 5, 10, 15, 30, 60]
rows = []
for e in E.itertuples():
    T = int(e.ts.timestamp()); tip = int(e.tip.timestamp()); sgn = -1 if e.kind == "OUT" else 1
    legs = []
    ml = f"KXNBAGAME-{e.gcode}-{E2K.get(e.team, e.team)}"
    if ml in C:
        legs.append(("ml", ml, sgn))
    tk = E2K.get(e.team, e.team)
    for t in by_event.get(f"KXNBAPTS-{e.gcode}", []):
        who = t.split("-")[2]
        if who.startswith(tk) and e.nk[:5] not in who.lower():
            legs.append(("mate_pts", t, -sgn))
    for kind, t, d in legs:
        q = {o: quote(t, T + 60 * o) for o in OFF if T + 60 * o < tip}
        qe = quote(t, tip - 60)
        if q.get(-60) is None or qe is None or t not in res:
            continue
        mid = lambda x: (x[0] + x[1]) / 2 if x else np.nan
        r = {"ticker": t, "leg": kind, "kind": e.kind, "player": e.player, "pts_avg": e.pts_avg, "mins_to_tip": e.mins_to_tip,
             "dir": d, "y": res[t], "close": mid(qe), "m-60": mid(q[-60])}
        for o in OFF:
            r[f"mv{o}"] = d * (mid(q.get(o)) - r["m-60"])
        for lag in (2, 5, 10, 30):
            x = q.get(lag)
            if x:                                       # buy the expected side at the ask `lag` minutes after T
                p = x[1] if d > 0 else 1 - x[0]
                win = r["y"] if d > 0 else 1 - r["y"]
                fee = 0.07 * p * (1 - p)
                r[f"roi{lag}"] = (win - p - fee) / (p + fee)
                r[f"clv{lag}"] = (r["close"] if d > 0 else 1 - r["close"]) - p
                r[f"p{lag}"] = p
        rows.append(r)
X = pd.DataFrame(rows)
X.to_pickle("work/injury_rows.pkl")
print(f"events (min_avg>={MINR}, >=10 min before tip): {len(E)} -> legs with prices {len(X)} "
      f"({X.groupby('leg').size().to_dict()})")
for (leg, kind), S in X.groupby(["leg", "kind"]):
    mv = S[[f"mv{o}" for o in OFF]].mean()
    print(f"\n{leg} / {kind}: {len(S)} markets, {S.player.nunique()} players")
    print("  move in expected direction vs 60 min before report (cents): " +
          "  ".join(f"{o:+d}m {100 * mv[f'mv{o}']:+.1f}" for o in OFF))
    print(f"  ...and at last pre-tip quote: {100 * (S.dir * (S.close - S['m-60'])).mean():+.1f}c")
    for lag in (2, 5, 10, 30):
        if f"roi{lag}" in S and S[f"roi{lag}"].notna().sum() >= 10:
            s = S[S[f"roi{lag}"].notna()]
            print(f"  buy at ask T+{lag:>2}m: n={len(s):4d} paid {s[f'p{lag}'].mean():.3f} hit {(s.y * (s.dir > 0) + (1 - s.y) * (s.dir < 0)).mean():.3f}"
                  f"  ROI after fee {s[f'roi{lag}'].mean():+.3f}  CLV vs pre-tip mid {100 * s[f'clv{lag}'].mean():+.2f}c")
