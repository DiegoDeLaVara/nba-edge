"""Historical Kalshi total-points candles, only strikes within 10 points of our pre-game model total (no outcome filter)."""
import sys, json; sys.path.insert(0, ".")
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tools.kalshi_candles import hist_one, PACE
from edge.kalshi import load_markets, parse_event, game_index
X = pd.read_pickle("work/gm_all.pkl"); gi = game_index(X); xt = dict(zip(X.gid, X.xtotal))
out = "work/kc_KXNBATOTAL.jsonl"
done = set()
try:
    done = {json.loads(l)[0] for l in open(out)}
except FileNotFoundError:
    pass
ms = []
for m in load_markets("KXNBATOTAL"):
    if not m.get("hist") or m.get("result") not in ("yes", "no") or m["ticker"] in done: continue
    pe = parse_event(m["event_ticker"]); gid = gi.get(pe) if pe else None
    if gid is None or abs(float(m["floor_strike"]) - xt[gid]) > 10: continue
    ms.append(m)
print("markets", len(ms), flush=True)
PACE.gap = 1 / 10
with open(out, "a") as fh, ThreadPoolExecutor(10) as ex:
    for i, res in enumerate(ex.map(lambda m: hist_one("KXNBATOTAL", m), ms)):
        for t, c in res:
            fh.write(json.dumps([t, c], separators=(",", ":")) + "\n")
        if i % 1000 == 0: print(i, flush=True)
print("done")
