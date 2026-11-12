import json, sys, time, requests
from collections import Counter
API = "https://api.elections.kalshi.com/trade-api/v2"
H = {"User-Agent": "research-client/1.0"}
def events(series):
    out, cur = [], None
    while True:
        p = {"series_ticker": series, "limit": 200, "status": "settled", "with_nested_markets": "true"}
        if cur: p["cursor"] = cur
        r = requests.get(f"{API}/events", params=p, headers=H, timeout=30); r.raise_for_status()
        d = r.json(); out += d.get("events", []); cur = d.get("cursor")
        if not cur or not d.get("events"): break
        time.sleep(0.35)
    return out
for s in sys.argv[1:]:
    ev = events(s)
    months = Counter(e["event_ticker"].split("-")[1][:5] for e in ev)
    nm = sum(len(e.get("markets") or []) for e in ev)
    vol = sum(float(m.get("volume_fp") or m.get("volume") or 0) for e in ev for m in (e.get("markets") or []))
    print(s, len(ev), "events", nm, "markets", "vol", round(vol), sorted(months.items()))
    json.dump(ev, open(f"work/ev_{s}.json", "w"))
