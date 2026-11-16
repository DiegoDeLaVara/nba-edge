"""Settled Kalshi combos (MVE) with at least one NBA prop leg, closing 02:30-08:00 UTC on each night of the season.
-> work/combos.jsonl: ticker, result, last_price, volume, open/close time, legs [(market_ticker, side)]."""
import sys, json, time, zlib, datetime as dt; sys.path.insert(0, ".")
import requests

S = requests.Session(); S.headers.update({"User-Agent": "nba-edge-research/1.0"})
API = "https://api.elections.kalshi.com/trade-api/v2"
PROP = ("KXNBAPTS", "KXNBAREB", "KXNBAAST", "KXNBA3PT", "KXNBAPRA", "KXNBA2D", "KXNBA3D")
OUT = "work/combos_full.jsonl"


def get(params):
    for i in range(40):                      # never give up mid-night: a partial night over-samples winners
        try:
            r = S.get(API + "/markets", params=params, timeout=60)
            if r.status_code == 200:
                return r.json()
            time.sleep(min(60, 2 + 2 * i))
        except requests.RequestException:
            time.sleep(min(60, 3 + 2 * i))
    raise SystemExit("Kalshi kept failing; stopping so no partial night is written")


NIGHTS = [dt.date.fromisoformat(x) for x in sys.argv[1:]]
seen = set()
try:
    seen = {json.loads(l)["ticker"] for l in open(OUT)}
except FileNotFoundError:
    pass
with open(OUT, "a") as fh:
    for d in NIGHTS:
        # whole night: from before the first tip (18:00 UTC) to well after the last game (09:00 UTC next day), so combos
        # that settle EARLY as losses are not cut off (a narrow window over-sampled winners)
        a = int(dt.datetime(d.year, d.month, d.day, 18, 0, tzinfo=dt.timezone.utc).timestamp())
        b = a + 15 * 3600
        cur, n, kept = "", 0, 0
        while True:
            q = {"mve_filter": "only", "status": "settled", "limit": 1000, "min_close_ts": a, "max_close_ts": b}
            if cur:
                q["cursor"] = cur
            r = get(q)
            ms = r.get("markets") or []
            n += len(ms)
            for m in ms:
                legs = m.get("mve_selected_legs") or []
                sers = [l["market_ticker"].split("-")[0] for l in legs]
                if (m["ticker"] in seen or not any(x in PROP for x in sers) or not all(x.startswith("KXNBA") for x in sers)
                        or not 2 <= len(sers) <= 8 or zlib.crc32(m["ticker"].encode()) % 100 >= 15):
                    continue
                seen.add(m["ticker"]); kept += 1
                fh.write(json.dumps({"ticker": m["ticker"], "result": m.get("result"), "last": m.get("last_price_dollars"),
                                     "vol": m.get("volume_fp"), "open": m.get("open_time"), "close": m.get("close_time"),
                                     "legs": [[l["market_ticker"], l["side"]] for l in legs]}, separators=(",", ":")) + "\n")
            cur = r.get("cursor")
            if not cur or not ms:
                break
            time.sleep(0.15)
        fh.flush(); print(d, "scanned", n, "kept", kept, "COMPLETE", time.strftime("%H:%M:%S"), flush=True)
