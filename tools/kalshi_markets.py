"""List every settled Kalshi NBA market (live + /historical) for the series we study, compact fields only.

  python -m tools.kalshi_markets [--series ...]
Writes work/kmk_<SERIES>.jsonl.
"""
import argparse, json, time
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
API = "https://api.elections.kalshi.com/trade-api/v2"
S = requests.Session(); S.headers.update({"User-Agent": "nba-edge-research/1.0"})
SERIES = ["KXNBAGAME", "KXNBASPREAD", "KXNBATOTAL", "KXNBAPTS", "KXNBAREB", "KXNBAAST", "KXNBA3PT", "KXNBAPRA",
          "KXNBAPR", "KXNBAPA", "KXNBARA", "KXNBATEAMTOTAL", "KXNBA2D", "KXNBA3D"]
KEEP = ["ticker", "event_ticker", "yes_sub_title", "no_sub_title", "title", "result", "open_time", "close_time",
        "expected_expiration_time", "occurrence_datetime", "floor_strike", "cap_strike", "strike_type", "custom_strike",
        "volume", "volume_fp", "settlement_value", "settlement_value_dollars", "expiration_value"]


def kget(path, params, gap=0.22):
    for i in range(6):
        try:
            r = S.get(API + path, params=params, timeout=40)
        except requests.RequestException:
            time.sleep(3 + 3 * i); continue
        if r.status_code == 200:
            time.sleep(gap); return r.json()
        if r.status_code == 429:
            time.sleep(2 + 3 * i); continue
        return {}
    return {}


def listing(path, series):
    out, cur = [], ""
    while True:
        p = {"series_ticker": series, "limit": 1000}
        if path == "/markets":
            p["status"] = "settled"
        if cur:
            p["cursor"] = cur
        d = kget(path, p)
        ms = d.get("markets") or []
        out += [{k: m.get(k) for k in KEEP if m.get(k) is not None} for m in ms]
        cur = d.get("cursor") or ""
        if not cur or not ms:
            return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--series", nargs="*", default=SERIES)
    a = ap.parse_args()
    for s in a.series:
        live = listing("/markets", s); hist = listing("/historical/markets", s)
        for m in hist: m["hist"] = True
        seen, rows = set(), []
        for m in live + hist:
            if m["ticker"] not in seen:
                seen.add(m["ticker"]); rows.append(m)
        with (ROOT / "work" / f"kmk_{s}.jsonl").open("w") as fh:
            for m in rows:
                fh.write(json.dumps(m, separators=(",", ":")) + "\n")
        print(s, "live", len(live), "hist", len(hist), "events", len({m["event_ticker"] for m in rows}), flush=True)


if __name__ == "__main__":
    main()
