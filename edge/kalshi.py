"""Join Kalshi NBA markets + candles to ESPN games and players; pre-tip quotes; fees."""
import json, math, re, unicodedata
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
K2E = {"GSW": "GS", "NOP": "NO", "NYK": "NY", "SAS": "SA", "UTA": "UTAH", "WAS": "WSH"}
MON = {m: i for i, m in enumerate(["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], 1)}
CODES = sorted({"ATL", "BKN", "BOS", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GSW", "HOU", "IND", "LAC", "LAL", "MEM", "MIA",
                "MIL", "MIN", "NOP", "NYK", "OKC", "ORL", "PHI", "PHX", "POR", "SAC", "SAS", "TOR", "UTA", "WAS"})


def fee(p):
    """Kalshi taker fee per contract (quadratic, 7%), ignoring the per-order cent rounding."""
    return 0.07 * p * (1 - p)


def parse_event(et):
    """KXNBAPTS-26NOV10MILNYK -> (date, away, home) with ESPN abbreviations."""
    code = et.split("-")[1]
    m = re.match(r"(\d\d)([A-Z]{3})(\d\d)([A-Z]+)$", code)
    if not m:
        return None
    yy, mon, dd, teams = m.groups()
    d = pd.Timestamp(year=2000 + int(yy), month=MON[mon], day=int(dd))
    for i in (3, 2, 4):
        a, h = teams[:i], teams[i:]
        if a in CODES and h in CODES:
            return d, K2E.get(a, a), K2E.get(h, h)
    return None


def game_index(G):
    """(ET date, away, home) -> gid."""
    et = (G["tip"].dt.tz_convert("America/New_York")).dt.tz_localize(None).dt.normalize()
    return {(d, a, h): gid for d, a, h, gid in zip(et, G["away"], G["home"], G["gid"])}


def load_markets(series):
    return [json.loads(l) for l in open(ROOT / "work" / f"kmk_{series}.jsonl")]


def load_candles(series):
    out = {}
    p = ROOT / "work" / f"kc_{series}.jsonl"
    if p.exists():
        for l in p.open():
            t, c = json.loads(l)
            out[t] = c
    return out


def quote_at(c, t_end, max_age=3 * 3600):
    """Last hourly candle ending at or before t_end with a sane two-sided quote -> (bid, ask, ts)."""
    best = None
    for end, b, a, v in c:
        if end <= t_end and b is not None and a is not None and 0 < a <= 1 and 0 <= b < a:
            if best is None or end > best[2]:
                best = (b, a, end)
    if best and t_end - best[2] <= max_age:
        return best
    return None


def norm_name(s):
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", s)
    return re.sub(r"[^a-z]", "", s)


def prop_rows(series, G, P, stat):
    """One row per settled Kalshi prop market with a pre-tip quote: gid, pid, line k (stat >= k), bid/ask, y."""
    gi = game_index(G)
    tips = dict(zip(G["gid"], G["tip"]))
    C = load_candles(series)
    pl = P[P["played"]].copy()
    pl["nn"] = pl["name"].map(norm_name)
    by_game = {g: d for g, d in pl.groupby("gid")}
    rows, miss = [], {"game": 0, "player": 0, "quote": 0, "dnp": 0}
    for m in load_markets(series):
        if m.get("result") not in ("yes", "no") or m["ticker"] not in C:
            continue
        pe = parse_event(m["event_ticker"])
        gid = gi.get(pe) if pe else None
        if gid is None:
            miss["game"] += 1; continue
        name, _, k = (m.get("yes_sub_title") or "").rpartition(":")
        try:
            k = float(k.strip().rstrip("+"))
        except ValueError:
            continue
        d = by_game.get(gid)
        nn = norm_name(name)
        hit = d[d["nn"] == nn] if d is not None else None
        if hit is None or hit.empty:
            # last-name fallback within the game
            if d is not None:
                last = norm_name(name.split()[-1]) if name.split() else ""
                hit = d[d["nn"].str.endswith(last) & d["nn"].str.startswith(nn[:1])] if last else d.iloc[0:0]
            if hit is None or len(hit) != 1:
                miss["player"] += 1; continue
        r = hit.iloc[0]
        tip = int(tips[gid].timestamp())
        q = quote_at(C[m["ticker"]], tip)
        if not q:
            miss["quote"] += 1; continue
        q3 = quote_at(C[m["ticker"]], tip - 3 * 3600)
        rows.append({"ticker": m["ticker"], "event": m["event_ticker"], "hist": bool(m.get("hist")), "gid": gid, "pid": r["pid"], "name": r["name"], "team": r["team"], "stat": stat,
                     "k": k, "bid": q[0], "ask": q[1], "qts": q[2], "tip": tip, "bid3": q3[0] if q3 else np.nan,
                     "ask3": q3[1] if q3 else np.nan, "vol": float(m.get("volume_fp") or m.get("volume") or 0),
                     "pvol": sum(v for e, b, a, v in C[m["ticker"]] if e <= tip),
                     "y": 1 if m["result"] == "yes" else 0, "actual": r[stat]})
    R = pd.DataFrame(rows)
    if len(R):
        bad = (R["actual"] >= R["k"]).astype(int) != R["y"]
        miss["result_mismatch"] = int(bad.sum())
    return R, miss
