"""Official NBA injury reports (ak-static.cms.nba.com, one PDF every 15 min on game days), parsed with the standard
library only (Flate text streams, column by x position). PDFs are not kept.
usage: injury_reports.py 2026-10-20 2026-11-16   -> work/ir_rows.jsonl (one row per report x player)"""
import sys, re, json, zlib, datetime as dt, threading
from concurrent.futures import ThreadPoolExecutor
import requests

S = requests.Session(); S.headers.update({"User-Agent": "nba-edge-research/1.0"})
S.mount("https://", requests.adapters.HTTPAdapter(pool_maxsize=8))
URL = "https://ak-static.cms.nba.com/referee/injury/Injury-Report_{}.pdf"
COLS = [("date", 0), ("time", 110), ("matchup", 190), ("team", 255), ("player", 415), ("status", 575), ("reason", 655)]


def et_to_utc(d, h, m):
    # US Eastern: EDT (UTC-4) until the first Sunday of November 02:00, EST (UTC-5) after
    nov1 = dt.date(d.year, 11, 1); first_sun = nov1 + dt.timedelta((6 - nov1.weekday()) % 7)
    off = 4 if d < first_sun else 5
    return dt.datetime(d.year, d.month, d.day, h, m, tzinfo=dt.timezone.utc) + dt.timedelta(hours=off)


def unesc(s):
    return re.sub(rb"\\([()\\])", rb"\1", s).decode("latin-1")


def parse(pdf):
    rows = []
    for m in re.finditer(rb"stream\r?\n", pdf):
        e = pdf.find(b"endstream", m.end())
        try:
            x = zlib.decompress(pdf[m.end():e])
        except zlib.error:
            continue
        if b"Tj" not in x:
            continue
        cells = []                                     # (y, x, text)
        for bt in re.finditer(rb"BT(.*?)ET", x, re.S):
            body = bt.group(1)
            tds = re.findall(rb"(-?[\d.]+) (-?[\d.]+) Td", body)
            words = [unesc(w) for w in re.findall(rb"\(((?:\\.|[^\\)])*)\) Tj", body)]
            if not tds or not words:
                continue
            cells.append((round(float(tds[0][1]), 0), float(tds[0][0]), " ".join(words)))
        lines = {}
        for y, xx, t in cells:
            lines.setdefault(y, []).append((xx, t))
        for y in sorted(lines):
            r = {}
            for xx, t in sorted(lines[y]):
                col = [c for c, x0 in COLS if xx >= x0][-1]
                r[col] = (r.get(col, "") + " " + t).strip()
            rows.append(r)
    return rows


def one(args):
    d, h, mi = args
    tag = f"{d.isoformat()}_{(h % 12) or 12:02d}_{mi:02d}{'AM' if h < 12 else 'PM'}"
    for i in range(4):
        try:
            r = S.get(URL.format(tag), timeout=30)
            if r.status_code in (403, 404):
                return []
            if r.status_code == 200:
                ts = et_to_utc(d, h, mi).isoformat()
                out, cur = [], {}
                for row in parse(r.content):
                    if row.get("player") in (None, "Player Name") or "Page" in row.get("player", ""):
                        if "team" in row and row["team"] != "Team":   # 'NOT YET SUBMITTED' team lines
                            cur["team"] = row["team"]
                        continue
                    for k in ("date", "time", "matchup", "team"):
                        if row.get(k):
                            cur[k] = row[k]
                    if "status" not in row:
                        continue
                    out.append({"ts": ts, **{k: cur.get(k) for k in ("date", "time", "matchup", "team")},
                                "player": row["player"], "status": row["status"], "reason": row.get("reason", "")})
                return out
        except requests.RequestException:
            pass
    return []


if __name__ == "__main__":
    a, b = (dt.date.fromisoformat(x) for x in sys.argv[1:3])
    jobs = []
    d = a
    while d <= b:
        jobs += [(d, h, m) for h in range(24) for m in (0, 15, 30, 45)]
        d += dt.timedelta(1)
    n = 0
    with open("work/ir_rows.jsonl", "w") as fh, ThreadPoolExecutor(6) as ex:
        for i, rows in enumerate(ex.map(one, jobs)):
            n += bool(rows)
            for r in rows:
                fh.write(json.dumps(r, separators=(",", ":")) + "\n")
            if i % 400 == 0:
                print(i, len(jobs), "reports", n, flush=True)
    print("done reports", n, flush=True)
