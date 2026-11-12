"""Collect NBA box scores + DraftKings closing lines (ESPN pickcenter) from ESPN's public site API.

  python -m tools.espn_collect --start 2022-10-18 --end 2026-11-10
Writes work/espn_games.jsonl (one compact row per game, resumable).
"""
import argparse, json, time, datetime as dt
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "work" / "espn_games.jsonl"
SITE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
H = {"User-Agent": "nba-edge-research/1.0"}
S = requests.Session(); S.headers.update(H)


def get(url, params=None, tries=4):
    for i in range(tries):
        try:
            r = S.get(url, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503):
                time.sleep(2 + 3 * i); continue
            return None
        except requests.RequestException:
            time.sleep(2 + 3 * i)
    return None


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def parse_players(team_block):
    out = []
    for st in team_block.get("statistics") or []:
        keys = st.get("keys") or []
        for a in st.get("athletes") or []:
            ath = a.get("athlete") or {}
            row = {"pid": ath.get("id"), "name": ath.get("displayName"), "pos": (ath.get("position") or {}).get("abbreviation"),
                   "starter": bool(a.get("starter")), "dnp": bool(a.get("didNotPlay")), "reason": a.get("reason") or "",
                   "ejected": bool(a.get("ejected"))}
            stats = a.get("stats") or []
            if stats and len(stats) == len(keys):
                for k, v in zip(keys, stats):
                    if "-" in k and k.count("-") == 1 and "-" in str(v):
                        k1, k2 = k.split("-"); v1, v2 = str(v).split("-", 1)
                        row[k1] = num(v1); row[k2] = num(v2)
                    else:
                        row[k] = num(str(v).replace("+", "")) if k == "plusMinus" else num(v)
            out.append(row)
    return out


def game(eid):
    d = get(f"{SITE}/summary", {"event": eid})
    if not d:
        return None
    hdr = d.get("header") or {}
    comp = (hdr.get("competitions") or [{}])[0]
    teams = {}
    for c in comp.get("competitors") or []:
        t = c.get("team") or {}
        teams[c.get("homeAway")] = {"id": t.get("id"), "abbr": t.get("abbreviation"), "score": num(c.get("score")),
                                    "winner": c.get("winner"), "ls": [num(x.get("displayValue")) for x in c.get("linescores") or []]}
    players = {}
    for b in (d.get("boxscore") or {}).get("players") or []:
        players[(b.get("team") or {}).get("abbreviation")] = parse_players(b)
    tstats = {}
    for b in (d.get("boxscore") or {}).get("teams") or []:
        tstats[(b.get("team") or {}).get("abbreviation")] = {s.get("name"): s.get("displayValue") for s in b.get("statistics") or []}
    pc = []
    for p in d.get("pickcenter") or []:
        p = dict(p); p.pop("links", None)
        for side in ("homeTeamOdds", "awayTeamOdds"):
            if isinstance(p.get(side), dict):
                p[side] = {k: v for k, v in p[side].items() if k not in ("team", "links")}
        p["provider"] = (p.get("provider") or {}).get("name")
        pc.append(p)
    gi = d.get("gameInfo") or {}
    return {"id": eid, "date": comp.get("date"), "season": (hdr.get("season") or {}).get("year"),
            "stype": (hdr.get("season") or {}).get("type"), "neutral": comp.get("neutralSite"),
            "status": ((comp.get("status") or {}).get("type") or {}).get("name"),
            "home": teams.get("home"), "away": teams.get("away"), "players": players, "tstats": tstats, "pick": pc,
            "venue": (gi.get("venue") or {}).get("fullName"), "att": gi.get("attendance"),
            "notes": [n.get("headline") for n in comp.get("notes") or [] if n.get("headline")]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2022-10-18"); ap.add_argument("--end", default=None)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    d0 = dt.date.fromisoformat(a.start); d1 = dt.date.fromisoformat(a.end) if a.end else dt.date.today()
    done = set()
    if OUT.exists():
        done = {json.loads(l)["id"] for l in OUT.open()}
    ids = []
    d = d0
    while d <= d1:
        if d.month not in (7, 8, 9):
            sb = get(f"{SITE}/scoreboard", {"dates": d.strftime("%Y%m%d"), "limit": 50})
            for e in (sb or {}).get("events") or []:
                if ((e.get("status") or {}).get("type") or {}).get("completed") and e["id"] not in done:
                    ids.append(e["id"])
            time.sleep(0.15)
        d += dt.timedelta(days=1)
    print("games to fetch", len(ids), flush=True)
    n = 0
    with OUT.open("a") as fh, ThreadPoolExecutor(a.workers) as ex:
        for g in ex.map(game, ids):
            if g:
                fh.write(json.dumps(g, separators=(",", ":")) + "\n"); n += 1
                if n % 200 == 0:
                    print(n, g["date"], flush=True); fh.flush()
    print("done", n)


if __name__ == "__main__":
    main()
