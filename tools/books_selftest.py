"""End-to-end check of the line-shopping path with a FAKE sportsbook feed built from live Kalshi prices
(no Odds API credits used). Writes into out_bt/ (copy of out/)."""
import sys, shutil, json; sys.path.insert(0, ".")
import pandas as pd
from edge import refresh, books, history
shutil.rmtree("out_bt", ignore_errors=True); shutil.copytree("out", "out_bt")
for f in ("books_log.json", "books_state.json"):
    try: (__import__("pathlib").Path("out_bt") / f).unlink()
    except FileNotFoundError: pass
now = pd.Timestamp.now(tz="UTC"); today = now.tz_convert("America/New_York").date()
games = [g for d in (today, today + pd.Timedelta(days=1)) for g in history.scoreboard(d) if not g["completed"]]
KL = books.kalshi_lines(refresh.kget, games)
NAME = {v: k for k, v in books.TEAM.items() if k != "LA Clippers"}
ev = []
for g in games:
    p = KL.get((g["gid"], "ml", g["home"], None))
    if not p: continue
    ph = p[0]
    tots = sorted([k[3] for k in KL if k[0] == g["gid"] and k[1] == "total"])
    mk = [{"key": "h2h", "outcomes": [{"name": NAME[g["home"]], "price": round(1.06 / ph, 3)}, {"name": NAME[g["away"]], "price": round(0.95 / (1 - ph), 3)}]}]
    if tots:
        t = tots[len(tots) // 2]; q = KL[(g["gid"], "total", None, t)][0]
        mk.append({"key": "totals", "outcomes": [{"name": "Over", "point": t, "price": round(0.97 / q, 3)}, {"name": "Under", "point": t, "price": round(1.04 / (1 - q), 3)}]})
    ev.append({"id": g["gid"], "home_team": NAME[g["home"]], "away_team": NAME[g["away"]], "commence_time": g["tip"],
               "bookmakers": [{"key": "draftkings", "markets": mk}]})
print("fake events", len(ev))
refresh._FAKE_EVENTS = ev
sys.argv = ["refresh", "--data", "out_bt"]
refresh.main()
B = json.load(open("out_bt/board.json"))["books"]
print(json.dumps(B["summary"]))
for e in B["open"][:8]: print(e["game"], "|", e["label"], "|", e["book_name"], e["american"], "fair", e["fair"], "EV", e["ev"], "units", e["units"], e["flag"])
