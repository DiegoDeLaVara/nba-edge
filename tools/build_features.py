"""Rebuild every research intermediate from work/espn_games.jsonl.gz:
G/P (games, player-games), Q (played rows + rolling features), A (absences/rotation), T2 (team context),
gm_all (walk-forward game model), D (player training frame). Then run tools/proj_wf.py for projections.

  ..\\tennis-edge\\.venv\\Scripts\\python.exe tools\\build_features.py
"""
import sys; sys.path.insert(0, ".")
import pandas as pd
from edge.data import load, team_games
from edge.features import player_features, team_absence, team_context
from edge.gamemodel import team_rows, walk_forward, game_preds
from edge.props import build

G, P = load(); G.to_pickle("work/G.pkl"); P.to_pickle("work/P.pkl")
P2, Q = player_features(P); Q.to_pickle("work/Q.pkl")
A = team_absence(P2, Q); A.to_pickle("work/A.pkl")
T = team_games(G, P); T2 = team_context(G, T); T2.to_pickle("work/T2.pkl")
R = team_rows(G, A); X = game_preds(G, R, walk_forward(R, hl_days=60, alpha=5.0))
X["m"] = X.hs - X["as"]; X["t"] = X.hs + X["as"]; X.to_pickle("work/gm_all.pkl")
D = build(Q, A, T2, X); D.to_pickle("work/D.pkl")
print("rebuilt", len(G), "games,", len(D), "player-game rows")
