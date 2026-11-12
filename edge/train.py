"""Fit the live models on the full history: one HistGBM per stat (same settings as the walk-forward research) and the
team-ratings ridge as of today. Saved with pickle (versions pinned in requirements.txt)."""
import pickle, time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from edge.features import player_features, team_absence, team_context
from edge.data import team_games
from edge.gamemodel import team_rows, design, walk_forward, game_preds
from edge.props import build, feats, STATS

PARAMS = dict(loss="poisson", learning_rate=0.05, max_iter=400, max_leaf_nodes=31, min_samples_leaf=80, l2_regularization=1.0)


def game_ridge(R, hl_days=60, alpha=5.0):
    d = R["day"].max() + pd.Timedelta(days=1)
    tr = R[(R["day"] >= d - pd.Timedelta(days=400)) & R["pts"].notna()]
    age = np.array([(d - x).days for x in tr["day"]], dtype=float)
    w = 0.5 ** (age / hl_days) * np.where(tr["season"].values != tr["season"].max(), 0.5, 1.0)
    return Ridge(alpha=alpha).fit(design(tr), tr["pts"].values, sample_weight=w)


def fit_all(G, P, log=print):
    t = time.time()
    Pf, Q = player_features(P)
    A = team_absence(Pf, Q)
    T2 = team_context(G, team_games(G, Pf))
    R = team_rows(G, A)
    Pr = walk_forward(R, hl_days=60, alpha=5.0, start=(G["tip"].max() - pd.Timedelta(days=500)).date())
    X = game_preds(G, R, Pr)
    D = build(Q, A, T2, X)
    D = D[D["n_prev"] >= 3]
    log(f"features {len(D)} rows in {time.time() - t:.0f}s")
    models = {"game": game_ridge(R)}
    for st in STATS:
        tr = D[D[st].notna()]
        models[st] = HistGradientBoostingRegressor(**PARAMS).fit(tr[feats(st)], tr[st])
    models["trained"] = str(G["tip"].max())
    log(f"trained in {time.time() - t:.0f}s")
    return models


def save(models, path):
    with open(path, "wb") as fh:
        pickle.dump(models, fh)


def load(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)
