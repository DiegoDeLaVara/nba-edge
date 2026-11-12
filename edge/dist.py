"""Turn a projected mean into P(stat >= k): variance function var(mu) plus a logistic calibration on the standardised
gap, both fitted only on games before the cut date (2025-11-01 for the research, rolling in the live job)."""
import json
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

GRID = {"pts": np.arange(5, 51, 5), "reb": np.arange(2, 19, 2), "ast": np.arange(2, 17, 2), "fg3m": np.arange(1, 9),
        "pra": np.arange(10, 66, 5)}


def var_fit(mu, y):
    r2 = (y - mu) ** 2
    A = np.c_[np.ones_like(mu), mu, mu ** 2]
    c, *_ = np.linalg.lstsq(A, r2, rcond=None)
    return c.tolist()


def zfeat(mu, k, c):
    v = np.maximum(c[0] + c[1] * mu + c[2] * mu ** 2, 0.25 * mu + 0.05)
    z = (k - 0.5 - mu) / np.sqrt(v)
    lm = np.log(np.maximum(mu, 0.1))
    return np.c_[z, z * lm, lm, z ** 2 * np.sign(z)]


class Dist:
    def __init__(self, c=None, w=None, b=None):
        self.c, self.w, self.b = c, w, b

    def fit(self, mu, y, grid):
        self.c = var_fit(mu, y)
        M = np.repeat(mu, len(grid)); K = np.tile(grid, len(mu)); Y = (np.repeat(y, len(grid)) >= K).astype(int)
        keep = (np.abs(K - M) < 4 * np.sqrt(np.maximum(M, 1)) + 3)
        lr = LogisticRegression(C=10.0, max_iter=500).fit(zfeat(M[keep], K[keep], self.c), Y[keep])
        self.w, self.b = lr.coef_[0].tolist(), float(lr.intercept_[0])
        return self

    def p_over(self, mu, k):
        mu = np.asarray(mu, float); k = np.asarray(k, float)
        x = zfeat(mu, k, self.c) @ np.array(self.w) + self.b
        return 1 / (1 + np.exp(-x))

    def to_json(self):
        return {"c": self.c, "w": self.w, "b": self.b}

    @classmethod
    def from_json(cls, d):
        return cls(d["c"], d["w"], d["b"])
