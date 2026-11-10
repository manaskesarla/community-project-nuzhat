"""FTIR preprocessing (doc 03). Same function is used at train and inference time."""
import numpy as np
from pybaselines import Baseline
from scipy.signal import savgol_filter

SAT = 3.0          # absorbance above this is detector-saturated / non-linear -> clip


def preprocess_one(y, window=11, poly=3, deriv=0, lam=1e5):
    y = np.clip(np.asarray(y, float), -0.05, SAT)
    y = savgol_filter(y, window, poly)
    y = y - Baseline(np.arange(len(y))).asls(y, lam=lam, p=0.01)[0]
    if deriv:
        y = savgol_filter(y, window, poly, deriv=deriv)
    s = y.std()
    return ((y - y.mean()) / s if s > 1e-9 else y * 0).astype(np.float32)   # SNV


def preprocess(X, **kw):
    return np.stack([preprocess_one(x, **kw) for x in X])
