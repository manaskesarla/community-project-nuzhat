"""Identify the polymer of a new raw FTIR spectrum.

Input: a text/CSV file with two columns (wavenumber in cm^-1, ABSORBANCE). A header line is allowed.
Usage:  python -m src.spectra.predict path/to/spectrum.csv [--threshold 0.5]
The same grid + preprocessing as training is applied (src/spectra/preprocess.py).
Transmittance data must be converted to absorbance first (A = -log10(T)).
"""
import argparse
from pathlib import Path

import joblib
import numpy as np

from .preprocess import preprocess

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "data/spectra_processed"
MODEL = ROOT / "models/ftir_svm.pkl"
_model = None


def read_spectrum(path):
    """Read (wavenumber, absorbance) from a 2-column file; skips non-numeric lines."""
    rows = []
    for line in Path(path).read_text(errors="ignore").splitlines():
        parts = line.replace(";", ",").replace("\t", ",").split(",") if "," in line or "\t" in line or ";" in line else line.split()
        try:
            rows.append((float(parts[0]), float(parts[1])))
        except (ValueError, IndexError):
            continue
    if len(rows) < 20:
        raise ValueError(f"{path}: expected two numeric columns (wavenumber, absorbance); found {len(rows)} usable rows")
    a = np.array(rows)
    return a[:, 0], a[:, 1]


def to_grid(wn, y, grid):
    o = np.argsort(wn)
    wn, y = wn[o], y[o]
    if wn.min() > grid.min() + 20 or wn.max() < grid.max() - 20:
        raise ValueError(f"spectrum covers {wn.min():.0f}-{wn.max():.0f} cm^-1; need about {grid.min():.0f}-{grid.max():.0f}")
    return np.interp(grid, wn, y).astype(np.float32)


def predict_spectrum(wn, y, threshold=0.5):
    """Return dict: polymer (or 'UNCERTAIN'), confidence, best_guess, probabilities."""
    global _model
    if _model is None:
        _model = joblib.load(MODEL)
    grid = np.load(D / "grid_ftir.npy")
    x = preprocess(to_grid(np.asarray(wn, float), np.asarray(y, float), grid)[None], deriv=1)
    p = _model.predict_proba(x)[0]
    order = np.argsort(-p)
    best, conf = str(_model.classes_[order[0]]), float(p[order[0]])
    return dict(polymer=best if conf >= threshold else "UNCERTAIN", best_guess=best, confidence=conf,
                probabilities={str(_model.classes_[i]): float(p[i]) for i in order})


def predict_file(path, threshold=0.5):
    return predict_spectrum(*read_spectrum(path), threshold=threshold)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--threshold", type=float, default=0.5, help="below this confidence the answer is UNCERTAIN")
    a = ap.parse_args()
    r = predict_file(a.file, a.threshold)
    print(f"Polymer: {r['polymer']}  (best guess {r['best_guess']}, confidence {r['confidence']:.2f})")
    for k, v in r["probabilities"].items():
        print(f"  {k:14s}{v:6.3f}")


if __name__ == "__main__":
    main()
