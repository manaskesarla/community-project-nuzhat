"""Build the unified FTIR dataset from databasedata/.

Outputs (data/spectra_processed/):
  X_ftir.npy        float32 (n, n_points)  absorbance on the common grid, UNnormalised (see doc 03)
  meta_ftir.csv     one row per spectrum: id, polymer, raw_label, mode, group, origin, ...
  class_table.csv   counts per class / mode / particle
Run:  python -m src.spectra.load
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .labels import map_label

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "databasedata"
OUT = ROOT / "data" / "spectra_processed"


def load_config():
    return yaml.safe_load(open(ROOT / "config.yaml"))


def make_grid(cfg):
    g = cfg["wavenumber_grid"]
    return np.arange(g["min"], g["max"] + 1e-9, g["step"])


def origin_of(row_id) -> str:
    s = "" if pd.isna(row_id) else str(row_id)
    if s.startswith("Lisa_"):
        return "environmental"      # real environmental particles
    if s.startswith("primpke"):
        return "primpke2018"
    return "library_other"


def main():
    cfg = load_config()
    grid = make_grid(cfg)

    meta = pd.read_csv(RAW / "Metadata_raw.csv", low_memory=False, encoding="latin-1")
    meta["raw_label"] = meta["SpectrumIdentity"]
    meta["polymer"] = meta["raw_label"].map(map_label)
    meta["mode"] = meta["SpectralCollectionMode"]
    meta["group"] = meta["Plate"].astype(str) + "_" + meta["Cell"].astype(str)   # physical particle
    meta["origin"] = meta["ID"].map(origin_of)
    meta = meta.rename(columns={"SpectrumID": "id"})
    n0 = len(meta)
    meta = meta[meta["polymer"].notna()].copy()
    print(f"dropped {n0 - len(meta)} unlabeled spectra")

    wide = pd.read_csv(RAW / "spread_os.csv")
    wn = wide["Wavelength"].to_numpy(float)
    order = np.argsort(wn)
    wn = wn[order]
    meta = meta[meta["id"].isin(wide.columns)].copy()
    Y = wide[meta["id"].tolist()].to_numpy(np.float32)[order].T          # (n, n_wn)
    X = np.stack([np.interp(grid, wn, y) for y in Y]).astype(np.float32)  # clamps outside range

    # coverage check: grid must lie inside the measured range
    assert grid.min() >= wn.min() - 2 and grid.max() <= wn.max() + 2, (wn.min(), wn.max())

    # de-duplicate identical spectra
    h = [hashlib.md5(np.round(x, 5).tobytes()).hexdigest() for x in X]
    meta["dup"] = pd.Series(h, index=meta.index).duplicated().to_numpy()
    keep = ~meta["dup"].to_numpy() & np.isfinite(X).all(1) & (X.std(1) > 1e-6)
    print(f"removed {(~keep).sum()} duplicate/flat/NaN spectra")
    X, meta = X[keep], meta[keep].reset_index(drop=True)

    cols = ["id", "polymer", "raw_label", "mode", "group", "origin", "Plate", "Cell", "ID",
            "SourceDatabase", "Plastic/other", "morphology", "Longest_dimension", "width"]
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "X_ftir.npy", X)
    np.save(OUT / "grid_ftir.npy", grid)
    meta[[c for c in cols if c in meta]].to_csv(OUT / "meta_ftir.csv", index=False)

    tab = (meta.groupby("polymer")
           .agg(spectra=("id", "size"), particles=("group", "nunique"))
           .join(meta.pivot_table(index="polymer", columns="mode", values="id", aggfunc="size", fill_value=0))
           .join(meta.pivot_table(index="polymer", columns="origin", values="id", aggfunc="size", fill_value=0)))
    tab.to_csv(OUT / "class_table.csv")
    print(tab.to_string()); print("X", X.shape, "grid", grid[0], grid[-1], len(grid))


if __name__ == "__main__":
    main()
