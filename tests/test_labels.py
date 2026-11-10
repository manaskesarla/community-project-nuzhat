import numpy as np, pandas as pd
from pathlib import Path
from src.spectra.labels import map_label

def test_mapping():
    assert map_label("HDPE") == "PE" and map_label("PET G") == "PET"
    assert map_label("fibre_polyamide_6") == "PA" and map_label("polyvinylchloride") == "PVC"
    assert map_label("EVA") == "OTHER_PLASTIC" and map_label("SAN") == "OTHER_PLASTIC"
    assert map_label("PBT") == "OTHER_PLASTIC" and map_label("fibre_cotton_combers") == "NON_PLASTIC"
    assert map_label("Unknown") is None and map_label(np.nan) is None

def test_dataset_shapes():
    d = Path(__file__).resolve().parents[1] / "data/spectra_processed"
    X, m, g = np.load(d/"X_ftir.npy"), pd.read_csv(d/"meta_ftir.csv"), np.load(d/"grid_ftir.npy")
    assert X.shape == (len(m), len(g)) and np.isfinite(X).all()
    assert m["id"].is_unique
