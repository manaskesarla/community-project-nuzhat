"""Build the two CONSTRUCTED demo samples in samples/.  python -m src.report.make_demo

Each sample = one SYNTHETIC filter image + spectra drawn from the held-out TEST particles with a chosen composition.
These are demos of the pipeline, NOT field samples. The chosen composition is stored in demo_truth.json (the
pipeline never reads it; it is only used to check the output).
"""
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "data/spectra_processed"
SAMPLES = {
    "sample_001": dict(image="SYNTH_0032", comp={"PE": .40, "PP": .25, "PET": .15, "PS": .08, "PA": .04, "OTHER_PLASTIC": .08},
                       location="(demo) river water, constructed", volume_L=1.0),
    "sample_002": dict(image="SYNTH_0035", comp={"PET": .35, "PS": .25, "PE": .15, "PP": .10, "PA": .10, "OTHER_PLASTIC": .05},
                       location="(demo) tap water, constructed", volume_L=2.0),
}
N_SPECTRA = 50


def main():
    m = pd.read_csv(D / "meta_ftir.csv"); X = np.load(D / "X_ftir.npy"); grid = np.load(D / "grid_ftir.npy")
    te = np.load(D / "split_test_idx.npy"); cfg = yaml.safe_load(open(ROOT / "config.yaml")); rng = np.random.default_rng(cfg["seed"])
    for sid, s in SAMPLES.items():
        out = ROOT / "samples" / sid; shutil.rmtree(out, ignore_errors=True); (out / "spectra").mkdir(parents=True)
        shutil.copy(ROOT / "data/images_synthetic" / f"{s['image']}.png", out / "image.png")
        used, truth = set(), {}
        for cls, frac in s["comp"].items():
            n = int(round(frac * N_SPECTRA)); pool = [i for i in te if m.polymer[i] == cls]; picks = []
            for i in rng.permutation(pool):                      # one spectrum per physical particle
                if m.group[i] not in used and len(picks) < n:
                    picks.append(i); used.add(m.group[i])
            truth[cls] = len(picks)
            for i in picks:
                np.savetxt(out / "spectra" / f"{cls}_{i}.csv", np.c_[grid, X[i]], delimiter=",", header="wavenumber_cm-1,absorbance", comments="", fmt="%.5g")
        # shuffle names so the folder listing does not reveal the class
        for k, f in enumerate(sorted((out / "spectra").glob("*.csv"), key=lambda _: rng.random())):
            f.rename(out / "spectra" / f"particle_{k + 1:03d}.csv")
        yaml.safe_dump(dict(id=sid, location=s["location"], date="2026-01-01 (placeholder)", volume_filtered_L=s["volume_L"],
                            pore_size_um=1.0, um_per_pixel=cfg["um_per_pixel"], image_is_synthetic=True, spectra_are_constructed_demo=True,
                            image_source=f"data/images_synthetic/{s['image']}.png"), open(out / "meta.yaml", "w"), sort_keys=False)
        json.dump(dict(chosen_composition_counts=truth, n_spectra=sum(truth.values()),
                       note="used only to verify the demo; pipeline does not read this file"), open(out / "demo_truth.json", "w"), indent=1)
        print(sid, truth)


if __name__ == "__main__":
    main()
