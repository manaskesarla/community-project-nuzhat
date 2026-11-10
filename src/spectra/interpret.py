"""Which wavenumber regions drive each class decision?  python -m src.spectra.interpret

Occlusion analysis on the held-out test particles: blank one wavenumber window at a time (set the
preprocessed first-derivative spectrum to 0 there) and measure how much the probability of the TRUE class
drops. Large drop = the model relies on that region. Outputs: outputs/interpretability.png, outputs/occlusion_importance.csv
"""
import joblib, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .train import D, OUT, ROOT

WIN_CM, STEP_CM = 100, 50     # window width / stride in cm^-1


def main():
    m = pd.read_csv(D / "meta_ftir.csv"); X = np.load(D / "Xp_ftir_d1.npy"); te = np.load(D / "split_test_idx.npy")
    grid = np.load(D / "grid_ftir.npy"); cal = joblib.load(ROOT / "models/ftir_svm.pkl")
    Xt, yt = X[te], m.polymer.to_numpy(object)[te]
    cls = list(cal.classes_); ti = np.array([cls.index(c) for c in yt])
    base = cal.predict_proba(Xt)[np.arange(len(Xt)), ti]
    step = int(STEP_CM / (grid[1] - grid[0])); w = int(WIN_CM / (grid[1] - grid[0]))
    starts = np.arange(0, len(grid) - w + 1, step); centres = grid[starts] + WIN_CM / 2
    drop = np.zeros((len(starts), len(cls)))
    for k, s in enumerate(starts):
        Xo = Xt.copy(); Xo[:, s:s + w] = 0
        d = base - cal.predict_proba(Xo)[np.arange(len(Xt)), ti]
        for j, c in enumerate(cls):
            drop[k, j] = d[yt == c].mean() if (yt == c).any() else np.nan
    df = pd.DataFrame(drop, index=centres, columns=cls); df.index.name = "window_centre_cm-1"
    df.to_csv(OUT / "occlusion_importance.csv")

    fig, axes = plt.subplots(len(cls), 1, figsize=(9, 1.6 * len(cls)), sharex=True)
    for ax, c in zip(axes, cls):
        mean_spec = Xt[yt == c].mean(0) if (yt == c).any() else np.zeros(len(grid))
        ax.plot(grid, mean_spec / (np.abs(mean_spec).max() + 1e-9), color="0.6", lw=0.8)
        ax2 = ax.twinx(); ax2.bar(centres, df[c], width=STEP_CM, color="#c0392b", alpha=0.7)
        ax.set_ylabel(c, rotation=0, ha="right", va="center"); ax.set_yticks([]); ax2.tick_params(labelsize=7)
    axes[-1].set_xlabel("Wavenumber (cm$^{-1}$)"); axes[-1].invert_xaxis()
    fig.suptitle("Where the model looks: probability drop when a 100 cm$^{-1}$ window is blanked (red)\n"
                 "grey = mean preprocessed (1st-derivative) test spectrum of the class", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "interpretability.png", dpi=150)
    top = {c: [int(x) for x in df[c].nlargest(3).index] for c in cls}
    print("Top-3 window centres (cm^-1) per class:"); [print(f"  {c:14s}{v}") for c, v in top.items()]


if __name__ == "__main__":
    main()
