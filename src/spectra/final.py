"""Fit the final calibrated PCA+SVM on the training particles and score it once on the held-out test particles.
Run after `python -m src.spectra.train` (which creates the split).   python -m src.spectra.final
"""
import warnings
import joblib, numpy as np, pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from .train import D, OUT, ROOT, SEED

warnings.filterwarnings("ignore")

def main():
    m = pd.read_csv(D / "meta_ftir.csv"); X = np.load(D / "Xp_ftir_d1.npy"); te = np.load(D / "split_test_idx.npy")
    tr = np.setdiff1d(np.arange(len(m)), te)
    y = m.polymer.to_numpy(object); g = m.group.to_numpy(object)
    splits = list(StratifiedGroupKFold(3, shuffle=True, random_state=SEED).split(X[tr], y[tr], g[tr]))
    cal = CalibratedClassifierCV(make_pipeline(PCA(120, random_state=SEED), SVC(C=3, class_weight="balanced")),
                                 method="sigmoid", cv=splits).fit(X[tr], y[tr])
    P = cal.predict_proba(X[te]); p = cal.classes_[P.argmax(1)]; conf = P.max(1)
    print(classification_report(y[te], p, digits=3)); print("macro-F1 %.3f" % f1_score(y[te], p, average="macro"))
    for t in (0.5, 0.7):
        s = conf >= t; print(f"confidence>={t}: coverage {s.mean():.2f}, accuracy {(p[s] == y[te][s]).mean():.3f}")
    (ROOT / "models").mkdir(exist_ok=True); joblib.dump(cal, ROOT / "models/ftir_svm.pkl")
    pd.DataFrame(confusion_matrix(y[te], p, labels=cal.classes_), index=cal.classes_, columns=cal.classes_).to_csv(OUT / "confusion_test.csv")

if __name__ == "__main__":
    main()
