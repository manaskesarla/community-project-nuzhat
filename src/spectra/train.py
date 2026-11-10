"""Train/evaluate FTIR classifiers with particle-grouped CV.  python -m src.spectra.train"""
import json, time, warnings
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from .preprocess import preprocess

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]; D = ROOT / "data/spectra_processed"; OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
SEED = 42

def models():
    return {
        "PCA+SVM": make_pipeline(PCA(40, random_state=SEED), SVC(C=10, probability=True, class_weight="balanced", random_state=SEED)),
        "RandomForest": RandomForestClassifier(300, class_weight="balanced_subsample", n_jobs=-1, random_state=SEED),
        "kNN-corr": KNeighborsClassifier(5, metric="correlation", algorithm="brute", weights="distance"),
        "LogReg": make_pipeline(PCA(60, random_state=SEED), LogisticRegression(C=1, max_iter=2000, class_weight="balanced")),
    }

def main():
    m = pd.read_csv(D / "meta_ftir.csv"); X = np.load(D / "X_ftir.npy")
    t = time.time(); Xp = preprocess(X, deriv=1); print("preprocess", round(time.time() - t), "s")
    np.save(D / "Xp_ftir_d1.npy", Xp)
    y = m.polymer.to_numpy(object); g = m.group.to_numpy(object)
    # final hold-out test set: 20% of particles, never used for model selection
    tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=SEED).split(Xp, y, g))
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
    res = {}
    for name, mdl in models().items():
        f1s, accs = [], []
        for a, b in cv.split(Xp[tr], y[tr], g[tr]):
            mdl.fit(Xp[tr][a], y[tr][a]); p = mdl.predict(Xp[tr][b])
            f1s.append(f1_score(y[tr][b], p, average="macro")); accs.append(accuracy_score(y[tr][b], p))
        res[name] = dict(cv_acc=float(np.mean(accs)), cv_acc_sd=float(np.std(accs)), cv_f1=float(np.mean(f1s)), cv_f1_sd=float(np.std(f1s)))
        print(name, {k: round(v, 3) for k, v in res[name].items()}, flush=True)
    json.dump(dict(cv=res, n_train=int(len(tr)), n_test=int(len(te))), open(OUT / "cv_results.json", "w"), indent=2)
    np.save(D / "split_test_idx.npy", te)

if __name__ == "__main__":
    main()
