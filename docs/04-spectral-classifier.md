# 04 — Spectral Classifier (Days 5–8)

**Output:** trained model, evaluation report (accuracy, macro-F1, confusion matrix), calibrated confidence, `predict.py`.

## 1. Split correctly (most common mistake)
Use **group-aware splits** so spectra from the same source/sample never straddle train and test:
```python
from sklearn.model_selection import StratifiedGroupKFold
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
```
Group = `source` (or sample ID if known). Hold out a final **test set** untouched until the end. If only one source exists, state that in limitations.

## 2. Baselines (classical)
Train all, compare via cross-validation:
| Model | Notes |
|---|---|
| **Cosine / Pearson library matching** | What spectroscopists actually do. Nearest-neighbour on correlation. Strong baseline. |
| **PCA + SVM (RBF)** | `Pipeline([PCA(30–50), SVC(probability=True, class_weight="balanced")])` |
| **PLS-DA** | Chemometrics standard |
| **Random Forest** | Feature importances map to wavenumbers → interpretability |
| **kNN (correlation distance)** | Simple |

```python
pipe = make_pipeline(PCA(40), SVC(C=10, probability=True, class_weight="balanced"))
scores = cross_validate(pipe, X, y, groups=g, cv=cv, scoring=["accuracy","f1_macro"])
```
Tune with `GridSearchCV` (nested inside the group CV if time allows).

## 3. Metrics
- Accuracy **and macro-F1** (imbalance!)
- Per-class precision/recall, **confusion matrix** (expect PE↔PP confusion; PET↔PA sometimes)
- Report mean ± std across folds.

## 4. Confidence
- Use `predict_proba`; **calibrate** with `CalibratedClassifierCV(method="isotonic" or "sigmoid")`.
- Plot a **reliability curve** and report Expected Calibration Error.
- Define **confidence = max class probability**, with an **abstain / "unknown"** threshold (e.g. < 0.6 → "unidentified"). Report coverage vs accuracy trade-off.
- Additionally compute **hit quality index** (HQI = correlation to best library match); low HQI ⇒ out-of-distribution (e.g. natural particle, dyed or weathered plastic).

## 5. Interpretability (cheap, impressive)
- RF feature importance or SVM/PLS loadings plotted against wavenumber; check that important regions match the peak table in doc 03.

## 6. Finalise
1. Refit on train+val, evaluate once on the test set.
2. `joblib.dump(model, "models/ftir_clf.pkl")` with preprocessing params and class list alongside.
3. `predict.py`: `predict(spectrum_file) -> {polymer, probs, confidence, hqi}`.
4. Repeat for Raman if you have data.

## Done when
Test macro-F1 reported with CI, confusion matrix saved to `outputs/`, calibrated probabilities available, model loads and predicts from a raw file in one call.

## Pitfalls
- Leakage through duplicates/sources → inflated 99% accuracy. Be sceptical of perfect scores.
- Don't tune on the test set.
- Library spectra are clean; real environmental particles are weathered, dirty, thin. Say so.
