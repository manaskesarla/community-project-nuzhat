# 03 — Spectral Preprocessing (Days 3–5)

**Output:** fixed-length, comparable spectra ready for ML; `preprocess.py` that works identically at train and inference time.

## Pipeline (apply in this order)
1. **Crop** to the informative region
   - FTIR: 650–4000 cm⁻¹ (optionally mask 2200–2500 CO₂ region)
   - Raman: 200–3200 cm⁻¹
2. **Resample / interpolate** onto a common grid (`np.interp` or `scipy.interpolate.interp1d`) — e.g. step 2 cm⁻¹ → ~1675 points for FTIR.
3. **Convert units consistently**: transmittance → absorbance (`A = 2 - log10(%T)`); ensure all FTIR are absorbance. Do not mix.
4. **Smoothing**: Savitzky–Golay (`scipy.signal.savgol_filter`, window 9–15, poly 2–3).
5. **Baseline correction**: asymmetric least squares (AsLS), rubber-band, or polynomial fit (order 3–5). Essential for Raman fluorescence background.
6. **Normalisation**: min–max to [0,1] or **SNV** (standard normal variate) — pick one and keep it.
7. **Optional derivative**: 1st/2nd Savitzky–Golay derivative reduces baseline effects and sharpens peaks (often boosts classical-ML accuracy).

```python
def preprocess(x, y, grid, window=11, poly=3):
    y = np.interp(grid, x, y)
    y = savgol_filter(y, window, poly)
    y = y - asls_baseline(y)          # implement or use pybaselines
    y = (y - y.mean()) / y.std()      # SNV
    return y
```
`pybaselines` package gives ready AsLS/airPLS.

## Characteristic peaks (use for sanity checks and for the report)
| Polymer | FTIR key bands (cm⁻¹) |
|---|---|
| PE | 2915, 2848 (C–H stretch), 1470, 720 |
| PP | 2950, 2917, 1455, 1376, 972, 841 |
| PS | 3025, 1601, 1492, 756, 697 |
| PET | 1715 (C=O), 1240, 1095, 725 |
| PVC | 2910, 1426, 1330, 1250, 690–610 (C–Cl) |
| PA (nylon) | 3300 (N–H), 1635 (amide I), 1540 (amide II) |

Plot class-mean spectra and verify these peaks appear — it's a free correctness test and a great report figure.

## Steps
1. Implement `preprocess.py` with config-driven parameters.
2. Run on all data → save `X` (n × features), `y`, `source`.
3. **Visual QA**: before/after plots for 3 spectra per class; class-mean overlays.
4. **Unit tests** (`tests/test_preprocess.py`): output length equals grid length; no NaNs; idempotent shape for any input range.
5. Keep **raw → processed** traceable (never overwrite raw).

## Done when
`X.shape == (N, n_points)`, no NaN, class-mean plots show expected peaks.
