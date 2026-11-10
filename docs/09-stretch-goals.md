# 09 — Stretch Goals (ONLY if ahead of schedule)

Start only when MVP items 1–4 are finished and committed. Do them in the order below (value per hour).

## 1. Noise-robustness tests (≈ 0.5–1 day) — best value
Evaluate the *already trained* classifier under perturbations of the **test** spectra:
- additive Gaussian noise at SNR = 40, 30, 20, 10, 5 dB,
- baseline drift (add polynomial/sloped baseline),
- spectral shift (± 2–8 cm⁻¹), resolution degradation (broaden by convolution),
- intensity scaling / saturation,
- weathering proxy: attenuate/merge peaks, add carbonyl band ~1715 to PE/PP.
Plot **accuracy and macro-F1 vs noise level** per model. For images: add blur, noise, uneven light and plot count error vs severity.
Also try **training with augmentation** (noise/shift/baseline) and show the improvement.

## 2. Classical vs 1D-CNN comparison (≈ 1–2 days)
```python
class Net(nn.Module):
    def __init__(s, n_classes, L):
        super().__init__()
        s.f = nn.Sequential(
            nn.Conv1d(1,16,7,padding=3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16,32,5,padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32,64,3,padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(8))
        s.c = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(64*8,n_classes))
    def forward(s,x): return s.c(s.f(x))
```
- Same group-aware splits and metrics as doc 04; 5 seeds; report mean ± std.
- Use augmentation (noise, shift, scale) — CNNs need it with small data.
- Compare: accuracy/macro-F1, calibration (temperature scaling), robustness curves (from #1), training time, interpretability (saliency / Grad-CAM-1D).
- Honest conclusion: with small data, SVM/PLS-DA often match or beat the CNN — say so if true.

## 3. Simple dashboard (≈ 1 day)
- **Streamlit** (`streamlit run app.py`) or a Jupyter notebook with `ipywidgets`.
- Pages: (a) upload spectrum → predicted polymer + probabilities + matched reference overlay; (b) upload membrane image → overlay, count, size histogram; (c) sample report view/download.
- Show the proof-of-concept / synthetic banners prominently.
- Deploy locally, or on Streamlit Community Cloud / Hugging Face Spaces (no private data).

## 4. Other cheap extras
- Out-of-distribution detection (Mahalanobis distance in PCA space) for natural-particle rejection.
- Ensemble of classical + CNN.
- Export report to PDF automatically.

## Rule
Stretch work must never delay the written report. Freeze features by **Day 17**.
