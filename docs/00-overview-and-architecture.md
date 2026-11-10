# 00 — Overview & Architecture

## Goal
Build a proof-of-concept **microplastics screening pipeline** that, for one "sample", reports:
polymer types, particle count, size distribution, and a confidence level — plus a written report with SDG justification and a community component.

## Deliverables (Minimum Viable Project)
| # | Deliverable | Module doc |
|---|-------------|-----------|
| 1 | Spectral classifier (PET, PE, PP, PS, PVC, PA) from FTIR/Raman | 02, 03, 04 |
| 2 | Image-analysis module: detect, count, size particles on filter-membrane images | 05, 06 |
| 3 | Combined report: polymers, counts, size distribution, confidence for one sample | 07 |
| 4 | Written report: SDG justification + community component | 08 |
| Stretch | Dashboard, classical vs 1D-CNN comparison, noise robustness | 09 |

## Architecture
```
 spectra (FTIR/Raman) ──► preprocess ──► classifier ──► polymer probs ─┐
                                                                        ├─► sample report
 membrane image ──► segmentation ──► particles (count, size µm) ───────┘     (JSON + PDF/HTML)
```
The two branches are independent. Link them at the report level: the sample's polymer mix comes from spectra; counts/sizes come from the image. Be honest that the link is by *sample ID*, not per-particle (per-particle linking needs µ-FTIR mapping, which is out of scope).

## Repo layout
```
project/
├── docs/                  # these guides
├── data/
│   ├── spectra_raw/  spectra_processed/
│   ├── images_real/  images_synthetic/
├── notebooks/             # 01_spectra.ipynb, 02_images.ipynb, 03_report.ipynb
├── src/
│   ├── spectra/ (load.py, preprocess.py, train.py, predict.py)
│   ├── images/  (segment.py, measure.py, synth.py)
│   └── report/  (combine.py, templates/)
├── models/  outputs/  tests/
├── requirements.txt
└── README.md
```

## Key design principles
1. **Honesty over polish** — label synthetic data, state limitations, never claim validated measurement for the image module.
2. **Split by source, not by spectrum** — avoid leakage (see 04).
3. **Reproducible** — fixed seeds, pinned requirements, one `make all` / `run_all.py`.
4. **Confidence is a first-class output** — every result carries a calibrated confidence.

## Read order
01 → 02 → 03 → 04 (spectra track) · 05 → 06 (image track, can run in parallel with a second teammate) → 07 → 08 → 09 (only if ahead) → 10 (schedule).
