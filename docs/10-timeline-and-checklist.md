# 10 — 3-Week Timeline & Final Checklist

## Week 1 — Data + foundations
| Day | Spectra track (A) | Image track (B) | Report (C) |
|---|---|---|---|
| 1 | Env setup (01), find datasets | Env setup, find image sources | SDG literature, outline |
| 2–3 | Download + loaders (02) | Build synthetic generator (05) | Pick community activity, plan |
| 4–5 | Preprocessing (03) | Generate synthetic set + GT | Draft Intro + SDG |
| 6–7 | Baseline models (04) | Start segmentation (06) | Start community activity |
**Milestone (end W1):** clean spectra dataset + first baseline accuracy; synthetic images with ground truth.

## Week 2 — Build the modules
| Day | Spectra | Image | Report |
|---|---|---|---|
| 8–9 | Tune, group-CV, calibration | Watershed, measurements | Draft Methods |
| 10–11 | Final model, confusion matrix, predict.py | Validation vs GT, confidence | Community results |
| 12–14 | **Combined pipeline (07)** together | | Report template |
**Milestone (end W2):** MVP items 1–3 working end-to-end on `sample_001`.

## Week 3 — Report + polish (+ stretch if ahead)
| Day | Task |
|---|---|
| 15–16 | Generate final figures/tables, second demo sample, tests |
| 17 | **Feature freeze.** Stretch goals (09) only if MVP is done |
| 18–19 | Write Results, Discussion, Limitations, Future work |
| 20 | Internal review, fix numbers/figures, README, reproducibility run from a clean clone |
| 21 | Buffer, export PDF, submit/present |

## Final checklist
**Spectral classifier**
- [ ] PET, PE, PP, PS, PVC, PA all covered; class counts documented
- [ ] Group-aware split, test set untouched until the end
- [ ] Accuracy + macro-F1 + confusion matrix + calibration
- [ ] Model saved, `predict.py` works on a raw file

**Image module**
- [ ] Detects, counts, sizes (µm) particles; overlay output
- [ ] Synthetic set clearly labelled; real images documented
- [ ] Validation vs ground truth (count & size error)
- [ ] Stated as proof of concept

**Combined report**
- [ ] One command → `result.json` + `report.html`
- [ ] Polymers, count, size distribution, confidence level
- [ ] Caveat banners auto-included

**Written report**
- [ ] SDG justification with targets and honest scope
- [ ] Community component documented (consent, anonymised)
- [ ] Limitations + data licences table
- [ ] All numbers reproducible from code

**Repo**
- [ ] README with setup + run steps, `requirements.txt`, seeds fixed
- [ ] `DATA_SOURCES.md` complete

## Risk register
| Risk | Mitigation |
|---|---|
| Too few spectra for a class | Merge sources, augmentation, report honestly |
| No open real images | Synthetic set + 5–10 hand-annotated real images |
| Inflated accuracy (leakage) | Group splits, de-duplicate |
| Schedule slip | Cut stretch goals first, then Raman (keep FTIR only) |
| Community activity falls through | Fallback: stakeholder interviews or open-source documentation |
