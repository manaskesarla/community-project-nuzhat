# Microplastics Screening Pipeline — Polymer Identification from FTIR Spectra

> **Who this is for:** biotech / life-science students and researchers who know some chemistry and lab work
> but little machine learning (ML). Every technical term is explained when it first appears; a glossary is at the end.
>
> **Status (read this first):** the **spectral (FTIR) polymer classifier is built, tested and working**.
> The **image-analysis module** (counting/sizing particles on filter images) and the **combined sample report** are
> **planned but not built yet** (see [Project status](#8-project-status-what-is-done-and-what-is-not)).

---

## Table of contents
1. [The problem: why microplastics need to be identified](#1-the-problem)
2. [The idea in one page](#2-the-idea-in-one-page)
3. [Background: FTIR spectroscopy in 5 minutes](#3-background-ftir-in-5-minutes)
4. [The data](#4-the-data)
5. [How the pipeline works, step by step](#5-how-the-pipeline-works)
6. [Results — and how to read them honestly](#6-results)
7. [Limitations (please read)](#7-limitations)
8. [Project status](#8-project-status-what-is-done-and-what-is-not)
9. [How to run it](#9-how-to-run-it)
10. [Repository map](#10-repository-map)
11. [Glossary](#11-glossary)
12. [Data provenance, ethics and references](#12-data-provenance-ethics-and-references)

---

## 1. The problem
**Microplastics** are plastic fragments and fibres smaller than 5 mm. They are found in drinking water, rivers, oceans,
sediments, food and living tissue. They matter to biology and health because:
* they carry **adsorbed pollutants and microbes** and can be ingested by organisms from plankton to humans;
* **different polymers behave differently** (density, degradation, toxicity of additives), so *which* plastic a particle
  is matters as much as *how many* there are.

Standard practice: filter a water sample, look at the filter under a microscope, then identify each suspicious particle
with a **spectroscopic** method (FTIR or Raman). Identification is slow and needs an expert comparing each spectrum
against reference libraries. **This project automates that step with machine learning**, and (planned) automates
counting/sizing particles from the filter image.

**Relevant UN Sustainable Development Goals:** SDG 14 (Life Below Water, target 14.1 marine pollution),
SDG 6 (Clean Water), SDG 3 (Health), SDG 12 (Responsible Consumption).

## 2. The idea in one page
```
 particle ──► FTIR instrument ──► spectrum (a curve: absorbance vs wavenumber)
                                        │
                         clean-up ("preprocessing")
                                        │
                      trained classifier (ML model)
                                        │
        polymer name  +  confidence score   (e.g. "PET, 93 % sure")
```
* A **spectrum** is a chemical fingerprint. Each polymer has characteristic peaks.
* A **classifier** is a program that learns, from thousands of labelled example spectra, which fingerprint belongs to
  which polymer — then labels new spectra it has never seen.
* Output is **a polymer class plus a confidence value**, and the model can say "not sure" when the confidence is low.

## 3. Background: FTIR in 5 minutes
**FTIR** (Fourier-transform infrared spectroscopy) shines infrared light through or onto a sample. Chemical bonds absorb
specific infrared frequencies (they vibrate: stretch, bend). The result is a plot:

* **x-axis: wavenumber (cm⁻¹)** — a measure of infrared frequency. Higher = higher energy. Data here: ~650–3900 cm⁻¹.
* **y-axis: absorbance** — how much light was absorbed. Peaks = bonds that absorb.

**Measurement modes present in this dataset** (they change the spectrum's shape, so the model must cope with all three):

| Mode | How it works | Typical issue |
|---|---|---|
| **ATR** (attenuated total reflectance) | Particle pressed onto a crystal; measures the surface layer | Depends on contact; intensities differ at low wavenumbers; surface contamination / stickers / weathering show up |
| **Transmission** | IR passes through a thin particle | Needs thin samples; saturates if too thick |
| **Reflectance** | IR bounces off the particle | Distorted peak shapes (derivative-like) |

**Characteristic bands used as a sanity check** (peak positions in cm⁻¹):

| Polymer | Key bands | What it is |
|---|---|---|
| **PE** polyethylene | 2915, 2848, 1470, 720 | C–H stretching/bending in –CH₂– chains |
| **PP** polypropylene | 2950, 1455, 1376, 972 | adds the CH₃ "umbrella" band at 1376 |
| **PS** polystyrene | 3025, 1601, 1492, 756, 697 | aromatic ring bands |
| **PET** polyethylene terephthalate | 1715 (C=O), 1240, 1095, 725 | ester group |
| **PVC** polyvinyl chloride | 1426, 1330, 1250, 690–610 | C–Cl stretching |
| **PA** polyamide / nylon | 3300 (N–H), 1635 (amide I), 1540 (amide II) | amide linkage |

We verified the data against this table: for example, the mean absorbance at 1715 cm⁻¹ for ATR spectra of PET is
0.355 versus 0.021 for all other materials, and at 1635 cm⁻¹ PA is 0.143 versus 0.031. So the labels and spectra behave
chemically as expected.

## 4. The data
All raw files are in `databasedata/`. **It comes from the Zenodo record
[10.5281/zenodo.10126851](https://zenodo.org/records/10126851)** (*Data and Code for High Throughput FTIR Analysis of Macro
and Microplastics with Plate Readers*, v2, 2023; licence CC BY-NC 4.0 — non-commercial use), distributed as
`databasedata.zip`. See [Section 12](#12-data-provenance-ethics-and-references). The metadata shows it contains spectra
originating from published libraries (e.g. Primpke et al. 2018, FLOPP/FLOPP-e by De Frond et al. 2021, Chabuka et al. 2020)
and measured plastic and non-plastic particles on 8 sample plates.

* **Technique:** FTIR only. **No Raman data. No microscope images.**
* **What one "sample unit" is:** a physical particle (identified by `Plate_Cell`, e.g. `6_F12`) measured at 1–3 spots in
  1–3 modes. So many spectra come from the *same* particle.
* **Labels:** an expert-assigned material name (`SpectrumIdentity`, ~225 free-text variants such as "PP", "polypropylene",
  "fibre_polypropylene"). We map them to **8 classes** in `src/spectra/labels.py`:

| Class | Meaning |
|---|---|
| PET, PE, PP, PS, PVC, PA | the six target polymers |
| OTHER_PLASTIC | any other synthetic polymer: rubbers, EVA, ABS, SAN, PBT, PLA, polycarbonate, epoxy, paints… |
| NON_PLASTIC | natural material: cotton, wool, cellulose, coal, amber, quartz, chitin… |

  Why include the last two? Real filters contain many non-target particles. A classifier that only knows six
  plastics would force every particle into one of them (a false positive). Having "other" classes lets it say
  "this isn't one of the six".
* **Size after cleaning:** 6,256 spectra (279 unlabeled dropped; 51 duplicate/flat/invalid removed) on a grid of
  1,616 wavenumber points (660–3890 cm⁻¹, every 2 cm⁻¹).

| Class | Spectra | Distinct particles |
|---|---:|---:|
| PE | 1,545 | 180 |
| OTHER_PLASTIC | 1,413 | 166 |
| PP | 999 | 115 |
| PET | 858 | 84 |
| NON_PLASTIC | 641 | 79 |
| PS | 539 | 69 |
| PA | 199 | 23 |
| **PVC** | **62** | **7** |

  **Important:** the number of *particles* matters more than the number of spectra (repeat measurements of one particle
  are not independent). PVC (7 particles) and PA (23) are thin — see [Limitations](#7-limitations).

## 5. How the pipeline works
### Step A — Build one clean dataset (`python -m src.spectra.load`)
1. Read the metadata and the spectra table.
2. Map messy labels → 8 classes.
3. **Resample** all spectra to the same wavenumber grid (instruments record at slightly different points; ML needs
   every spectrum to have the same length).
4. **Remove duplicates** (the same spectrum appearing twice would let the model "cheat" by memorising it).
5. Save `X_ftir.npy` (the numbers), `meta_ftir.csv` (labels, mode, particle id, …) and a class-count table.

### Step B — Preprocess each spectrum (`src/spectra/preprocess.py`)
Raw spectra differ for reasons unrelated to chemistry (how much sample, instrument drift, noise). Preprocessing removes
this so the model compares *shape*, not *size*:

| Operation | Plain-language purpose |
|---|---|
| **Clip saturation** (absorbance > 3) | Very strong peaks are non-linear/detector-saturated (some spectra hit 6.0) |
| **Savitzky–Golay smoothing** | Reduces random noise without flattening peaks |
| **Baseline correction (AsLS)** | Removes the sloping/curved background from scattering or contact effects |
| **First derivative** | Sharpens overlapping peaks and removes leftover baseline offset |
| **SNV normalisation** (subtract mean, divide by standard deviation, per spectrum) | Makes a thick and a thin particle comparable |

### Step C — Train the classifier
* **Model: SVM** (support vector machine) after **PCA** (principal component analysis).
  * *PCA* compresses 1,616 numbers per spectrum into 120 summary numbers capturing most variation (like a few "master
    peaks" that combine many wavenumbers).
  * *SVM* learns boundaries in that compressed space separating the classes.
  * `class_weight="balanced"` stops the big classes (PE) from drowning out the small ones (PVC).
* **Other models compared:** Random Forest, k-nearest-neighbours with correlation distance (the "library matching" idea
  spectroscopists use), and logistic regression. They scored similarly (cross-validation accuracy 0.80–0.85); the SVM was
  chosen on cross-validation, not on the test set.

### Step D — Validate *without fooling ourselves*  (the most important design choice)
Beginners often split spectra randomly into train/test. Here that would be **cheating**: the same particle measured at 3
spots in 3 modes would appear on both sides, and the model would simply recognise the particle.

We therefore use **group-aware splitting**:
* **Hold-out test set:** 20% of *particles* (145 particles / 1,247 spectra) set aside, never used for choosing or tuning
  anything.
* **Cross-validation (CV):** the remaining 80% is split into 5 folds *by particle*; the model is trained on 4 folds and
  tested on the 5th, rotating. This estimates performance and helps choose settings.
* The final model is scored **once** on the hold-out test set.

### Step E — Confidence (calibration)
An SVM's raw scores are not probabilities. We **calibrate** them (sigmoid / Platt scaling, fitted with particle-grouped
folds) so "70 % confident" means roughly "right about 70 % of the time". A **threshold** (e.g. ≥ 0.7) lets the system
abstain ("unidentified") instead of guessing.

## 6. Results
### 6.1 Final model on the held-out test particles (1,247 spectra, 145 particles)
**Overall accuracy 87.7 %, macro-F1 0.873.**

| Class | Precision | Recall | F1 | Test spectra |
|---|---:|---:|---:|---:|
| PET | 0.954 | 0.954 | 0.954 | 174 |
| PP | 0.926 | 0.897 | 0.911 | 252 |
| PE | 0.930 | 0.872 | 0.900 | 305 |
| PS | 0.808 | 0.899 | 0.851 | 89 |
| OTHER_PLASTIC | 0.854 | 0.831 | 0.843 | 261 |
| NON_PLASTIC | 0.685 | 0.927 | 0.788 | 96 |
| PA | 0.808 | 0.677 | 0.737 | 62 |
| PVC | 1.000 | 1.000 | 1.000 | **8 (≈ 1 particle — not meaningful)** |

**Confidence vs coverage** (accuracy when we only accept confident predictions):

| Rule | Share of spectra answered | Accuracy of those answers |
|---|---:|---:|
| confidence ≥ 0.5 | 94 % | 90.5 % |
| confidence ≥ 0.7 | 73 % | 95.2 % |

### 6.2 Cross-validation by measurement mode (training portion, grouped 5-fold)
| Mode | Accuracy |
|---|---:|
| Reflectance | 0.887 |
| Transmission | 0.853 |
| **ATR** | **0.678** (weakest; only 1,167 spectra) |

### 6.3 How to read these numbers
* **Accuracy** = fraction correct. **Precision** = of those called "PET", how many truly are. **Recall** = of true PETs,
  how many we found. **F1** = balance of the two. **Macro-F1** averages F1 over classes equally, so small classes count
  as much as big ones (more honest than accuracy when classes are unbalanced).
* **Why test (0.877) is higher than CV (0.85)?** They are different sets of particles; with 145 test particles some
  difference is just chance. Treat ~0.85–0.88 as the realistic range, not a precise value.
* A **confusion matrix** (saved as `outputs/confusion_test.csv`) shows *which* classes get mixed up.
* Per-class cross-validation F1 (training portion): PET 0.93, PE 0.88, PP 0.87, PS 0.81, PA 0.76, **PVC 0.51**.

## 7. Limitations
1. **PVC is essentially not validated.** Only 7 particles exist in the dataset; cross-validation recall for PVC was 0.43.
   The perfect test score is one particle. Do not claim PVC performance.
2. **PA is weak** (23 particles, test recall 0.68).
3. **ATR spectra perform poorly** (0.68 accuracy). If your lab uses ATR, the model needs more ATR training data.
4. **"Other" classes are heterogeneous**, which limits accuracy regardless of data size.
5. **Single-source data.** All spectra come from one family of libraries; a different instrument/lab may shift
   performance (*domain shift*). Library spectra are cleaner than real weathered, dirty, biofouled particles.
6. **FTIR only.** No Raman model.
7. **The data is licensed for non-commercial use only** (CC BY-NC 4.0, Section 12), so this project and anything derived from it must stay non-commercial and cite the source.
8. **This is a research prototype, not a regulatory or diagnostic tool.**

## 8. Project status: what is done and what is not
| Component | Status |
|---|---|
| Environment + repo skeleton | ✅ done |
| Dataset audit + label mapping + unified FTIR dataset | ✅ done |
| Preprocessing | ✅ done |
| Classifier training, group-aware CV, calibration, test evaluation | ✅ done |
| `predict` function for a new raw spectrum file | ⏳ not yet written |
| Interpretability plots (which wavenumbers drive each decision) | ⏳ not yet |
| Extra data to fix PVC / PA / ATR gaps (e.g. the Open Specy library, OSF project `x7dpz`) | ⏳ to download |
| Image module: detect/count/size particles in filter images (needs synthetic images) | ⏳ not started |
| Combined report for one sample (polymers + count + sizes + confidence) | ⏳ not started |
| Written report: SDG justification + community component | ⏳ not started |
| Stretch: noise-robustness tests, 1D-CNN vs classical, dashboard | ⏳ optional |

The detailed step-by-step plan for every module is in `docs/`.

## 9. How to run it
Requirements: Python 3.10+ and the `databasedata/` folder present at the project root. The data is not stored in this
repository (it is ~2 GB); download `databasedata.zip` from
<https://zenodo.org/records/10126851> and unzip it at the project root so that `databasedata/` exists.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m src.spectra.load     # 1. build the clean dataset      -> data/spectra_processed/
python -m src.spectra.train    # 2. compare models with grouped cross-validation (~1–2 min)
python -m src.spectra.final    # 3. fit calibrated final model, score on held-out test -> models/, outputs/
python -m pytest -q tests      # sanity tests (label mapping, dataset shapes)
```
All random choices use seed 42, so results reproduce. `config.yaml` holds the class list and wavenumber grid.
Note: `data/`, `models/` and `outputs/` are not committed to git — they are regenerated by the commands above.

## 10. Repository map
```
databasedata/            raw open data (spectra + metadata)                    [input, untouched]
docs/                    step-by-step guides, one file per module (00–10)
src/spectra/
   labels.py             messy label text -> 8 classes
   load.py               build unified dataset
   preprocess.py         smoothing, baseline, derivative, normalisation
   train.py              model comparison with grouped CV
   final.py              final calibrated model + test evaluation
tests/                   automated checks
data/spectra_processed/  generated: spectra matrix, metadata, class table, split
models/ftir_svm.pkl      generated: trained calibrated model
outputs/                 generated: cv_results.json, confusion_test.csv
config.yaml              seed, classes, wavenumber grid
data/DATA_SOURCES.md     provenance log
```

## 11. Glossary
| Term | Meaning |
|---|---|
| **Absorbance** | How strongly the sample absorbs IR light at a given wavenumber |
| **ATR** | FTIR mode measuring the surface by pressing the sample on a crystal |
| **Baseline** | The slowly varying background under the peaks that is not chemical signal |
| **Calibration (of probabilities)** | Adjusting a model's scores so stated confidence matches real accuracy |
| **Classifier** | A model that assigns a category (here a polymer) to an input (a spectrum) |
| **Confusion matrix** | Table of true class vs predicted class |
| **Cross-validation** | Repeatedly training on part of the data and testing on the rest to estimate performance |
| **Data leakage** | Information from the test data accidentally reaching training, giving falsely high scores |
| **Domain shift** | Performance drop when new data differ from training data (new instrument, lab, weathering) |
| **F1 / macro-F1** | Combined precision/recall score; macro averages across classes equally |
| **FTIR** | Fourier-transform infrared spectroscopy |
| **Group-aware split** | Keeping all measurements of one particle on the same side of a train/test split |
| **Hold-out test set** | Data kept untouched until the final evaluation |
| **PCA** | Compresses many correlated measurements into a few summary components |
| **Precision / Recall** | Of predicted X, fraction correct / of true X, fraction found |
| **SNV** | Standard normal variate: normalise each spectrum to mean 0, spread 1 |
| **SVM** | Support vector machine, a classifier that finds the best separating boundary |
| **Wavenumber (cm⁻¹)** | Spectroscopic x-axis unit, proportional to IR frequency |
| **PE / PP / PS / PET / PVC / PA** | Polyethylene / polypropylene / polystyrene / polyethylene terephthalate / polyvinyl chloride / polyamide (nylon) |

## 12. Data provenance, ethics and references
* **Source of `databasedata/`:** Zenodo record [10.5281/zenodo.10126851](https://zenodo.org/records/10126851) (v2,
  published 2023-11-14; file `databasedata.zip`, ~1.1 GB; earlier v1 is [10.5281/zenodo.7772572](https://zenodo.org/records/7772572)).
  It accompanies Cowger W. et al. (2024) *Generation of macro- and microplastic databases by high-throughput FTIR analysis
  with microplate readers*, Analytical and Bioanalytical Chemistry 416(6), 1311–1320, DOI 10.1007/s00216-024-05127-w.
  Component libraries are listed in the `Organization` column of the metadata.
* **Licence:** CC BY-NC 4.0 (non-commercial). Cite the Zenodo record and the paper above when using the data.
* The dataset is not committed to this repository (too large for GitHub; `databasedata/` is in `.gitignore`).
* **Ethics:** no human or personal data are involved. Any future community component (surveys, interviews, sampling) needs
  consent and anonymisation.
* **Honesty statement:** results here are from a prototype on library-style spectra; they are not a validated field
  measurement. The planned image module will be a proof of concept using clearly labelled synthetic images.
* **Key references:**
  * Cowger W. et al. (2024) *Generation of macro- and microplastic databases by high-throughput FTIR analysis with microplate readers.* Analytical and Bioanalytical Chemistry 416(6), 1311–1320. DOI 10.1007/s00216-024-05127-w. Data: DOI 10.5281/zenodo.10126851
  * Cowger W. et al. (2021) *Microplastic Spectral Classification Needs an Open Source Community: Open Specy to the Rescue!* Analytical Chemistry. DOI 10.1021/acs.analchem.1c00123
  * De Frond H., Rubinovitz R., Rochman C. (2021) *µATR-FTIR Spectral Libraries of Plastic Particles (FLOPP and FLOPP-e) for the Analysis of Microplastics.* Analytical Chemistry 93(48). DOI 10.1021/acs.analchem.1c02549
  * Primpke S. et al. (2018) FTIR spectral library for microplastic identification (cited in dataset metadata).
