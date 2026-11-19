# Microplastics Screening Pipeline — Polymer Identification from FTIR Spectra

> **Who this is for:** biotech / life-science students and researchers who know some chemistry and lab work
> but little machine learning (ML). Every technical term is explained when it first appears; a glossary is at the end.
>
> **Status (read this first):** the **spectral (FTIR) polymer classifier is built, tested and working**
> (87.7 % accuracy on held-out particles). The **image module** (count/size particles) and the **combined sample
> report** are built too, but the image module has been validated **on synthetic images only** — it is a proof of
> concept, not a validated measurement. A written-report draft exists; the community component is still to do.
> The complete story of what was done and why is in [Section 13](#13-project-log-everything-that-has-been-done-step-by-step).

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
13. [Project log: everything done, step by step](#13-project-log-everything-that-has-been-done-step-by-step)

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
| `predict` function for a new raw spectrum file (`src/spectra/predict.py`) | ✅ done |
| Interpretability plots (`src/spectra/interpret.py` → `outputs/interpretability.png`) | ✅ done |
| Extra data to fix PVC / PA / ATR gaps (e.g. the Open Specy library, OSF project `x7dpz`) | ⏳ to download |
| Image module: synthetic images + segment/measure/validate (`src/images/`) — **synthetic-data proof of concept only**, no real images yet | ✅ done (PoC) |
| Combined report for one sample (`python run_sample.py samples/sample_001` → `outputs/sample_001/report.html`) | ✅ done |
| Written report draft (`report/REPORT.md`; community component and some citations left for the team) | 🟡 drafted |
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
python -m src.spectra.predict my_spectrum.csv   # identify one spectrum (2 columns: wavenumber, absorbance)
python -m src.spectra.interpret   # occlusion plot of which wavenumbers drive each class
python -m src.images.synth --n 150      # 150 labelled SYNTHETIC filter images + ground truth -> data/images_synthetic/
python -m src.images.validate           # image-module validation vs ground truth -> outputs/image_validation.*
python -m src.images.measure img.png    # count/size particles in one image
python -m src.report.make_demo          # (re)build the two constructed demo samples in samples/
python run_sample.py samples/sample_001 # combined report -> outputs/sample_001/report.html + result.json
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
   predict.py            identify one raw spectrum file
   interpret.py          occlusion plot: which wavenumbers drive each class
src/images/
   synth.py              synthetic filter images + ground truth (all labelled SYNTHETIC)
   segment.py            find particles (background, hysteresis threshold, watershed)
   measure.py            count, sizes in µm, size bins, image confidence, overlay
   validate.py           image module vs ground truth (dev / held-out)
src/report/
   combine.py            image + spectra -> result.json + report.html
   make_demo.py          builds the two constructed demo samples
   template.html        report template
run_sample.py            one-command combined report for a sample folder
samples/                 sample_001, sample_002 (constructed demos: synthetic image + test spectra)
report/REPORT.md         written-report draft (+ figures/)
tests/                   automated checks (11)
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
| **Feret diameter** | The longest distance across a particle outline (its maximum extent) |
| **Ground truth** | The known correct answer, used to score a method (for synthetic images we know it exactly) |
| **Hysteresis threshold** | Keep a region only if it has at least one strong pixel, but grow it through weaker ones; rejects isolated noise |
| **IoU (intersection over union)** | Overlap between a predicted and a true outline; 1 = identical, 0 = none |
| **Occlusion analysis** | Blank part of the input and see how much the model's answer changes, to find what it relies on |
| **Synthetic data** | Computer-generated data with known answers; useful for testing, not proof of real-world accuracy |
| **Watershed** | Algorithm that splits touching blobs along the "valleys" between them |
| **Wilson interval** | A 95 % confidence range for a proportion, reliable even for small samples |
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

## 13. Project log: everything that has been done, step by step
This section is the full story of the project in the order it happened, written so that someone new can follow *what* was
done, *why*, *how*, and *what came out*. File names are given so every claim can be traced to code. Git commit ids are
from `git log`.

### Overview in one table
| # | Step | Main files | Result |
|---|---|---|---|
| 0 | Plan and project skeleton | `docs/`, `config.yaml`, `requirements*.txt` | Repo, plan documents, environment |
| 1 | Understand the dataset | `databasedata/` | Know what a "spectrum", "particle" and "mode" are |
| 2 | Map labels, build one clean dataset | `labels.py`, `load.py` | 6,256 spectra × 1,616 points, 8 classes |
| 3 | Preprocess spectra | `preprocess.py` | Spectra comparable by shape |
| 4 | Compare models, final calibrated model | `train.py`, `final.py` | 87.7 % accuracy, macro-F1 0.873 on held-out particles |
| 5 | Publish to GitHub | `.gitignore` | Code on GitHub; 2 GB dataset kept out |
| 6 | Dataset lost and restored | — | Dataset restored from Zenodo, verified identical |
| 7 | Document the dataset source | `README.md`, `data/DATA_SOURCES.md` | Zenodo record + CC BY-NC 4.0 licence recorded |
| 8 | `predict` for a new spectrum | `predict.py` | Raw file in → polymer + confidence out |
| 9 | Interpretability | `interpret.py` | Plot of which wavenumbers drive each class |
| 10 | Synthetic filter images with ground truth | `synth.py` | 150 labelled images |
| 11 | Segmentation (find particles) | `segment.py` | First attempt failed; fixed in two stages |
| 12 | Measurement and image confidence | `measure.py` | Count, sizes (µm), size bins, Low/Medium/High |
| 13 | Validate the image module | `validate.py` | Errors reported on held-out synthetic images |
| 14 | Combined sample report | `combine.py`, `run_sample.py`, `make_demo.py` | HTML + JSON report for a sample |
| 15 | Automated tests | `tests/` | 11 tests pass |
| 16 | Written report draft | `report/REPORT.md` | Draft; community part left to the team |

### Step 0 — Plan and project skeleton (commit `457748f`)
* **What:** wrote a step-by-step plan, one document per module, in `docs/00`–`docs/10` (overview, environment, spectral
  data, preprocessing, classifier, image data, image module, combined report, written report/SDG, stretch goals,
  3-week timeline). Created `config.yaml` (random seed 42, the six target classes, the wavenumber grid), package folders
  (`src/spectra`, `src/images`, `src/report`), `requirements.txt` plus a pinned `requirements.lock.txt`, `.gitignore`,
  and an `.mcp.json` that configures a Colab connector for the coding assistant.
* **Why:** the project has three modules (spectra, images, report) that are easier to build and check one at a time.
  A fixed seed makes every result reproducible.
* **Design decision recorded in the plan:** the spectra and the image are linked **by sample, not by particle** (linking a
  specific particle in the image to its spectrum needs µ-FTIR mapping, which is out of scope).

### Step 1 — Understand the dataset
* **What:** inspected the files in `databasedata/`. The key ones: `Metadata_raw.csv` (one row per spectrum: label, plate,
  cell, measurement mode, source library), `spread_os.csv` (the spectra themselves, one column per spectrum, wavenumber
  down the rows), `CleanedSpectra_raw.csv` (the same data in long format) and folders `ATR/`, `Reflectance/`,
  `Transmission/` (raw instrument exports).
* **Findings that shaped everything after:**
  1. A "sample unit" is a **physical particle** (plate + cell, e.g. `6_F12`), measured at several spots in up to three
     modes, so many spectra belong to the *same* particle. (This is why random train/test splits would cheat — see Step 4.)
  2. Labels are ~225 free-text variants ("PP", "polypropylene", "fibre_polypropylene", …).
  3. Only FTIR is present: no Raman, no microscope images. So the image module needs synthetic images (Step 10).
  4. Only 7 particles are PVC and 23 are PA — too few to claim reliable results for them.

### Step 2 — Map labels and build one clean dataset (commit `72d7b41`)
* **What:** `src/spectra/labels.py` turns the free-text labels into **8 classes**: PET, PE, PP, PS, PVC, PA,
  OTHER_PLASTIC, NON_PLASTIC. It works with ordered regular-expression rules; copolymers and blends (SAN, ABS, EVA, PBT,
  polyurethane, …) are matched *first* so they are not wrongly swallowed by a base polymer. Anything unresolvable
  ("Unknown", empty) is dropped. `src/spectra/load.py` then builds the dataset.
* **Why the two extra classes:** real filters hold many non-target particles. A model that knows only six plastics would
  force every particle into one of them, creating false positives.
* **How `load.py` works:** read the metadata → map labels → drop unlabelled rows (279) → define the particle id
  `group = Plate_Cell` → read the spectra table, sort by wavenumber → **resample every spectrum to one grid** (660–3890
  cm⁻¹ in steps of 2 → 1,616 points) so all spectra have the same length → remove exact duplicates and flat/invalid
  spectra (51) → save `X_ftir.npy`, `meta_ftir.csv`, `class_table.csv`.
* **Result:** 6,256 spectra × 1,616 points (table in Section 4). `tests/test_labels.py` checks the label rules.

### Step 3 — Preprocess spectra (commit `013ce53`)
* **What:** `src/spectra/preprocess.py` applies, in order: clip saturated absorbance (> 3), Savitzky–Golay smoothing,
  asymmetric-least-squares (AsLS) baseline removal, first derivative, SNV normalisation.
* **Why:** two spectra of the same polymer can differ because of sample thickness, a sloping background or noise. These
  steps remove such differences so the model compares *shape*. The same function is used for training and for new
  spectra, otherwise predictions would not match training.

### Step 4 — Compare models and fit the final calibrated model (commits `013ce53`, `b84fa5c`)
* **What:** `train.py` compares four classifiers — PCA + SVM, random forest, correlation-distance kNN (the classic
  "library matching" idea), logistic regression — using **particle-grouped cross-validation**. `final.py` fits the
  chosen model (PCA 120 components + SVM, `class_weight="balanced"`) wrapped in **sigmoid probability calibration**, and
  scores it once on the held-out test particles.
* **Why grouped:** all spectra of one particle stay on the same side of every split. Otherwise the model could recognise
  a particle it already saw (data leakage) and the score would be falsely high. 20 % of particles (145; 1,247 spectra)
  were locked away as the test set and never used to choose anything.
* **Results:** CV accuracy — random forest 0.847, PCA + SVM 0.838, kNN 0.821, logistic regression 0.801. Final
  calibrated SVM on the test set: **accuracy 0.877, macro-F1 0.873**; when it answers only if confidence ≥ 0.7 it answers
  73 % of spectra and is right 95.2 % of the time. Weak spots: ATR spectra (0.68 CV accuracy), PA (F1 0.74), PVC (only
  one test particle, so its 1.0 is meaningless). Details in Section 6.
* **Output:** `models/ftir_svm.pkl`, `outputs/cv_results.json`, `outputs/confusion_test.csv`.

### Step 5 — Publish to GitHub
* **What:** connected the local repository to `github.com/manaskesarla/community-project-nuzhat` and pushed.
* **Problems met, and how each was handled:**
  1. The remote already held a GitHub-generated "Initial commit" containing only a one-line README, so the first push
     was rejected. the local commits were **rebased** on top of it (no history was overwritten). The two README files
     conflicted; the project README was kept.
  2. The raw dataset was 2 GB and had been committed. GitHub refuses single files over 100 MB, and two files were
     larger (`CleanedSpectra_raw.csv` 843 MB, `spread_os.csv` 110 MB). On request, the dataset was removed from the
     local, not-yet-pushed commits, `databasedata/` was added to `.gitignore`, and only then pushed.
* **Result:** the repository on GitHub contains code, docs, tests and small demo files — never the dataset or any
  co-author line.

### Step 6 — The dataset was lost and restored (an honest account)
* **What went wrong:** removing the dataset from git history was done with `git filter-branch`. When it finished, git also
  **deleted the now-untracked dataset files from the working folder**, and git's old objects were then cleaned up, so no copy
  remained on the computer (no Trash entry, no Time Machine backup). This was a mistake by the coding assistant that did the cleanup; the data should have been
  copied somewhere safe first.
* **Recovery:** the dataset was found again as `databasedata.zip` on Zenodo, downloaded, unzipped and copied into the
  project. Checks done: the zip's md5 (`72fdd6f281fe3f515a221b38ee80d06b`) equals the value Zenodo lists for version 2,
  and **re-running `load.py` rebuilt a dataset identical to the earlier one** (6,256 × 1,616, arrays equal). The zip was
  then deleted at the user's request, leaving one copy in the project.
* **Lesson recorded:** before any operation that rewrites git history, copy untracked-after-the-change files elsewhere.

### Step 7 — Document where the data comes from
* **What:** the README and `data/DATA_SOURCES.md` had said the source was "unknown". Once found, they were updated: Zenodo
  record 10.5281/zenodo.10126851 (v2, 2023-11-14), paper Cowger et al. 2024 (DOI 10.1007/s00216-024-05127-w),
  **licence CC BY-NC 4.0 (non-commercial)**. Limitations were updated accordingly: the project must stay non-commercial
  and must cite the source. `data/DATA_SOURCES.md` lives in the git-ignored `data/` folder, so it is local only.

### Step 8 — `predict`: identify a new spectrum (`src/spectra/predict.py`)
* **What:** `python -m src.spectra.predict file.csv` reads a two-column file (wavenumber, absorbance; header optional;
  comma/tab/semicolon/space separated), puts it on the common grid, applies the **same** preprocessing, and returns the
  polymer, the calibrated confidence and the probability of every class. Below confidence 0.5 it answers `UNCERTAIN`
  and shows the best guess instead of guessing silently.
* **Checks built in:** it refuses files with fewer than 20 numeric rows or that do not cover about 660–3890 cm⁻¹.
  Transmittance must be converted to absorbance first.
* **Check:** 40 held-out spectra were written to files and predicted through this path: 37/40 correct.
  `tests/test_predict.py` covers parsing, bad input and range checks.

### Step 9 — Interpretability: why does the model decide as it does? (`src/spectra/interpret.py`)
* **What:** **occlusion analysis**. For each 100 cm⁻¹ window (stride 50) the preprocessed spectrum is set to zero there
  and we measure how much the probability of the *true* class falls, averaged over the held-out test spectra of that
  class. A big drop means the model relies on that region. (The model is an RBF-kernel SVM, so it has no simple
  coefficients to read; occlusion works for any model.)
* **Result:** `outputs/interpretability.png`. Peaks fall where chemistry predicts: **PA ≈ 1650 cm⁻¹** (amide I),
  **PET ≈ 1720 cm⁻¹** (ester C=O), **PVC ≈ 800–850 cm⁻¹** (C–Cl region), PE/PS also at ≈ 700–760 cm⁻¹.
* **Caution recorded in the report:** several classes lean on the lowest-wavenumber edge (≈ 700 cm⁻¹); that may be real
  bands or an edge/baseline artefact. PP importance is weak and noisy.

### Step 10 — Synthetic filter images with ground truth (`src/images/synth.py`)
* **Why synthetic:** the dataset has no images and open, licence-clear images of filtered microplastics are scarce. To
  test an image-analysis method we need images where the **true answer is known**, so we generate them.
* **How:** each 1024 × 1024 image imitates a brightfield filter membrane: pale background with uneven illumination,
  vignetting and faint pore texture; N random particles (10–500) of three kinds — **fragments** (random polygons),
  **fibres** (random-walk lines), **films** (smooth blobs) — with sizes from a **log-normal** distribution (many small,
  few large; 8–600 µm), seven realistic colours, then defocus blur, sensor noise, and **distractors** (dust, scratches)
  that must *not* be counted. Three difficulty levels (easy/medium/hard, 50 images each). Scale: **1 pixel = 2 µm**
  (`config.yaml`).
* **Ground truth saved per image:** `SYNTH_0001.json` (count, each particle's area, equivalent diameter, Feret
  diameter) and `SYNTH_0001_mask.png` (which pixel belongs to which particle; overlaps are resolved so only the visible
  part counts).
* **Labelled as synthetic everywhere:** filename prefix `SYNTH_`, a "SYNTHETIC" tag drawn in the image corner,
  `is_synthetic = true` in the JSON, a `README.txt` in the folder.

### Step 11 — Segmentation: finding the particles (`src/images/segment.py`)
Pipeline: estimate and subtract the background → measure each pixel's colour distance from the background → threshold →
clean up → split touching particles → label each particle.

* **First attempt failed, openly:** a simple "median + 5 noise-sd" threshold gave **267 particles on an image that really
  had 22** (easy image) and sizes about 10 µm too large. Diagnosis (by viewing the difference map): the membrane's pore
  texture makes bright speckles that exceed the threshold, and the blurred particle edges make particles look bigger.
* **Fix 1 – hysteresis thresholding:** keep a region only if it contains at least one *strong* pixel (above a high
  threshold) but grow it while pixels are above a lower one. Isolated speckles never reach the high threshold, real
  particles do.
* **Fix 2 – edge refinement:** re-threshold each particle at **50 % of its own contrast**, which places the edge where
  the blur has dropped to half, removing the size bias.
* **Also:** hole filling; minimum area 15 px (60 µm²) to ignore dust; **watershed splitting** of touching compact blobs
  (not applied to long fibres, which it would wrongly cut).
* **Honest tuning protocol:** parameters were adjusted using images 1–30 only ("dev"); images 31–150 ("held-out") were
  kept for reporting.

### Step 12 — Measurement and image confidence (`src/images/measure.py`)
* **Per particle:** area, equivalent diameter, maximum Feret diameter (the longest extent), major/minor axis, aspect
  ratio, solidity, and a shape class (fibre / fragment / film). Output also includes the size distribution in bins
  (<20, 20–100, 100–500, 500–1000, >1000 µm), D10/D50/D90, and an overlay image with each outline drawn.
* **Image confidence (Low / Medium / High), rule written in the code:** a weighted score of (a) fraction of particles
  near the resolution floor, (b) fraction ambiguous (touching the border, split by watershed, irregular), (c) noise
  penalty, (d) **crowding** penalty (fraction of the image covered).
* **Why crowding was added:** the first version called 117 of 120 held-out images "High" even though crowded images
  were being undercounted by ~30 %. After adding the crowding term, held-out "High" images had **14.6 %** mean count
  error and "Medium" **27.0 %**, so the label now means something.

### Step 13 — Validate the image module (`src/images/validate.py`)
Compares predictions with the ground truth on the held-out images (matching a predicted particle to a true one when
their overlap, IoU, is at least 0.5). Results (**synthetic images only**):

| Subset | Count error (MAPE) | Count correlation r | Precision | Recall | F1 | Size error (equiv. diameter) |
|---|---:|---:|---:|---:|---:|---:|
| All 120 | 19.7 % | 0.97 | 0.78 | 0.72 | 0.74 | 3.0 µm |
| Easy | 14.4 % | 0.96 | 0.83 | 0.87 | 0.84 | 1.1 µm |
| Medium | 14.8 % | 0.92 | 0.74 | 0.74 | 0.73 | 2.3 µm |
| Hard | 29.8 % | 0.89 | 0.78 | 0.54 | 0.64 | 3.7 µm |

* **Where it fails:** crowded images (overlapping particles merge and are counted once: mean −107 particles on hard
  images) and fibres (size bias +4.2 µm versus +1.3 µm for fragments). About 3 % of true particles are below the 15-px
  floor and are deliberately not expected.
* Output: `outputs/image_validation.json`, `…_per_image.csv`, `…png`.
* **This is not evidence about real microscope images.** No real filter images were used.

### Step 14 — Combined sample report (`src/report/combine.py`, `run_sample.py`, `make_demo.py`)
* **What:** one command, `python run_sample.py samples/sample_001`, takes a sample folder (`meta.yaml`, `image.png`,
  `spectra/*.csv`) and writes `outputs/sample_001/report.html`, `result.json`, an annotated image, a size histogram,
  a polymer bar chart with error bars and a confidence histogram.
* **How it combines the two branches:** the image branch gives count, sizes, image confidence; the spectral branch
  predicts a polymer for every spectrum (abstaining below 0.5 → "unidentified"). Polymer proportions get **95 % Wilson
  confidence intervals**. Scaling the proportions up to the whole filter is shown **only with the explicit assumption**
  that the analysed spectra are a random subsample of the particles. Concentration = count ÷ litres filtered.
* **Overall confidence rule (printed in the report):** the **lowest** of the image level, the spectral level (mean
  confidence of identified particles × fraction not abstained: ≥ 0.80 High, ≥ 0.60 Medium) and a sample-size cap
  (< 10 spectra → Low; < 30 → Medium).
* **Always shown in the report:** "Proof of concept — not a validated measurement", a "Synthetic data used" banner when
  relevant, the inputs, the assumptions and an automatically built caveat list.
* **Demo samples:** `samples/sample_001` and `sample_002` are **constructed**: a synthetic image plus 50 spectra drawn
  from the held-out test particles (one spectrum per physical particle) with a chosen composition, stored in
  `demo_truth.json`. The pipeline never reads that file. On `sample_001` the chosen mix PE 40 %, PP 24 %, PET 16 % was
  recovered as PE 36 %, PP 24 %, PET 16 %; overall confidence **Medium**.

### Step 15 — Automated tests (`tests/`)
11 tests, all passing: label mapping; spectrum file parsing and range checks; synthetic generator is deterministic and
labelled synthetic; the segmenter counts an easy image within tolerance and returns zero on a blank image; IoU matching;
an end-to-end sample run checking the JSON keys, that polymer proportions sum to 1, and that the banners are present.
Run with `python -m pytest -q tests`.

### Step 16 — Written report draft (`report/REPORT.md`)
Abstract, problem, SDG justification (SDG 14 primary; 6 and 12 secondary, each with what the tool does *not* do),
methods, results with the numbers above, discussion, limitations and honesty statement, ethics/licences, future work,
references, and reproduction commands. **Left for the team, clearly marked:** the community component (nothing was
carried out, so nothing is claimed) and problem-scale citations (WHO/UNEP) that have not been verified.

### What has NOT been done (so no one over-trusts the results)
* No **real** filter-membrane images were obtained or tested; the image module is a synthetic-data proof of concept.
* No community activity (interviews, sampling kit, workshop) has taken place.
* No Raman model; no extra data to repair the weak PVC, PA and ATR cases (e.g. the Open Specy library).
* The pipeline has not been run on a **field** sample; both demo samples are constructed.
* Stretch goals (dashboard, 1D-CNN comparison, noise-robustness tests) are not started.

### Commit history
| Commit | What it added |
|---|---|
| `0277b3e` | GitHub's automatic "Initial commit" (one-line README) |
| `457748f` | Step 0: skeleton, plan documents, config, environment |
| `72d7b41` | Step 2: label mapping, unified dataset loader, tests |
| `013ce53` | Steps 3–4: preprocessing, cross-validated classifiers, calibrated SVM |
| `b84fa5c` | `final.py` and the detailed README |
| `ed8d283` | `databasedata/` added to `.gitignore` |
| `6ec0746` | README: dataset source and licence |
| `26321fa` | Step 8–9: `predict`, interpretability, tests, README status |
| `3d845b2` | Steps 10–14, 16: image module, combined report, report draft |
