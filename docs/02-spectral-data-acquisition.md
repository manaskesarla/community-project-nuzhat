# 02 — Spectral Data Acquisition (Days 1–3)

**Output:** a single tidy dataset of labelled spectra for 6 classes, with provenance.

## Where to look (verify each licence before use)
- **Open Specy** (openspecy.org) — open FTIR + Raman reference library for plastics and natural materials; downloadable library files.
- **SLoPP / SLoPP-E** (Raman library from the Rochman lab, University of Toronto) — microplastics Raman spectra.
- **FLOPP / FLOPP-e** (FTIR libraries for plastics, environmental weathered variants).
- **Zenodo / Figshare / Mendeley Data** — search "microplastic FTIR spectra", "polymer Raman dataset", "ATR-FTIR plastics".
- **NIST Chemistry WebBook**, **RRUFF** (minerals only — useful as *negative / non-plastic* examples), **SDBS** (check terms).
- Papers with data supplements (search Google Scholar: "microplastics machine learning FTIR dataset").

## Steps
1. **Download** into `data/spectra_raw/<source>/`. Record URL, licence, date in `DATA_SOURCES.md`.
2. **Write loaders** (`src/spectra/load.py`) that convert every source into a common schema:
   | column | meaning |
   |---|---|
   | `id` | unique |
   | `source` | dataset name (used for group split!) |
   | `technique` | FTIR / Raman |
   | `polymer` | PET, PE, PP, PS, PVC, PA, other |
   | `x` | list of wavenumbers (cm⁻¹) |
   | `y` | list of intensities |
   | `meta` | mode (ATR/transmission), weathered?, colour, etc. |
   Common formats: CSV, JCAMP-DX (`.jdx`, use `jcamp` package or write a small parser), SPC (`spectrochempy`), R `.rds` (convert in R once).
3. **Normalise labels**: map synonyms (`polyethylene terephthalate`, `polyester`, `PETE` → PET; `nylon`, `polyamide 6/66` → PA; `HDPE/LDPE` → PE).
4. **Decide technique scope.** Recommended: **train FTIR and Raman as separate models** (their spectra look nothing alike). If data is limited, do FTIR first (usually more abundant), Raman as second model.
5. **Add a "Other/Non-plastic" class** if possible (cellulose, cotton, chitin, minerals such as quartz/calcite). Real filters contain natural particles; without this class the classifier will force every particle into a plastic label.
6. **Sanity-check the data**: count per class and per source, plot 5 random spectra per class, look for mislabelled/flat/duplicate spectra.
   ```python
   df.groupby(["technique","polymer"]).size().unstack()
   ```
7. **Remove duplicates** (same spectrum appearing in multiple libraries) — otherwise leakage inflates accuracy.

## Minimum target
≥ 30–50 spectra per class per technique, from ≥ 2 sources if possible. If a class is tiny, say so in the report.

## Risks
- Heavily imbalanced classes → handle in doc 04 (class weights, macro-F1).
- Different instruments/resolutions → handled by resampling in doc 03.

## Done when
`data/spectra_processed/all_spectra.parquet` (or pickle) exists, class table printed, provenance logged.
