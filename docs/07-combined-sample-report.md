# 07 — Combined Sample Report (Days 12–14)

**Output:** a single command that takes one "sample" (spectra + membrane image) and produces a report with polymer types, particle count, size distribution and overall confidence.

## Define a "sample"
```
samples/sample_001/
├── meta.yaml          # id, location, date, volume filtered (L), mesh/pore size, um_per_pixel
├── image.png          # filter-membrane image
└── spectra/*.csv      # one file per analysed particle (FTIR or Raman)
```
For the demo, build `sample_001` by combining a real or synthetic image with spectra drawn from the test set (clearly state this is a constructed demo sample). Optionally create `sample_002` with a different composition to show the output varies.

## Pipeline (`src/report/combine.py`)
1. **Image branch**: run doc 06 → count, sizes, size histogram, image confidence.
2. **Spectral branch**: for each spectrum run doc 04 `predict` → polymer, probability, HQI; apply abstain threshold.
3. **Aggregate polymers**: counts and % per polymer (+ "unidentified"). Extrapolate to the whole image **only with an explicit assumption** (spectra are a random subsample of particles → proportions extrapolated; give a binomial / Wilson confidence interval on proportions).
4. **Concentration** (optional): `particles / L = count / volume_filtered`.
5. **Overall confidence**: combine
   - spectral confidence (mean calibrated prob. of identified particles, fraction abstained),
   - image confidence,
   - sample size (few spectra → wider CI).
   Use a transparent rule (e.g. the minimum of the two, or a weighted mean) and label **Low/Medium/High**. Document the rule.
6. **Outputs** (`outputs/sample_001/`):
   - `result.json` (machine-readable),
   - figures: annotated image, size histogram, polymer pie/bar with CI, confidence-vs-prob plot,
   - `report.html` (Jinja2 template) → optionally PDF.

## Example `result.json`
```json
{
  "sample_id": "sample_001",
  "image": {"particle_count": 214, "median_size_um": 58.2,
            "size_bins": {"<20": 31, "20-100": 120, "100-500": 58, ">500": 5},
            "confidence": "Medium", "is_synthetic": true},
  "polymers": {"PE": 0.38, "PP": 0.27, "PET": 0.14, "PS": 0.05,
               "PVC": 0.02, "PA": 0.04, "unidentified": 0.10},
  "polymers_ci95": {"PE": [0.28, 0.49]},
  "n_spectra": 60,
  "overall_confidence": "Medium",
  "caveats": ["Image module is a proof of concept", "Synthetic image used"]
}
```

## Report template must always show
- Banner: **"Proof of concept — not a validated measurement"** and **"Synthetic data used"** where applicable.
- Inputs and parameters used.
- Assumptions (subsampling, scale calibration).
- Caveats list generated automatically.

## Tests
- Run on 2 samples end-to-end with no manual steps.
- Unit test: output JSON has all required keys; proportions sum to 1.

## Done when
`python run_sample.py samples/sample_001` writes `report.html` + `result.json`.
