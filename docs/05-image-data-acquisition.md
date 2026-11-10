# 05 — Image Data Acquisition & Synthetic Set (Days 4–7, parallel to spectra)

**Output:** a folder of real/public filter-membrane images (if any) plus a clearly labelled synthetic set with ground truth.

## Be upfront
Open, licensed images of filtered microplastic particles are scarce. The brief says: use what you can find, add a **clearly labelled synthetic set**, and state the module is a **proof of concept**.

## A. Search for real/public images
- Zenodo, Figshare, Mendeley Data, Kaggle, Roboflow Universe, Hugging Face Datasets: "microplastics microscope", "microplastic filter", "microplastic segmentation", "fibre fragment microscopy".
- Papers with open image data (CC-BY). Check each licence; log in `data/DATA_SOURCES.md`.
- Wikimedia Commons (CC) for micrographs of particles/fibres as qualitative examples.
- **Do not** scrape images without a licence that permits reuse.

Save in `data/images_real/<source>/` and note for each: magnification, scale bar (µm/pixel) if known, stain (e.g. Nile Red fluorescence vs brightfield).

## B. Create a **synthetic** image set (`src/images/synth.py`)
Mimic a filter membrane under a stereo microscope or fluorescence:
1. **Background**: light-grey/white (brightfield) or dark (fluorescence) + Gaussian/Perlin noise + vignetting + a faint filter-pore texture.
2. **Particles**: draw N random objects with known geometry
   - *fragments*: random polygons / ellipses
   - *fibres*: bezier or random-walk polylines with thickness
   - *films*: irregular blobs
   Size drawn from a **log-normal** distribution (realistic: many small, few large). Random colours (blue, red, black, transparent-ish).
3. **Realism**: Gaussian blur (defocus), sensor noise, uneven illumination, a few dust specks and **distractors** (non-plastic debris, scratches), overlapping particles, edge clipping.
4. **Ground truth**: save per image a JSON with particle count, each particle's area/Feret diameter in px and µm, plus a binary/instance mask.
5. Generate ~100–200 images with varied density (10–500 particles) and difficulty levels (easy/medium/hard).
6. **Label everything**: filename prefix `SYNTH_`, a `README` in the folder, watermark or metadata flag, and a column `is_synthetic=True` in all result tables.

```python
rng = np.random.default_rng(42)
sizes_um = rng.lognormal(mean=np.log(50), sigma=0.8, size=n)   # 5 µm–1 mm range
```

## C. Calibration scale
Choose and document `um_per_pixel` (e.g. 1 px = 2 µm). For real images, read from the scale bar: measure bar length in pixels with a known µm value.

## D. Minimal manual ground truth for real images
Hand-annotate 5–10 real images (count particles with a click-counter or [CVAT]/[LabelMe]/ImageJ Cell Counter) to give *some* real validation.

## Done when
- ≥ 1 real image folder (even small) documented, and ≥ 100 synthetic images with ground-truth JSON/masks.
- Everything synthetic is unmistakably labelled.
