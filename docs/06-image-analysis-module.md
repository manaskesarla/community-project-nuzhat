# 06 — Image-Analysis Module: Detect, Count, Size (Days 7–11)

**Output:** `segment.py` + `measure.py` returning particle count, per-particle size (µm), size distribution, and a confidence estimate; validation vs synthetic ground truth.

## Classical pipeline (OpenCV / scikit-image)
```
load → grayscale/channel select → illumination correction → denoise
     → threshold → morphology cleanup → (split touching) → connected components
     → filter by size/shape → measure → overlay + table
```
### Steps
1. **Load & scale**: read image, `um_per_pixel` from config.
2. **Channel/contrast**: for fluorescence use green/brightest channel; for brightfield use grayscale or `a/b` channels in LAB (coloured particles stand out from white background).
3. **Illumination correction**: subtract large-scale background (`cv2.GaussianBlur` with big kernel, or morphological top-hat / rolling ball).
4. **Denoise**: median or Gaussian blur (3–5 px).
5. **Threshold**: Otsu, or **adaptive** (`cv2.adaptiveThreshold`) for uneven light; try Li/triangle in `skimage.filters`.
6. **Morphology**: opening to remove specks, closing to fill holes (`cv2.morphologyEx`). Remove objects < min area (noise, e.g. < 10 px).
7. **Split touching particles**: distance transform + watershed (`skimage.segmentation.watershed`).
8. **Measure** with `skimage.measure.regionprops`: area, equivalent diameter, major/minor axis, **Feret diameter**, eccentricity, solidity.
   - Convert: `d_um = d_px * um_per_pixel`.
   - **Shape class**: fibre (aspect ratio > ~3 and thin), fragment, film.
9. **Size distribution**: bin by standard classes (e.g. <20, 20–100, 100–500, 500–1000, >1000 µm) + histogram / cumulative curve; report median, D10/D50/D90.
10. **Overlay** contours + IDs on the image for visual QA, save to `outputs/`.

```python
props = regionprops(label(mask))
d_um = [p.equivalent_diameter * um_per_px for p in props if p.area >= min_area]
```

## Validation (vs synthetic ground truth)
- **Count error**: MAE, MAPE, correlation of predicted vs true count.
- **Size error**: compare predicted vs true diameters (MAE in µm, Bland–Altman).
- **Detection P/R/F1** with IoU ≥ 0.5 match between predicted and true instances.
- Break down by difficulty (easy/medium/hard, density). Show where it fails (overlap, fibres, small particles).
- On the 5–10 hand-annotated real images, report count agreement separately and **don't merge with synthetic metrics**.

## Confidence level (image side)
Combine simple, defensible signals into a 0–1 score:
- validation error for similar density/difficulty (from synthetic tests),
- fraction of particles with ambiguous segmentation (touching/low contrast),
- image quality (focus via Laplacian variance, SNR),
- fraction of particles near the min-size limit.
Report as Low / Medium / High with the rule documented.

## Optional ML upgrade (only if time)
Fine-tune a small U-Net or use pretrained **Cellpose/StarDist** for segmentation; compare vs classical. Keep classical as the primary baseline.

## Limits to state in the report
- Optical imaging can't tell plastic from non-plastic: counts are *particles of interest*, not confirmed plastics.
- Resolution floor (~ a few µm–tens of µm depending on optics); sub-limit particles missed.
- Proof-of-concept on synthetic + limited real data, **not a validated measurement**.

## Done when
`python -m src.images.measure img.png` prints count + size stats, saves overlay and CSV; validation table exists.
