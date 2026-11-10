"""Measure segmented particles and report count, sizes (µm), size distribution and a confidence level (doc 06).

python -m src.images.measure path/to/image.png [--um-per-pixel 2.0] [--out outputs/image_run]
Writes <out>/<stem>_particles.csv, <stem>_overlay.png, <stem>_summary.json
Counts are "particles of interest", NOT confirmed plastics (optical imaging cannot identify polymer).
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from skimage.measure import regionprops

from .segment import MIN_AREA_PX, load_rgb, segment

ROOT = Path(__file__).resolve().parents[2]
BINS_UM = [0, 20, 100, 500, 1000, np.inf]
BIN_LABELS = ["<20", "20-100", "100-500", "500-1000", ">1000"]


def default_um_per_pixel():
    return float(yaml.safe_load(open(ROOT / "config.yaml"))["um_per_pixel"])


def shape_class(p):
    ar = p.major_axis_length / max(p.minor_axis_length, 1e-6)
    if ar > 3.0 and p.solidity < 0.95 or ar > 5:
        return "fibre"
    if p.solidity < 0.8 and p.area > 400:
        return "film"
    return "fragment"


def confidence(df, img_u8, info, labels):
    """Transparent 0-1 score -> Low/Medium/High.
    score = 1 - (0.3*frac_small + 0.25*frac_ambiguous + 0.2*noise_penalty + 0.25*crowding_penalty)
      frac_small     : particles under 4x the minimum area (near the resolution floor)
      frac_ambiguous : particles touching the border, split by watershed, or with solidity < 0.7
      noise_penalty  : 1 - contrast/noise ratio / 15 (clipped 0-1); contrast = threshold-relative median object signal
      crowding_penalty: (fraction of image covered by particles - 0.05) / 0.10, clipped 0-1 (overlaps -> merged/missed particles)
    High >= 0.75, Medium >= 0.5, else Low. Fewer than 5 particles -> Low (not enough evidence)."""
    if df.empty:
        return dict(score=0.0, level="Low", frac_small=0, frac_ambiguous=0, noise_penalty=1, crowding_penalty=0, focus_laplacian_var=0)
    h, w = labels.shape
    frac_small = float((df.area_px < 4 * MIN_AREA_PX).mean())
    touch = df.apply(lambda r: r.bbox_r0 == 0 or r.bbox_c0 == 0 or r.bbox_r1 >= h or r.bbox_c1 >= w, axis=1)
    amb = (touch | (df.solidity < 0.7)).mean() + info["n_split"] / max(len(df), 1)
    frac_amb = float(min(amb, 1.0))
    sig = float(np.median([info["diff_map"][labels == i].mean() for i in df.id[:200]]))
    snr = sig / max(info["noise_sd"], 1e-6)
    pen = float(np.clip(1 - snr / 15, 0, 1))
    crowd = float(np.clip(((labels > 0).mean() - 0.05) / 0.10, 0, 1))
    score = float(1 - (0.3 * frac_small + 0.25 * frac_amb + 0.2 * pen + 0.25 * crowd))
    level = "High" if score >= 0.75 else "Medium" if score >= 0.5 else "Low"
    if len(df) < 5:
        level = "Low"
    focus = float(cv2.Laplacian(cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())
    return dict(score=round(score, 3), level=level, frac_small=round(frac_small, 3), frac_ambiguous=round(frac_amb, 3),
                noise_penalty=round(pen, 3), crowding_penalty=round(crowd, 3), focus_laplacian_var=round(focus, 2))


def measure_image(img_u8, um_per_px=None, **seg_kw):
    um = um_per_px or default_um_per_pixel()
    labels, info = segment(img_u8, **seg_kw)
    rows = []
    for p in regionprops(labels):
        r0, c0, r1, c1 = p.bbox
        rows.append(dict(id=p.label, area_px=int(p.area), area_um2=p.area * um ** 2, equiv_diameter_um=p.equivalent_diameter_area * um,
                         feret_um=p.feret_diameter_max * um, major_um=p.major_axis_length * um, minor_um=p.minor_axis_length * um,
                         aspect=p.major_axis_length / max(p.minor_axis_length, 1e-6), eccentricity=p.eccentricity, solidity=p.solidity,
                         shape=shape_class(p), cx=p.centroid[1], cy=p.centroid[0], bbox_r0=r0, bbox_c0=c0, bbox_r1=r1, bbox_c1=c1))
    df = pd.DataFrame(rows)
    conf = confidence(df, img_u8, info, labels)
    if df.empty:
        summ = dict(particle_count=0, size_bins={b: 0 for b in BIN_LABELS}, median_size_um=None)
    else:
        bins = pd.cut(df.feret_um, BINS_UM, labels=BIN_LABELS, right=False).value_counts().reindex(BIN_LABELS).fillna(0).astype(int)
        q = df.feret_um.quantile([.1, .5, .9]).to_numpy()
        summ = dict(particle_count=int(len(df)), size_bins=bins.to_dict(), median_size_um=float(q[1]),
                    D10_um=float(q[0]), D50_um=float(q[1]), D90_um=float(q[2]),
                    shape_counts=df["shape"].value_counts().to_dict(), size_metric="max Feret diameter")
    summ.update(um_per_pixel=um, confidence=conf, min_detectable_area_px=MIN_AREA_PX,
                note="proof of concept; counts are particles of interest, not confirmed plastics")
    return df, labels, summ


def overlay(img_u8, labels, df):
    out = img_u8.copy()
    cnts, _ = cv2.findContours((labels > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, (255, 0, 255), 1)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("image"); ap.add_argument("--um-per-pixel", type=float)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "image_run")); a = ap.parse_args()
    img = load_rgb(a.image); df, labels, summ = measure_image(img, a.um_per_pixel)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); stem = Path(a.image).stem
    df.drop(columns=[c for c in df if c.startswith("bbox")]).to_csv(out / f"{stem}_particles.csv", index=False)
    cv2.imwrite(str(out / f"{stem}_overlay.png"), cv2.cvtColor(overlay(img, labels, df), cv2.COLOR_RGB2BGR))
    json.dump(summ, open(out / f"{stem}_summary.json", "w"), indent=1, default=float)
    print(f"{stem}: {summ['particle_count']} particles; median size {summ['median_size_um']} µm; "
          f"confidence {summ['confidence']['level']} ({summ['confidence']['score']})")
    print("size bins (µm):", summ["size_bins"]); print("saved to", out)


if __name__ == "__main__":
    main()
