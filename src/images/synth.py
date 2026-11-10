"""Synthetic filter-membrane images WITH ground truth (doc 05).  python -m src.images.synth [--n 150]

EVERYTHING produced here is synthetic: files are prefixed SYNTH_, carry a visible "SYNTHETIC" tag,
and every JSON has is_synthetic=true. Brightfield look: pale membrane, coloured/dark fragments, fibres,
films, plus distractors (dust, scratches) that are NOT counted in the ground truth.

Per image in data/images_synthetic/:
  SYNTH_0001.png        the image
  SYNTH_0001.json       ground truth (count, per-particle size in px/um, is_synthetic=true)
  SYNTH_0001_mask.png   uint16 instance-id mask (visible, post-overlap pixels)
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml
from skimage.measure import regionprops

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "images_synthetic"
SIZE = 1024
DIFFICULTY = {  # n particles (range), sensor noise sd, defocus blur sigma (px), dust specks, scratches, illumination gradient
    "easy":   dict(n=(10, 60),   noise=0.012, blur=0.8, dust=(0, 8),    scratches=(0, 1), grad=0.03),
    "medium": dict(n=(60, 200),  noise=0.025, blur=1.2, dust=(8, 40),   scratches=(1, 4), grad=0.06),
    "hard":   dict(n=(200, 500), noise=0.04,  blur=1.8, dust=(40, 120), scratches=(3, 8), grad=0.10),
}
PALETTE = {  # RGB 0-1: typical microplastic colours
    "blue": (0.10, 0.25, 0.75), "red": (0.75, 0.12, 0.12), "black": (0.08, 0.08, 0.08),
    "green": (0.15, 0.55, 0.25), "white": (0.97, 0.97, 0.97), "clear": (0.80, 0.82, 0.84), "orange": (0.90, 0.50, 0.10),
}
COLOUR_P = [.25, .2, .2, .1, .05, .1, .1]


def _background(rng, d):
    y, x = np.mgrid[0:SIZE, 0:SIZE] / SIZE
    bg = rng.uniform(0.84, 0.92) + d["grad"] * (rng.uniform(-1, 1) * x + rng.uniform(-1, 1) * y)   # uneven illumination
    bg = bg - 0.10 * ((x - .5) ** 2 + (y - .5) ** 2)                                                 # vignetting
    pores = cv2.GaussianBlur(rng.normal(0, 1, (SIZE, SIZE)).astype(np.float32), (0, 0), 3)
    bg = bg + 0.012 * pores / pores.std()                                                           # faint pore texture
    return np.repeat(bg[..., None], 3, 2).astype(np.float32)


def _polygon(rng, cx, cy, r):
    k = rng.integers(5, 10)
    ang = np.sort(rng.uniform(0, 2 * np.pi, k)); rad = r * rng.uniform(0.6, 1.2, k); sq = rng.uniform(0.5, 1.0)
    return np.c_[cx + rad * np.cos(ang), cy + sq * rad * np.sin(ang)].astype(np.int32)


def _draw_particle(rng, kind, diam_px):
    m = np.zeros((SIZE, SIZE), np.uint8)
    cx, cy = rng.uniform(0, SIZE, 2)
    if kind == "fibre":                                  # random-walk polyline
        th = rng.uniform(0, 2 * np.pi); pts = [(cx, cy)]
        for _ in range(max(int(diam_px / 8), 3)):
            th += rng.normal(0, 0.25); pts.append((pts[-1][0] + 8 * np.cos(th), pts[-1][1] + 8 * np.sin(th)))
        cv2.polylines(m, [np.array(pts, np.int32).reshape(-1, 1, 2)], False, 1, int(rng.integers(2, 6)), cv2.LINE_AA)
    elif kind == "film":                                 # irregular smooth blob
        cv2.fillPoly(m, [_polygon(rng, cx, cy, diam_px / 2 * 1.2)], 1)
        m = (cv2.GaussianBlur(m.astype(np.float32), (0, 0), diam_px / 12 + 0.5) > 0.5).astype(np.uint8)
    else:                                                # fragment
        cv2.fillPoly(m, [_polygon(rng, cx, cy, diam_px / 2)], 1)
    return m.astype(bool)


def make_image(seed, difficulty, um_per_px):
    rng = np.random.default_rng(seed); d = DIFFICULTY[difficulty]
    img = _background(rng, d); inst = np.zeros((SIZE, SIZE), np.uint16); kinds = {}
    n = int(rng.integers(*d["n"]))
    sizes_um = np.clip(rng.lognormal(np.log(50), 0.8, n), 8, 600)   # many small, few large
    for i, s_um in enumerate(sizes_um, 1):
        kind = str(rng.choice(["fragment", "fibre", "film"], p=[0.6, 0.25, 0.15]))
        m = _draw_particle(rng, kind, s_um / um_per_px * (3 if kind == "fibre" else 1))   # fibres: length >> width
        if m.sum() < 4:
            continue
        cname = str(rng.choice(list(PALETTE), p=COLOUR_P)); col = np.array(PALETTE[cname], np.float32)
        alpha = 0.55 if cname == "clear" else 0.95
        img[m] = (1 - alpha) * img[m] + alpha * np.clip(col * (1 + rng.normal(0, 0.05)), 0, 1)
        inst[m] = i; kinds[i] = (kind, cname)
    for _ in range(int(rng.integers(*d["dust"]))):                   # distractors: NOT in ground truth
        g = float(rng.uniform(.3, .6))
        cv2.circle(img, tuple(int(v) for v in rng.uniform(0, SIZE, 2)), int(rng.integers(1, 3)), (g, g, g), -1, cv2.LINE_AA)
    for _ in range(int(rng.integers(*d["scratches"]))):
        p0, p1 = rng.uniform(0, SIZE, 2), rng.uniform(0, SIZE, 2)
        cv2.line(img, tuple(int(v) for v in p0), tuple(int(v) for v in p1), (0.78,) * 3, 1, cv2.LINE_AA)
    img = cv2.GaussianBlur(img, (0, 0), d["blur"])                                                      # defocus
    img = np.clip(img + rng.normal(0, d["noise"], img.shape).astype(np.float32), 0, 1)                  # sensor noise
    gt = []                                                          # truth = VISIBLE instances (>= 4 px after overlap)
    for p in regionprops(inst):
        if p.area < 4:
            continue
        kind, cname = kinds[p.label]
        gt.append(dict(id=int(p.label), kind=kind, colour=cname, area_px=int(p.area), area_um2=float(p.area * um_per_px ** 2),
                       equiv_diameter_um=float(p.equivalent_diameter_area * um_per_px), feret_um=float(p.feret_diameter_max * um_per_px),
                       centroid_xy=[float(p.centroid[1]), float(p.centroid[0])]))
    inst[~np.isin(inst, [g["id"] for g in gt])] = 0
    img = np.ascontiguousarray((img * 255).astype(np.uint8))
    cv2.putText(img, "SYNTHETIC", (10, SIZE - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (115, 115, 115), 1, cv2.LINE_AA)
    meta = dict(is_synthetic=True, difficulty=difficulty, seed=int(seed), um_per_pixel=um_per_px,
                image_size_px=SIZE, particle_count=len(gt), particles=gt)
    return img, inst, meta


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=150); a = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / "config.yaml")); upp = float(cfg["um_per_pixel"]); OUT.mkdir(parents=True, exist_ok=True)
    levels = list(DIFFICULTY)
    for i in range(a.n):
        img, inst, gt = make_image(cfg["seed"] * 1000 + i, levels[i % 3], upp); stem = f"SYNTH_{i + 1:04d}"
        cv2.imwrite(str(OUT / f"{stem}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(OUT / f"{stem}_mask.png"), inst)
        json.dump(gt, open(OUT / f"{stem}.json", "w"), indent=1)
    (OUT / "README.txt").write_text(
        "ALL FILES HERE ARE SYNTHETIC (generated by src/images/synth.py; seed from config.yaml).\n"
        "They are NOT real microscope images. *_mask.png = uint16 instance ids; *.json = ground truth (is_synthetic=true).\n")
    print(f"wrote {a.n} synthetic images to {OUT}")


if __name__ == "__main__":
    main()
