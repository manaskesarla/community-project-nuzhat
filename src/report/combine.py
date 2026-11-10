"""Combine one sample (membrane image + per-particle spectra) into result.json + report.html (doc 07).

python run_sample.py samples/sample_001      (or: python -m src.report.combine samples/sample_001)

Overall-confidence rule (documented, deliberately simple):
  spectral score = mean calibrated probability of the particles that were identified * (1 - fraction abstained)
                   High >= 0.80, Medium >= 0.60, else Low
  image level    = from src/images/measure.py (High / Medium / Low)
  sample-size cap: < 10 spectra -> at most Low; < 30 spectra -> at most Medium
  OVERALL        = the LOWEST of (spectral level, image level, sample-size cap)
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from jinja2 import Template

from src.images.measure import BIN_LABELS, measure_image, overlay
from src.images.segment import load_rgb
from src.spectra.predict import predict_file

ROOT = Path(__file__).resolve().parents[2]
CLASSES = ["PE", "PP", "PET", "PS", "PVC", "PA", "OTHER_PLASTIC", "NON_PLASTIC", "unidentified"]
LEVELS = ["Low", "Medium", "High"]
THRESHOLD = 0.5       # abstain below this calibrated probability


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 1.0]
    p = k / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [float(max(c - h, 0)), float(min(c + h, 1))]


def spectral_level(score):
    return "High" if score >= 0.80 else "Medium" if score >= 0.60 else "Low"


def run(sample_dir, out_root=None):
    sdir = Path(sample_dir); meta = yaml.safe_load(open(sdir / "meta.yaml")); sid = meta["id"]
    out = Path(out_root or ROOT / "outputs") / sid; out.mkdir(parents=True, exist_ok=True)

    # ---- image branch
    img = load_rgb(sdir / "image.png"); df, labels, isum = measure_image(img, meta.get("um_per_pixel"))
    cv2.imwrite(str(out / "annotated_image.png"), cv2.cvtColor(overlay(img, labels, df), cv2.COLOR_RGB2BGR))
    df.drop(columns=[c for c in df if c.startswith("bbox")]).to_csv(out / "particles.csv", index=False)

    # ---- spectral branch
    files = sorted((sdir / "spectra").glob("*.csv")); preds = [predict_file(f, THRESHOLD) for f in files]
    n = len(preds); calls = [p["polymer"] if p["polymer"] != "UNCERTAIN" else "unidentified" for p in preds]
    counts = {c: calls.count(c) for c in CLASSES}; props = {c: counts[c] / n if n else 0.0 for c in CLASSES}
    ci = {c: wilson(counts[c], n) for c in CLASSES}
    ident = [p["confidence"] for p in preds if p["polymer"] != "UNCERTAIN"]
    abstain = counts["unidentified"] / n if n else 1.0
    sscore = float(np.mean(ident) * (1 - abstain)) if ident else 0.0
    slevel = spectral_level(sscore)

    # ---- extrapolation to the whole filter (explicit assumption) + concentration
    npart = isum["particle_count"]
    expected = {c: dict(estimate=round(props[c] * npart, 1), low=round(ci[c][0] * npart, 1), high=round(ci[c][1] * npart, 1)) for c in CLASSES}
    conc = npart / meta["volume_filtered_L"] if meta.get("volume_filtered_L") else None

    # ---- overall confidence
    cap = "Low" if n < 10 else "Medium" if n < 30 else "High"
    lv = {"image": isum["confidence"]["level"], "spectral": slevel, "sample_size_cap": cap}
    overall = LEVELS[min(LEVELS.index(v) for v in lv.values())]

    caveats = ["Proof of concept: the image module is validated on synthetic images only and is not a validated measurement.",
               "Optical imaging cannot identify polymers: particle counts are 'particles of interest', not confirmed plastics.",
               "Polymer proportions assume the analysed spectra are a RANDOM subsample of the particles on the filter.",
               "The spectral model was trained on library-style reference spectra; weathered or contaminated environmental particles may be misidentified (see README limitations)."]
    if meta.get("image_is_synthetic"):
        caveats.insert(0, "SYNTHETIC IMAGE used for this sample.")
    if meta.get("spectra_are_constructed_demo"):
        caveats.insert(1, "Spectra were drawn from held-out test data to CONSTRUCT this demo sample; it is not a field sample.")
    if n < 30:
        caveats.append(f"Only {n} spectra analysed: proportion confidence intervals are wide.")
    if abstain > 0.15:
        caveats.append(f"{abstain:.0%} of spectra were below the confidence threshold and reported as unidentified.")
    if conc is not None and not np.isfinite(conc):
        conc = None

    result = dict(sample_id=sid, meta=meta,
                  image=dict(particle_count=npart, median_size_um=isum.get("median_size_um"), D10_um=isum.get("D10_um"), D90_um=isum.get("D90_um"),
                             size_bins=isum["size_bins"], shape_counts=isum.get("shape_counts", {}), confidence=isum["confidence"]["level"],
                             confidence_detail=isum["confidence"], is_synthetic=bool(meta.get("image_is_synthetic")), size_metric="max Feret diameter"),
                  n_spectra=n, polymer_counts=counts, polymers=props, polymers_ci95=ci, expected_particles_on_filter=expected,
                  spectral=dict(abstain_threshold=THRESHOLD, fraction_unidentified=abstain, mean_confidence_identified=float(np.mean(ident)) if ident else None,
                                score=sscore, level=slevel),
                  particles_per_litre=conc, confidence_components=lv, overall_confidence=overall, caveats=caveats)
    json.dump(result, open(out / "result.json", "w"), indent=2, default=float)

    # ---- figures
    fig, ax = plt.subplots(figsize=(5, 3)); ax.bar(BIN_LABELS, [isum["size_bins"][b] for b in BIN_LABELS], color="#457b9d")
    ax.set_xlabel("max Feret diameter (µm)"); ax.set_ylabel("particles"); ax.set_title("Size distribution (image)"); fig.tight_layout(); fig.savefig(out / "size_hist.png", dpi=130); plt.close(fig)
    cl = [c for c in CLASSES if counts[c] > 0]; fig, ax = plt.subplots(figsize=(5.5, 3))
    y = np.arange(len(cl)); v = [props[c] for c in cl]
    ax.barh(y, v, xerr=[[props[c] - ci[c][0] for c in cl], [ci[c][1] - props[c] for c in cl]], color="#2a9d8f", capsize=3)
    ax.set_yticks(y); ax.set_yticklabels(cl); ax.invert_yaxis(); ax.set_xlabel("proportion of analysed particles (95% Wilson CI)"); ax.set_xlim(0, 1)
    fig.tight_layout(); fig.savefig(out / "polymer_bar.png", dpi=130); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 3)); ax.hist([p["confidence"] for p in preds], bins=np.linspace(0, 1, 21), color="#e9c46a")
    ax.axvline(THRESHOLD, color="k", ls="--", lw=.8); ax.set_xlabel("calibrated probability of top class"); ax.set_ylabel("spectra")
    ax.set_title("Spectral confidence"); fig.tight_layout(); fig.savefig(out / "confidence_hist.png", dpi=130); plt.close(fig)

    html = Template((ROOT / "src/report/template.html").read_text()).render(r=result, classes=[c for c in CLASSES if counts[c] > 0], levels=LEVELS)
    (out / "report.html").write_text(html)
    return result, out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("sample"); a = ap.parse_args()
    r, out = run(a.sample)
    print(f"{r['sample_id']}: {r['image']['particle_count']} particles, {r['n_spectra']} spectra, overall confidence {r['overall_confidence']}")
    print("polymers:", {k: round(v, 2) for k, v in r["polymers"].items() if v}); print("wrote", out / "report.html", "and", out / "result.json")


if __name__ == "__main__":
    main()
