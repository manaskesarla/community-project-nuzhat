"""Validate the image module against synthetic ground truth.  python -m src.images.validate

Images SYNTH_0001-0030 are the DEV set (thresholds were tuned on them); SYNTH_0031-0150 are the HELD-OUT set.
Headline numbers are reported on the held-out set only. SYNTHETIC DATA ONLY: this says nothing about real images.
Outputs: outputs/image_validation_per_image.csv, outputs/image_validation.json, outputs/image_validation.png
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .measure import measure_image
from .segment import load_rgb, MIN_AREA_PX

ROOT = Path(__file__).resolve().parents[2]
SYN = ROOT / "data" / "images_synthetic"
OUT = ROOT / "outputs"
N_DEV = 30


def match(gt_mask, pr_mask, iou_thr=0.5):
    """Greedy IoU matching of instance masks. Returns list of (gt_id, pred_id)."""
    g, p = gt_mask.ravel().astype(np.int64), pr_mask.ravel().astype(np.int64)
    sel = (g > 0) & (p > 0)
    if not sel.any():
        return []
    gi, pi = np.unique(g[sel]), np.unique(p[sel])
    pair = pd.Series(g[sel] * 1_000_000 + p[sel]).value_counts()
    ga = pd.Series(g[g > 0]).value_counts(); pa = pd.Series(p[p > 0]).value_counts()
    cand = []
    for key, inter in pair.items():
        a, b = divmod(int(key), 1_000_000)
        iou = inter / (ga[a] + pa[b] - inter)
        if iou >= iou_thr:
            cand.append((iou, a, b))
    used_g, used_p, out = set(), set(), []
    for iou, a, b in sorted(cand, reverse=True):
        if a not in used_g and b not in used_p:
            used_g.add(a); used_p.add(b); out.append((a, b))
    return out


def main():
    rows, size_rows = [], []
    files = sorted(SYN.glob("SYNTH_*.png")); files = [f for f in files if not f.stem.endswith("_mask")]
    for k, f in enumerate(files):
        gt = json.load(open(f.with_suffix(".json"))); gmask = np.array(__import__("cv2").imread(str(f.parent / f"{f.stem}_mask.png"), -1))
        df, labels, summ = measure_image(load_rgb(f), gt["um_per_pixel"])
        pairs = match(gmask, labels); gtp = {p["id"]: p for p in gt["particles"]}
        det_gt = [p for p in gt["particles"] if p["area_px"] >= MIN_AREA_PX]            # GT particles above the resolution floor
        n_gt, n_det, n_pred, tp = len(gt["particles"]), len(det_gt), summ["particle_count"], len(pairs)
        tp_det = sum(1 for a, _ in pairs if gtp[a]["area_px"] >= MIN_AREA_PX)
        prec = tp / n_pred if n_pred else 0; rec = tp_det / n_det if n_det else 0
        rows.append(dict(image=f.stem, split="dev" if k < N_DEV else "heldout", difficulty=gt["difficulty"], gt_count=n_gt,
                         gt_count_detectable=n_det, pred_count=n_pred, tp=tp, precision=prec, recall_detectable=rec,
                         f1=2 * prec * rec / (prec + rec) if prec + rec else 0, confidence=summ["confidence"]["level"],
                         confidence_score=summ["confidence"]["score"]))
        prs = df.set_index("id")
        for a, b in pairs:
            size_rows.append(dict(image=f.stem, split=rows[-1]["split"], difficulty=gt["difficulty"], kind=gtp[a]["kind"],
                                  gt_feret_um=gtp[a]["feret_um"], pred_feret_um=prs.loc[b, "feret_um"],
                                  gt_equiv_um=gtp[a]["equiv_diameter_um"], pred_equiv_um=prs.loc[b, "equiv_diameter_um"]))
    r = pd.DataFrame(rows); s = pd.DataFrame(size_rows)
    OUT.mkdir(exist_ok=True); r.to_csv(OUT / "image_validation_per_image.csv", index=False)
    s["err_feret"] = s.pred_feret_um - s.gt_feret_um; s["err_equiv"] = s.pred_equiv_um - s.gt_equiv_um

    def summarise(rr, ss):
        e = rr.pred_count - rr.gt_count_detectable
        return dict(n_images=int(len(rr)), count_MAE=float(np.abs(e).mean()),
                    count_MAPE_pct=float((np.abs(e) / rr.gt_count_detectable.clip(lower=1)).mean() * 100),
                    count_bias=float(e.mean()), count_pearson_r=float(np.corrcoef(rr.pred_count, rr.gt_count_detectable)[0, 1]),
                    precision=float(rr.precision.mean()), recall_detectable=float(rr.recall_detectable.mean()), F1=float(rr.f1.mean()),
                    size_MAE_equiv_um=float(ss.err_equiv.abs().mean()) if len(ss) else None,
                    size_bias_equiv_um=float(ss.err_equiv.mean()) if len(ss) else None,
                    size_MAE_feret_um=float(ss.err_feret.abs().mean()) if len(ss) else None)

    ho, hs = r[r.split == "heldout"], s[s.split == "heldout"]
    res = dict(note="synthetic data only. count error compares predicted count to GT particles >= min detectable area",
               dev=summarise(r[r.split == "dev"], s[s.split == "dev"]), heldout=summarise(ho, hs),
               heldout_by_difficulty={d: summarise(ho[ho.difficulty == d], hs[hs.difficulty == d]) for d in ["easy", "medium", "hard"]},
               heldout_all_gt_count_vs_pred=dict(
                   missed_below_floor_pct=float(100 * (1 - ho.gt_count_detectable.sum() / ho.gt_count.sum()))),
               heldout_by_confidence_level={lv: dict(n=int((ho.confidence == lv).sum()),
                                                      mean_abs_count_pct_err=float((np.abs(ho[ho.confidence == lv].pred_count - ho[ho.confidence == lv].gt_count_detectable)
                                                                                    / ho[ho.confidence == lv].gt_count_detectable.clip(lower=1)).mean() * 100))
                                            for lv in ["High", "Medium", "Low"] if (ho.confidence == lv).any()},
               heldout_size_error_by_shape={k: dict(n=int(len(g)), MAE_equiv_um=float(g.err_equiv.abs().mean()), bias_equiv_um=float(g.err_equiv.mean()))
                                            for k, g in hs.groupby("kind")})
    json.dump(res, open(OUT / "image_validation.json", "w"), indent=2)

    fig, ax = plt.subplots(1, 3, figsize=(13, 4)); col = {"easy": "#2a9d8f", "medium": "#e9c46a", "hard": "#e76f51"}
    for d, g in ho.groupby("difficulty"):
        ax[0].scatter(g.gt_count_detectable, g.pred_count, s=14, label=d, color=col[d])
    m = ho.gt_count_detectable.max(); ax[0].plot([0, m], [0, m], "k--", lw=.8); ax[0].set_xlabel("true count (detectable)"); ax[0].set_ylabel("predicted count")
    ax[0].legend(); ax[0].set_title("Count: predicted vs true (held-out, synthetic)")
    mean = (hs.gt_equiv_um + hs.pred_equiv_um) / 2; ax[1].scatter(mean, hs.err_equiv, s=3, alpha=.3); ax[1].axhline(0, color="k", lw=.8)
    ax[1].set_xscale("log"); ax[1].set_xlabel("mean equiv. diameter (µm)"); ax[1].set_ylabel("predicted − true (µm)"); ax[1].set_title("Size: Bland–Altman")
    ho.boxplot(column="f1", by="difficulty", ax=ax[2]); ax[2].set_title("Detection F1 (IoU≥0.5) by difficulty"); ax[2].set_xlabel(""); fig.suptitle("")
    fig.tight_layout(); fig.savefig(OUT / "image_validation.png", dpi=140)
    print(json.dumps({k: res[k] for k in ["dev", "heldout", "heldout_by_difficulty", "heldout_by_confidence_level", "heldout_size_error_by_shape",
                                          "heldout_all_gt_count_vs_pred"]}, indent=1))


if __name__ == "__main__":
    main()
