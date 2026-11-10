"""Classical particle segmentation for brightfield filter images (doc 06).

load -> illumination correction -> colour-difference map -> robust threshold -> clean-up -> split touching -> labels
Proof of concept: tuned on synthetic images only (see src/images/validate.py).
"""
import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import apply_hysteresis_threshold
from skimage.measure import label, regionprops
from skimage.morphology import h_maxima
from skimage.segmentation import watershed

MIN_AREA_PX = 15      # smaller blobs are treated as noise/dust (resolution floor)
K_LOW = 5.0           # hysteresis: grow region while diff > median + K_LOW * robust noise sd
K_HIGH = 12.0         # ...but keep a region only if it contains a pixel > median + K_HIGH * sd (kills isolated speckle)
FLOOR_LOW, FLOOR_HIGH = 0.05, 0.09   # absolute minimums on the 0-1 colour-difference scale
EDGE_FRAC = 0.5       # refine each particle's edge at this fraction of its own contrast (avoids blur-halo size bias)


def load_rgb(path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def estimate_background(img_u8, down=8, ksize=41):
    """Large median filter on a down-scaled copy: removes particles smaller than ~ksize*down px, keeps slow illumination."""
    h, w = img_u8.shape[:2]
    small = cv2.resize(img_u8, (w // down, h // down), interpolation=cv2.INTER_AREA)
    small = cv2.medianBlur(small, ksize)
    bg = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    return cv2.GaussianBlur(bg.astype(np.float32) / 255, (0, 0), down)


def difference_map(img_u8):
    """Per-pixel colour distance from the estimated background (so dark, coloured AND pale particles stand out)."""
    img = img_u8.astype(np.float32) / 255
    d = np.sqrt(((img - estimate_background(img_u8)) ** 2).sum(2))
    return cv2.GaussianBlur(d, (0, 0), 1.0)


def refine_edges(mask, d, low):
    """Re-threshold every connected region at EDGE_FRAC of its own (90th-percentile) contrast, inside a 3-px dilated region.
    The low hysteresis threshold otherwise grows blurred particles outward and biases sizes upward."""
    lab = label(mask, connectivity=2); out = np.zeros_like(mask)
    for p in regionprops(lab):
        r0, c0, r1, c1 = p.bbox; r0, c0, r1, c1 = max(r0 - 3, 0), max(c0 - 3, 0), r1 + 3, c1 + 3
        reg = lab[r0:r1, c0:c1] == p.label
        reg = cv2.dilate(reg.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
        sub = d[r0:r1, c0:c1]; core = np.percentile(sub[lab[r0:r1, c0:c1] == p.label], 90)
        out[r0:r1, c0:c1] |= reg & (sub > max(low, EDGE_FRAC * core))
    return out


def segment(img_u8, min_area=MIN_AREA_PX, k_low=K_LOW, k_high=K_HIGH, split=True):
    """Return (labels int32, info dict). labels==0 is background."""
    d = difference_map(img_u8)
    med = float(np.median(d)); sd = 1.4826 * float(np.median(np.abs(d - med)))
    low, high = max(med + k_low * sd, FLOOR_LOW), max(med + k_high * sd, FLOOR_HIGH)
    mask = apply_hysteresis_threshold(d, low, high)
    mask = refine_edges(mask, d, low)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)).astype(bool)
    mask = ndi.binary_fill_holes(mask)
    lab = label(mask, connectivity=2)
    n_split = 0
    if split:                                    # watershed only for compact, non-convex blobs (not fibres)
        nxt = lab.max() + 1
        for p in regionprops(lab):
            if p.area < 150 or p.solidity > 0.88 or p.major_axis_length / max(p.minor_axis_length, 1) > 3.5:
                continue
            r0, c0, r1, c1 = p.bbox; sub = np.pad(p.image, 2)
            dist = ndi.distance_transform_edt(sub)
            peaks = label(h_maxima(dist, 2.0))
            if peaks.max() < 2:
                continue
            ws = watershed(-dist, peaks, mask=sub)[2:-2, 2:-2]
            view = lab[r0:r1, c0:c1]
            for j in range(2, ws.max() + 1):
                view[(ws == j) & p.image] = nxt; nxt += 1
            n_split += ws.max() - 1
    out = np.zeros_like(lab, np.int32); k2 = 0
    for p in regionprops(lab):
        if p.area >= min_area:
            k2 += 1; out[lab == p.label] = k2
    return out, dict(threshold=low, noise_sd=sd, n_split=int(n_split), diff_map=d)
