import numpy as np
import pytest

from src.images.measure import measure_image
from src.images.synth import make_image
from src.images.validate import match


def test_synth_is_labelled_and_deterministic():
    a, ia, ga = make_image(123, "easy", 2.0); b, ib, gb = make_image(123, "easy", 2.0)
    assert ga["is_synthetic"] is True and np.array_equal(a, b) and np.array_equal(ia, ib)
    assert ga["particle_count"] == len(ga["particles"]) == len(np.unique(ia)) - 1


def test_measure_counts_easy_image_close_to_truth():
    img, _, gt = make_image(7, "easy", 2.0)
    df, labels, summ = measure_image(img, 2.0)
    detectable = sum(p["area_px"] >= 15 for p in gt["particles"])
    assert abs(summ["particle_count"] - detectable) <= max(5, 0.3 * detectable)
    assert summ["confidence"]["level"] in {"High", "Medium", "Low"}
    assert (df.equiv_diameter_um > 0).all()


def test_blank_image_gives_zero_particles():
    df, _, summ = measure_image(np.full((256, 256, 3), 220, np.uint8), 2.0)
    assert summ["particle_count"] == 0 and summ["confidence"]["level"] == "Low"


def test_match_iou():
    g = np.zeros((20, 20), np.uint16); g[2:10, 2:10] = 1
    p = np.zeros((20, 20), np.int32); p[2:10, 2:9] = 5
    q = np.zeros((20, 20), np.int32); q[15:19, 15:19] = 5
    assert match(g, p) == [(1, 5)] and match(g, q) == []
