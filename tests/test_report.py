import json
from pathlib import Path

import pytest

from src.report.combine import run, wilson

SAMPLES = Path(__file__).resolve().parents[1] / "samples"
KEYS = {"sample_id", "image", "n_spectra", "polymers", "polymers_ci95", "overall_confidence", "caveats", "particles_per_litre"}


def test_wilson_bounds():
    lo, hi = wilson(5, 20); assert 0 <= lo < 0.25 < hi <= 1
    assert wilson(0, 0) == [0.0, 1.0]


@pytest.mark.skipif(not (SAMPLES / "sample_001").exists(), reason="demo samples not built")
def test_end_to_end_sample(tmp_path):
    r, out = run(SAMPLES / "sample_001", tmp_path)
    assert KEYS <= set(r) and (out / "report.html").exists() and (out / "result.json").exists()
    assert abs(sum(r["polymers"].values()) - 1) < 1e-9
    assert r["overall_confidence"] in {"Low", "Medium", "High"}
    html = (out / "report.html").read_text()
    assert "Proof of concept" in html and "Synthetic data used" in html
    assert json.load(open(out / "result.json"))["sample_id"] == "sample_001"
