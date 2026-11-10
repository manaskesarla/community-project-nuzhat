import numpy as np
import pytest

from src.spectra import predict as P


def test_read_spectrum_handles_header_and_separators(tmp_path):
    f = tmp_path / "s.csv"
    f.write_text("wavenumber,absorbance\n" + "\n".join(f"{700+i},{i*0.01}" for i in range(50)))
    wn, y = P.read_spectrum(f)
    assert len(wn) == 50 and wn[0] == 700


def test_read_spectrum_rejects_garbage(tmp_path):
    f = tmp_path / "bad.csv"
    f.write_text("hello\nworld\n")
    with pytest.raises(ValueError):
        P.read_spectrum(f)


def test_to_grid_rejects_narrow_range():
    grid = np.arange(660, 3891, 2.0)
    with pytest.raises(ValueError):
        P.to_grid(np.linspace(1000, 2000, 100), np.zeros(100), grid)
