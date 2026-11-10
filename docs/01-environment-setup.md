# 01 — Environment Setup (Day 1)

**Time:** ~2 hours · **Output:** working Python env, repo skeleton, git initialised.

## Steps
1. **Python 3.10–3.12** and a virtual env:
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   ```
2. **requirements.txt**
   ```
   numpy
   pandas
   scipy
   scikit-learn
   matplotlib
   seaborn
   opencv-python
   scikit-image
   Pillow
   jupyter
   joblib
   tqdm
   jinja2
   pyyaml
   # optional / stretch
   torch
   streamlit
   weasyprint   # or use HTML export only
   ```
   `pip install -r requirements.txt && pip freeze > requirements.lock.txt`
3. **Create folders** from the layout in doc 00:
   ```bash
   mkdir -p data/{spectra_raw,spectra_processed,images_real,images_synthetic} notebooks src/{spectra,images,report} models outputs tests
   ```
4. `git init`, add `.gitignore` (`.venv/`, `data/`, `models/*.pkl`, `__pycache__/`).
5. **Config file** `config.yaml`: seeds, wavenumber grid, pixel-to-µm scale, class list.
   ```yaml
   seed: 42
   classes: [PET, PE, PP, PS, PVC, PA]
   wavenumber_grid: {min: 650, max: 4000, step: 2}   # FTIR cm^-1
   raman_grid: {min: 200, max: 3200, step: 2}
   um_per_pixel: 1.0   # set from calibration (doc 06)
   ```
6. **Data log** `data/DATA_SOURCES.md`: for every dataset record URL, licence, date downloaded, what you used. You will need this in the report.

## Done when
- `python -c "import cv2, sklearn, skimage"` works.
- Folder tree exists and is committed.
- Team roles assigned (suggest: A = spectra, B = images, C = report/community; all review).
