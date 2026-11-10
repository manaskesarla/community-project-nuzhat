# 08 — Written Report: SDG Justification & Community Component (Days 15–19, drafting from Week 1)

**Output:** a ~10–15 page report (PDF) + README.

## Suggested structure
1. **Abstract** (150–200 words).
2. **Introduction & problem** — microplastics in water/food; why detection is slow and expensive (µ-FTIR/Raman need experts, hours per sample); scale of the issue (cite WHO, UNEP, peer-reviewed reviews).
3. **SDG justification** (see below).
4. **Community component** (see below).
5. **Methods** — spectral data & preprocessing (02–03), classifier (04), image data incl. synthetic (05), image analysis (06), combination & confidence (07).
6. **Results** — tables/figures: confusion matrix, macro-F1 ± std, calibration curve, image count/size errors, example sample report.
7. **Discussion** — what worked, failure cases, comparison to manual workflow.
8. **Limitations & honesty statement** — image module = proof of concept; synthetic data; library spectra ≠ weathered environmental particles; label noise; domain shift; not for regulatory use.
9. **Ethics, data & licences** — table from `DATA_SOURCES.md`, privacy for community data.
10. **Future work** — real labelled datasets, µ-FTIR mapping linking per-particle, deep learning, field validation.
11. **References**, **Appendix** (code repo, reproduction steps).

## SDG justification (pick and argue; don't just list)
- **SDG 14 – Life Below Water** (Target 14.1: prevent and reduce marine pollution incl. marine debris) — primary.
- **SDG 6 – Clean Water & Sanitation** (6.3 water quality; microplastics in drinking/surface water).
- **SDG 3 – Good Health** (exposure pathways) and **SDG 12 – Responsible Consumption & Production** (12.5 waste reduction) as secondary.
- **SDG 9/17** for open tools and partnerships (optional).
For each chosen SDG: *target number → how your tool contributes → what indicator it could inform → what it does NOT do* (avoid overclaiming).

## Community component — ideas (choose one that's feasible in 3 weeks)
1. **Citizen-science sampling kit design**: a simple protocol (sample water, filter, photograph with phone macro lens + scale card) feeding the image module; test with 5–10 volunteers / classmates.
2. **Stakeholder interviews** (3–5): local water-quality officers, fisher communities, school teachers, NGOs, lab technicians — ask what they need from a screening tool. Summarise themes and how the design responds.
3. **Awareness workshop / school session** using the dashboard or printed sample reports; collect pre/post survey.
4. **Open-source release + documentation** in plain language (and local language if relevant), with a "how to read the report" one-pager.
5. **Local pilot**: collect water from a local lake/river/tap (with permission), photograph filters, report findings to the community.

Document: who, how many, consent, what you learned, what changed in the project because of it. Keep personal data out of the report (anonymise; get consent).

## Writing tips
- Write the Methods section as you build; draft figures directly from notebooks (save to `outputs/figures/`).
- Every number in the text must be traceable to a script/notebook.
- Have someone not on the team read the abstract and limitations.

## Done when
PDF exported, all figures embedded, references complete, limitations section explicitly mentions the image module is a proof of concept on synthetic + limited data.
