# MICA: Model-Informed Change-point Analysis — Major Revision Materials

This repository contains the revised manuscript, supplementary material, response letter, and key figures for the major revision of:

> **MICA: Model-Informed Change-point Analysis**  
> PLOS Computational Biology — Major Revision (PCOMPBIOL-D-26-00585)

## Contents

- `manuscript/unmarked/` — Clean revised main manuscript (`Manuscript_PLOS.tex` / `.pdf`).
- `manuscript/marked/` — Diff-marked revised main manuscript (`Manuscript_PLOS_diff.tex` / `.pdf`), showing all changes against the original submission.
- `manuscript/supplementary/` — Revised supplementary material (`Supplementary_Material.tex` / `.pdf`).
- `manuscript/nofigures/` — Figure-free versions of the three documents (draft-mode compilation) for text-only review.
- `manuscript/tables/` — External LaTeX table inputs included by the main manuscript.
- `response/` — Point-by-point response to reviewers (`reviewer_comments_response_clean.md` / `.docx`).
- `figures/main/` — PLOS-style TIFF figures used in the main manuscript.
- `figures/supplementary/` — PDF figures used in the supplementary material.

## Key revisions

- Quantitative benchmarking on nine synthetic toy datasets and the 42-series Turing Change Point Dataset (TCPD).
- Clarified penalty and model-selection framework (BIC, MDL, AIC, plus legacy κ-heuristic).
- Redesigned core methodology figures (segmentation module, optimization module, DCLGA chromosome encoding).
- Profile-likelihood uncertainty quantification and structural-identifiability analysis for the COVID-19 and wind-turbine case studies.
- Updated COVID-19 and wind-turbine applications with principled information-criterion penalties and sensitivity analyses.
- Expanded limitations discussion and unified "Model-Informed" terminology.

## Source code

The implementation of MICA is available in the separate `Mica.jl` Julia package repository:
`https://github.com/Mehdilotfi7/MICA`

The profile-likelihood engine used for the uncertainty analyses is available as `BreakpointProfiles.jl`.

## License

See `LICENSE`.

## Contact

Lars Kaderali — lars.kaderali@uni-greifswald.de
