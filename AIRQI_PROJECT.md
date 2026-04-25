# AIRQI Project Summary

This file maps main files and folders to the five-step project flow and gives quick notes for graders or new contributors.

1) Data Collection — `step1_data_collection/`
- Purpose: acquire raw JSON / CSV data and populate the SQLite DB.
- Key files: `code/setup_pull.py`, `code/pull.py`, `code/parse.py`, `get_aqi.py`.

2) Data Preparation — `step2_data_prep/`
- Purpose: initial cleaning and normalization.
- Key files: `step2_data_prep/scripts/feature_engineering.py` (entry point for engineering), `organize,py` (helper).

3) Feature Engineering — `step3_data_preprocessing/` and top-level artifacts
- Purpose: create engineered columns, rolling features, encodings used for EDA and modeling.
- Key artifacts: `featureengineering.xlsx` (EDA-friendly) and `featureengineering_model_ready.xlsx` (model-safe — no leakage).

4) Exploratory Data Analysis — `step4_eda/`
- Purpose: generate visual artifacts and narrative analysis.
- Key notebooks: `step4_eda/notebooks/step4_eda.ipynb` (reproducible EDA), generated PNGs in `step4_eda/plots/`.

5) Model Selection & Deployment — `step5_model_selection/`
- Purpose: training experiments, model selection, saved artifacts, and a small demo dashboard.
- Key contents:
  - `step5_model_selection/models/` — picked model (e.g., `stacked_ensemble.pkl`).
  - `step5_model_selection/metrics/` — CSVs/text files with evaluation metrics.
  - `step5_model_selection/data/` — `featureengineering_model_ready.xlsx` used for modeling.
  - `step5_model_selection/visuals/` — selected PNGs for reporting.
  - `step5_model_selection/open_source_dashboard/` — Streamlit dashboard script for quick demo.

Canonical artifacts (for grading):
- `step5_model_selection/models/stacked_ensemble.pkl` (final selected model)
- `step5_model_selection/metrics/best_model_metrics.csv` (final metrics)
- `step4_eda/plots/` (EDA visualizations)
- `featureengineering_model_ready.xlsx` (model-ready dataset)

Notes on duplication
- Several experiment folders exist (e.g., `dual_dataset_steps_2_to_5/...`). The canonical, final artifacts are consolidated under `step5_model_selection/`. Use `AIRQI_PROJECT.md` and `COMPLETE_PROJECT_REPORT.txt` for pointers.
