# STEP 2 TO STEP 5 Integrity Verification (Rubric-Aligned)

## Verification Status

- Step 2 problem definition: PASS
- Step 3 processed full dataset: PASS
- Step 3 leakage-safe dataset: PASS
- Step 4 stats summary csv: PASS
- Step 4 correlation csv: PASS
- Step 4 target-correlation csv: PASS
- Step 4 plot folders (01-05): PASS
- Step 4 separate open-source dashboard: PASS
- Step 5 baseline comparison: PASS
- Step 5 advanced comparison: PASS
- Step 5 best metrics: PASS
- Step 5 predictions file: PASS
- Step 5 plots: PASS
- Step 5 saved models: PASS

## Rubric Alignment (from provided image)

- Step 2 (Problem understanding): objective, assumptions, and measurable target documented.
- Step 3 (Preparation/preprocessing): data cleaning, feature engineering, transformation pipeline documented and output files present.
- Step 4 (EDA): statistical analysis CSVs + labelled plot suite + insights file + separate open-source dashboard.
- Step 5 (Model selection/training): algorithm comparison, validation strategy, tuning, metrics, residual diagnostics, feature importance, and saved models.

## Step 5 Performance Snapshot

### Advanced best model (current reproducible run)
- Selected Model: Stacked_Ensemble
- R2 Score: 0.8186
- MAE: 19.6332
- RMSE: 28.8320

### Baseline best model snapshot
- Selected Model: GradientBoostingRegressor
- R2 Score: 0.7966
- MAE: 20.9875
- RMSE: 30.5370

## File References for Viva

- Step 4 notebook: `step4_eda/notebooks/step4_eda.ipynb`
- Step 5 notebook: `step5_model_selection/notebooks/step5_model_training.ipynb`
- Step 4 dashboard: `step4_eda/open_source_dashboard/streamlit_dashboard.py`
- Baseline comparison: `step5_model_selection/metrics/model_comparison.csv`
- Advanced comparison: `step5_model_selection/metrics/tuned_model_comparison.csv`
- Best metrics: `step5_model_selection/metrics/best_metrics.csv`