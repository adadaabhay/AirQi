# Step 5 Modeling Notes

- Problem type: Regression (target: Pm2.5).
- Split: train_test_split(test_size=0.2, random_state=42).
- Validation: 5-fold CV on train split.
- Baselines: LinearRegression, RandomForestRegressor, GradientBoostingRegressor.
- Advanced: RF tuned, ET tuned, GB tuned, Stacked Ensemble.
- Best advanced model: Stacked_Ensemble.
- Best advanced test R2: 0.8186.
- Best advanced test MAE: 19.6332.
- Best advanced test RMSE: 28.8320.
