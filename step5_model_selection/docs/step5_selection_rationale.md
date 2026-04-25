# Step 5 Model Selection Rationale

## Why Stacked_Ensemble?
- It achieved the best test R2 (0.8186).
- It also achieved the lowest MAE (19.6332) among tuned models.
- It achieved the lowest RMSE (28.8320) among tuned models.
- It combines strengths of multiple tree-based learners instead of depending on one model family.

## Why not LinearRegression?
- The data relationships are not purely linear.
- The baseline linear model produced a negative R2, which indicates poor fit.

## Why not RandomForest or GradientBoosting alone?
- They performed well, but not as well as the stacked ensemble on the test set.
- The ensemble reduced error further by combining complementary prediction patterns.

## Why these Step 5 visualizations?
- Model comparison chart: shows why one algorithm is selected over others.
- Actual vs predicted: shows alignment between predictions and real values.
- Residual distribution: shows whether errors are centered and reasonably controlled.
- Feature importance: explains which variables the model is relying on.
