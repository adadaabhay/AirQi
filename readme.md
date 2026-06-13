# AIRQI

AIRQI is a professional end-to-end air-quality analytics project organized as a five-step workflow: data collection, problem definition, preprocessing, exploratory data analysis, and model selection. The final objective is to predict `PM2.5` concentration from pollutant and location-derived features using regression models, supported by reproducible notebooks, exported metrics, plots, and an open-source dashboard. 

## Project Goal

The project answers a practical air-quality question:

- given pollutant observations and engineered features, can we estimate `PM2.5` reliably?

This matters because `PM2.5` is one of the most important air-pollution indicators for public-health monitoring and environmental decision support.

## Five-Step Flow

### Step 1. Data Collection and Understanding

Folder: `step1_data_collection/`

What happens here:
- AQI data is collected from government sources using scraper scripts.
- Raw CSV files are stored and documented.
- The original CPCBCCR scraping logic is preserved for traceability.

Main contents:
- raw data files
- acquisition scripts
- notebook for step walkthrough

### Step 2. Problem Understanding and Objective Definition

Folder: `step2_data_prep/`

What happens here:
- the prediction problem is defined
- the target variable is identified as `PM2.5`
- assumptions and project scope are documented

Main contents:
- problem definition notes
- data preparation scripts
- notebook for step walkthrough

### Step 3. Data Preparation and Preprocessing

Folder: `step3_data_preprocessing/`

What happens here:
- raw records are cleaned
- missing values are handled
- features are engineered
- a model-ready dataset is created

Main contents:
- processed datasets
- preprocessing summary
- notebook for step walkthrough

### Step 4. Exploratory Data Analysis

Folder: `step4_eda/`

What happens here:
- summary statistics are computed
- correlations are studied
- explanatory visualizations are generated
- an open-source dashboard version of the EDA is provided

Main contents:
- statistical CSV outputs
- plot exports
- notebook for detailed EDA explanation
- open-source visualization dashboard

### Step 5. Model Selection and Training

Folder: `step5_model_selection/`

What happens here:
- baseline models are trained
- tuned models are compared
- the best model is selected using test metrics
- final model artifacts, plots, and a professional dashboard are produced

Main contents:
- metrics CSV files
- trained model files
- performance plots
- notebook for Step 5 explanation
- open-source dashboard for review/presentation

## Final Model Result

The final selected model is:

- `Stacked_Ensemble`

Current saved performance:

- `R2 = 0.8186`
- `MAE = 19.6332`
- `RMSE = 28.8320`

Interpretation:
- the model explains about 81.86% of the variance in `PM2.5`
- prediction error remains measurable, so it should be used as a decision-support model, not as a perfect replacement for sensors

## Repository Structure

```text
step1_data_collection/
step2_data_prep/
step3_data_preprocessing/
step4_eda/
step5_model_selection/
```

## Open-Source Visualization

This project includes open-source dashboards for presentation and grading:

- Step 4 EDA dashboard: `step4_eda/open_source_dashboard/`
- Step 5 model dashboard: `step5_model_selection/open_source_dashboard/`

The Step 5 dashboard uses:

- `Streamlit`
- `Plotly`

This is a good fit because the project needs model comparison, residual analysis, actual-vs-predicted inspection, and feature-importance visualization rather than only BI-style reporting.

## How to Run

Create and activate the environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -U pip
pip install pandas numpy matplotlib seaborn scikit-learn plotly streamlit joblib openpyxl
```

Run the main Step 4 and Step 5 pipelines:

```bash
python step4_eda/scripts/run_step4_eda.py
python step5_model_selection/scripts/run_step5_modeling.py
```

Run the Step 5 dashboard:

```bash
streamlit run step5_model_selection/open_source_dashboard/streamlit_model_dashboard.py
```

## What the Model Can Do

Right now, the model can:

- estimate `PM2.5` from prepared AQI feature inputs
- support pollution-level analysis
- support review dashboards and academic reporting
- highlight which features most influence PM2.5

It should be described as:

- a predictive support model for air-quality analysis
- not a medical diagnosis model
- not a perfect substitute for direct field measurement

## Notes

- The repository includes only the main five-step deliverables needed for presentation, review, and grading.
- The project is organized to make each step reviewable independently.
- Step 5 is especially presentation-ready through its notebook, metrics, plots, and dashboard.

## License

This repository includes original CPCBCCR scraping references and derived project work. See [LICENSE](LICENSE) for the base license details.
