# Zynxis Intern Performance Predictor — Week 8 Final Capstone

A complete end-to-end machine learning project for Zynxis: data preparation,
model training, evaluation, and a deployed Streamlit app that predicts
whether an intern will be a high performer / get placed, based on
technical, project, and behavioral metrics.

This capstone builds directly on the Week 3 model and Week 7 deployment.

## Model

- **Algorithm**: Random Forest Classifier (`n_estimators=100`, `max_depth=7`, `min_samples_split=4`)
- **Source**: Week 3 notebook (`classification_model.ipynb`)
- **Test performance**: 69% accuracy, F1 = 0.75, ROC-AUC = 0.71
- **Features**: Technical_Score, Project_Completion_Rate, Attendance_Punctuality,
  Soft_Skills_Rating, Code_Review_Score, Prior_Experience_Months, Education_Level
  (one-hot encoded: High_School / Undergrad / Grad)

## Project Structure

```
capstone/
├── app.py                              # Streamlit app: predict, batch, comparison, importance
├── train_model.py                      # Data prep + trains RF (deployed) + LogReg/DTree (comparison)
├── eda.py                              # Exploratory data analysis, generates report visuals
├── evaluate.py                         # Model evaluation, generates metrics + report visuals
├── zynxis_intern_performance.csv       # Training data (500 rows)
├── classification_model.ipynb          # Original Week 3 notebook
├── Zynxis_Capstone_Report.pdf          # Full project report
├── Zynxis_Capstone_Slides.pptx         # 5-minute presentation deck
├── requirements.txt
├── README.md
├── best_model_random_forest.pkl        # Deployed model (pre-generated, included)
├── model_logistic_regression.pkl       # Comparison model (pre-generated, included)
├── model_decision_tree.pkl             # Comparison model (pre-generated, included)
├── model_comparison.csv                # Side-by-side metrics for all 3 models
├── scaler.pkl                          # Fitted StandardScaler (pre-generated, included)
├── feature_names.pkl                   # Column order used at train time
├── evaluation_metrics.txt              # Plain-text metrics summary
└── assets_*.png                        # Charts used in the report/slides (EDA + evaluation)
```

## Setup & Run Locally

```bash
# 1. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Regenerate everything from scratch — the .pkl files and
#    charts are already included, so you can skip straight to step 4.
python eda.py             # exploratory data analysis + charts
python train_model.py     # trains the model, saves .pkl artifacts
python evaluate.py        # test-set metrics + evaluation charts

# 4. Launch the app
streamlit run app.py
```

The app will open at `http://localhost:8501`.



## What the App Does

The app has four tabs:

1. **🔮 Predict** — sliders/dropdown for all 7 inputs; on "Predict", shows a clear
   High Performer / Needs Support verdict, the predicted probability with a progress
   bar, a plain-language "Why this prediction?" explanation (which of the model's
   top features pushed the result up or down, compared to typical values), and an
   expandable view of the exact feature row sent to the model.
2. ** Batch Prediction** — upload a CSV of multiple interns (a template is provided)
   and get predictions + probabilities for the whole cohort at once, with a
   downloadable results CSV and a High Performer / Needs Support count summary.
3. ** Model Comparison** — Random Forest vs. Logistic Regression vs. Decision Tree,
   all trained on the identical data split, with accuracy/precision/recall/F1/ROC-AUC
   side by side and a chart, plus a note on why Random Forest was selected for deployment.
4. ** Feature Importance** — an interactive bar chart of what the deployed model
   weighs most heavily, with a plain-language takeaway.

The app handles missing model artifacts and malformed uploads gracefully with
on-screen errors instead of crashing.
