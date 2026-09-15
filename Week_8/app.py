"""
Zynxis Intern Performance Predictor — Capstone Edition
A Streamlit app that serves the Random Forest classification model, with
single-intern prediction + explanation, batch CSV prediction, a model
comparison view, and feature importance.

Run with:
    streamlit run app.py
"""

import pandas as pd
import numpy as np
import joblib
import altair as alt
import streamlit as st

st.set_page_config(
    page_title="Zynxis Intern Performance Predictor",
    page_icon="⚡",
    layout="centered",
)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Alegreya:wght@700;800;900&display=swap');

    :root {
        --accent-color: #722F37;
    }
    
    /* App Background */
    .stApp, [data-testid="stAppViewContainer"], .main {
        background-color: #B0A999 !important;
    }
    
    /* Headings in Alegreya Font (Extra Bold) */
    h1, h2, h3, h4, h5, h6, 
    [data-testid="stHeader"], 
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
    div[data-testid="stSubheader"] {
        font-family: 'Alegreya', serif !important;
        font-weight: 900 !important;
        color: #595E48 !important;
        letter-spacing: -0.5px;
    }
    
    /* General Labels & Text */
    p, label, .stMarkdown, div[data-testid="stWidgetLabel"] p {
        color: #071A35 !important;
    }
    
    /* Slider Handle (Thumb) */
    div[data-baseweb="slider"] div[role="slider"] {
        background-color: #722F37 !important;
        border: 4px solid #ffffff !important;
        box-shadow: 0 0 8px rgba(71, 47, 43, 0.4) !important;
    }
    
    /* Slider Active Fill Track */
    div[data-baseweb="slider"] div[style*="background-color"] {
        background-color: #722F37 !important;
    }
    
    /* Button Styling */
    .stButton > button {
        background-color: #722F37 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 12px rgba(71, 47, 43, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        background-color: #722F37 !important;
        color: #B2BEB5 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 18px rgba(71, 47, 43, 0.5) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,

)
NUM_COLS = [
    "Technical_Score",
    "Project_Completion_Rate",
    "Attendance_Punctuality",
    "Soft_Skills_Rating",
    "Code_Review_Score",
    "Prior_Experience_Months",
]
REQUIRED_RAW_COLS = NUM_COLS + ["Education_Level"]
BLUE_SHADES = [
    "#0B2D5C", "#174A7C", "#2166A5", "#2F80C0",
    "#5B9BD5", "#9DC3E6", "#D6E5F5",
]


def style_table(df: pd.DataFrame):
    return df.style.set_properties(
        **{"background-color": "#B2B7BD", "color": "#071A35"}
    ).set_table_styles([
        {
            "selector": "th",
            "props": [
                ("background-color", "#C9CED4"),
                ("color", "#071A35"),
                ("font-weight", "bold"),
            ],
        }
    ])


def render_bar_chart(data: pd.DataFrame, x: str, y: str, color: str):
    chart = (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X(x, sort="-y", title=None),
            y=alt.Y(y, title=None),
            color=alt.Color(
                color,
                scale=alt.Scale(range=BLUE_SHADES),
                legend=None,
            ),
            tooltip=[x, y],
        )
        .properties(height=340)
    )
    st.altair_chart(chart, use_container_width=True)


@st.cache_resource
def load_artifacts():
    model = joblib.load("best_model_random_forest.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_names = joblib.load("feature_names.pkl")
    return model, scaler, feature_names


@st.cache_resource
def load_comparison_models():
    """Returns None values gracefully if comparison artifacts aren't present."""
    try:
        logreg = joblib.load("model_logistic_regression.pkl")
        dtree = joblib.load("model_decision_tree.pkl")
        return logreg, dtree
    except FileNotFoundError:
        return None, None


@st.cache_data
def load_comparison_table():
    try:
        return pd.read_csv("model_comparison.csv")
    except FileNotFoundError:
        return None


def build_input_row(inputs: dict, feature_names: list) -> pd.DataFrame:
    """Turn raw form inputs into a one-hot-encoded row matching training features."""
    row = {col: inputs[col] for col in NUM_COLS}
    row["Education_Level_High_School"] = 1 if inputs["Education_Level"] == "High_School" else 0
    row["Education_Level_Undergrad"] = 1 if inputs["Education_Level"] == "Undergrad" else 0
    df = pd.DataFrame([row])
    return df[feature_names]


def build_batch_rows(df_raw: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    """Encode a batch dataframe of raw columns into the model's expected feature matrix."""
    missing = [c for c in REQUIRED_RAW_COLS if c not in df_raw.columns]
    if missing:
        raise ValueError(f"CSV is missing required column(s): {', '.join(missing)}")

    df = df_raw.copy()
    df["Education_Level_High_School"] = (df["Education_Level"] == "High_School").astype(int)
    df["Education_Level_Undergrad"] = (df["Education_Level"] == "Undergrad").astype(int)
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    return df[feature_names]


def explain_prediction(row_raw: dict, model, feature_names: list):
    """Simple, readable explanation: compares the intern's values against dataset-typical
    ranges and surfaces which top-importance features pushed the prediction up or down."""
    importances = pd.Series(model.feature_importances_, index=feature_names)
    top_features = importances.sort_values(ascending=False).head(4).index.tolist()

   
    reference = {
        "Technical_Score": 75, "Project_Completion_Rate": 85, "Attendance_Punctuality": 88,
        "Soft_Skills_Rating": 3.4, "Code_Review_Score": 6.7, "Prior_Experience_Months": 4,
    }

    notes = []
    for feat in top_features:
        if feat not in reference:
            continue
        val = row_raw.get(feat)
        ref = reference[feat]
        if val is None:
            continue
        diff_pct = (val - ref) / ref if ref else 0
        label = feat.replace("_", " ")
        if diff_pct > 0.08:
            notes.append(f"✅ **{label}** ({val}) is above the typical range ({ref}) — a strong positive signal.")
        elif diff_pct < -0.08:
            notes.append(f"⚠️ **{label}** ({val}) is below the typical range ({ref}) — this pulls the prediction down.")
        else:
            notes.append(f"➖ **{label}** ({val}) is close to typical ({ref}) — a neutral signal.")
    return notes


def render_verdict(prediction, probability):
    if prediction == 1:
        st.success(" Likely to be Placed")
    else:
        st.warning(" Not Likely to be Placed")
    st.metric("Predicted Probability of High Performance", f"{probability:.1%}")
    st.progress(float(probability))


def tab_predict(model, scaler, feature_names):
    st.subheader("Intern Metrics")
    col1, col2 = st.columns(2)

    with col1:
        technical_score = st.slider("Technical Score", 0.0, 100.0, 75.0, 0.5,
                                     help="Score on technical coding assignments (0–100)")
        project_completion = st.slider("Project Completion Rate (%)", 0.0, 100.0, 85.0, 0.5)
        attendance = st.slider("Attendance & Punctuality (%)", 0.0, 100.0, 90.0, 0.5)

    with col2:
        soft_skills = st.slider("Soft Skills Rating", 1.0, 5.0, 3.5, 0.1)
        code_review = st.slider("Code Review Score", 1.0, 10.0, 6.5, 0.1)
        prior_experience = st.slider("Prior Experience (months)", 0, 24, 3, 1)

    education_level = st.selectbox("Education Level", ["High_School", "Undergrad", "Grad"], index=1)

    if st.button("Predict", type="primary", use_container_width=True):
        inputs = {
            "Technical_Score": technical_score,
            "Project_Completion_Rate": project_completion,
            "Attendance_Punctuality": attendance,
            "Soft_Skills_Rating": soft_skills,
            "Code_Review_Score": code_review,
            "Prior_Experience_Months": prior_experience,
            "Education_Level": education_level,
        }
        try:
            row = build_input_row(inputs, feature_names)
            row_scaled = row.copy()
            row_scaled[NUM_COLS] = scaler.transform(row[NUM_COLS])

            prediction = model.predict(row_scaled)[0]
            probability = model.predict_proba(row_scaled)[0][1]

            st.subheader("Result")
            render_verdict(prediction, probability)

            st.subheader("Why this prediction?")
            for note in explain_prediction(inputs, model, feature_names):
                st.markdown(note)

            with st.expander("What did the model see?"):
                st.dataframe(style_table(row), use_container_width=True)

        except Exception as e:
            st.error(f"Something went wrong while generating the prediction: {e}")


def tab_batch(model, scaler, feature_names):
    st.subheader("Batch Prediction")
    st.markdown(
        "Upload a CSV with one row per intern to screen a whole cohort at once. "
        "Required columns: `Technical_Score`, `Project_Completion_Rate`, "
        "`Attendance_Punctuality`, `Soft_Skills_Rating`, `Code_Review_Score`, "
        "`Prior_Experience_Months`, `Education_Level`."
    )

    template = pd.DataFrame([{
        "Technical_Score": 82.5, "Project_Completion_Rate": 90.0, "Attendance_Punctuality": 95.0,
        "Soft_Skills_Rating": 4.0, "Code_Review_Score": 7.5, "Prior_Experience_Months": 5,
        "Education_Level": "Undergrad",
    }])
    st.download_button(
        "Download CSV template", template.to_csv(index=False),
        file_name="intern_batch_template.csv", mime="text/csv",
    )

    uploaded = st.file_uploader("Upload intern CSV", type=["csv"])
    if uploaded is not None:
        try:
            df_raw = pd.read_csv(uploaded)
            X = build_batch_rows(df_raw, feature_names)
            X_scaled = X.copy()
            X_scaled[NUM_COLS] = scaler.transform(X[NUM_COLS])

            preds = model.predict(X_scaled)
            probs = model.predict_proba(X_scaled)[:, 1]

            results = df_raw.copy()
            results["Predicted_Outcome"] = np.where(preds == 1, "High Performer", "Needs Support")
            results["Probability"] = (probs * 100).round(1)

            st.success(f"Scored {len(results)} interns.")
            st.dataframe(style_table(results), use_container_width=True)

            st.download_button(
                "Download results CSV", results.to_csv(index=False),
                file_name="intern_predictions.csv", mime="text/csv",
            )

            col1, col2 = st.columns(2)
            col1.metric("High Performers", int((preds == 1).sum()))
            col2.metric("Needs Support", int((preds == 0).sum()))

        except Exception as e:
            st.error(f"Couldn't process that file: {e}")


def tab_comparison():
    st.subheader("Model Comparison")
    comp = load_comparison_table()
    if comp is None:
        st.info("Run `python train_model.py` to generate the comparison table (model_comparison.csv).")
        return

    st.markdown(
        "Three models were trained and evaluated on the same held-out test set. "
        "**Random Forest** is the model deployed in this app."
    )
    display = comp.copy()
    for c in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        display[c] = (display[c] * 100).round(1).astype(str) + "%"
    display.columns = ["Model", "Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    st.dataframe(style_table(display), use_container_width=True, hide_index=True)

    chart_data = comp.melt(
        id_vars="model", value_vars=["accuracy", "f1", "roc_auc"],
        var_name="Metric", value_name="Score",
    )
    render_bar_chart(chart_data, "model", "Score", "Metric")
    st.caption(
        "Random Forest was selected as the deployed model for its balance of accuracy, F1, "
        "and interpretability via feature importance — see the Feature Importance tab."
    )


def tab_feature_importance(model, feature_names):
    st.subheader("Feature Importance")
    st.markdown("What the deployed Random Forest model weighs most heavily when making a prediction.")

    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=True)
    display_names = pd.Series(
        [f.replace("_", " ") for f in importances.index], index=importances.index
    )
    chart_df = pd.DataFrame({"Feature": display_names.values, "Importance": importances.values})
    render_bar_chart(chart_df, "Feature", "Importance", "Feature")

    top = importances.sort_values(ascending=False)
    st.markdown(
        f"**{top.index[0].replace('_', ' ')}** and **{top.index[1].replace('_', ' ')}** are the two most "
        f"influential features, together accounting for **{(top.iloc[0] + top.iloc[1]):.0%}** of the model's "
        "total decision weight. Education Level barely registers — demonstrated performance matters far "
        "more than formal credentials in this model."
    )


def main():
    try:
        model, scaler, feature_names = load_artifacts()
    except FileNotFoundError:
        st.error(
            "Model artifacts not found. Run `python train_model.py` first to "
            "generate best_model_random_forest.pkl and scaler.pkl."
        )
        st.stop()

    st.title("📊 Zynxis Intern Performance Predictor")
    st.markdown(
        "Predict whether an intern is likely to be a **high performer / placed** "
        "based on technical, project, and behavioral metrics.\n\n"
        "*Model: Random Forest Classifier — 69% test accuracy, F1 = 0.75*"
    )
    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(
        ["🔮 Predict", "📁 Batch Prediction", "⚖️ Model Comparison", "📈 Feature Importance"]
    )
    with tab1:
        tab_predict(model, scaler, feature_names)
    with tab2:
        tab_batch(model, scaler, feature_names)
    with tab3:
        tab_comparison()
    with tab4:
        tab_feature_importance(model, feature_names)

    st.divider()
    st.caption(
        "Built for Zynxis Week 8 — Final Capstone. Underlying model: "
        "RandomForestClassifier trained on 500 intern records."
    )


if __name__ == "__main__":
    main()
