"""
Zynxis Intern Performance Predictor
A Streamlit app that serves the Week 3 Random Forest classification model.

Run with:
    streamlit run app.py
"""

import pandas as pd
import joblib
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
        background-color: #472F2B !important;
        border: 2px solid #ffffff !important;
        box-shadow: 0 0 8px rgba(71, 47, 43, 0.4) !important;
    }
    
    /* Slider Active Fill Track */
    div[data-baseweb="slider"] div[style*="background-color"] {
        background-color: #472F2B !important;
    }
    
    /* Button Styling */
    .stButton > button {
        background-color: #472F2B !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 12px rgba(71, 47, 43, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        background-color: #32201D !important;
        color: #ffffff !important;
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


@st.cache_resource
def load_artifacts():
    model = joblib.load("best_model_random_forest.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_names = joblib.load("feature_names.pkl")
    return model, scaler, feature_names


def build_input_row(inputs: dict, feature_names: list) -> pd.DataFrame:
    """Turn raw form inputs into a one-hot-encoded row matching training features."""
    row = {col: inputs[col] for col in NUM_COLS}
    row["Education_Level_High_School"] = 1 if inputs["Education_Level"] == "High_School" else 0
    row["Education_Level_Undergrad"] = 1 if inputs["Education_Level"] == "Undergrad" else 0
    
    df = pd.DataFrame([row])
    return df[feature_names]  

def main():
    import os
    if not (os.path.exists("best_model_random_forest.pkl") and os.path.exists("scaler.pkl") and os.path.exists("feature_names.pkl")):
        import train_model
        train_model.main()
        st.cache_resource.clear()

    try:
        model, scaler, feature_names = load_artifacts()
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        st.stop()

    st.title(" Zynxis Intern Performance Predictor")
    st.markdown(
        "Predict whether an intern is likely to be placed based on technical, project, and behavioral metrics.\n"
        
    )

    st.divider()
    st.subheader("Intern Metrics")

    col1, col2 = st.columns(2)

    with col1:
        technical_score = st.slider(
            "Technical Score", min_value=0.0, max_value=100.0, value=75.0, step=0.5,
            help="Score on technical coding assignments (0–100)",
        )
        project_completion = st.slider(
            "Project Completion Rate (%)", min_value=0.0, max_value=100.0, value=85.0, step=0.5,
        )
        attendance = st.slider(
            "Attendance & Punctuality (%)", min_value=0.0, max_value=100.0, value=90.0, step=0.5,
        )

    with col2:
        soft_skills = st.slider(
            "Soft Skills Rating", min_value=1.0, max_value=5.0, value=3.5, step=0.1,
        )
        code_review = st.slider(
            "Code Review Score", min_value=1.0, max_value=10.0, value=6.5, step=0.1,
        )
        prior_experience = st.slider(
            "Prior Experience (months)", min_value=0, max_value=24, value=3, step=1,
        )

    education_level = st.selectbox(
        "Education Level", options=["High_School", "Undergrad", "Grad"], index=1,
    )

    st.divider()

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
            if prediction == 1:
                st.success(f"Likely to be Placed")
            else:
                st.warning(f"Needs Support / Not Likely to be Placed")

            st.metric("Predicted Probability of High Performance", f"{probability:.1%}")
            st.progress(float(probability))

            with st.expander("What did the model see?"):
                st.dataframe(row, use_container_width=True)

        except Exception as e:
            st.error(f"Something went wrong while generating the prediction: {e}")

    st.divider()
    st.caption(
        "Built for Zynxis Week 7 — Model Deployment. Underlying model: "
        "RandomForestClassifier trained on 500 intern records ."
    )


if __name__ == "__main__":
    main()
