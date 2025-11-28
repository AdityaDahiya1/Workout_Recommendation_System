import streamlit as st
import pandas as pd
import joblib

from main_ml_script import (
    load_data as base_load_data,
    train_level_models,
    train_rating_regressor,
    recommend_exercises,
)

# ------------------------------------
# STREAMLIT CONFIG & CACHED DATA
# ------------------------------------
st.set_page_config(page_title="Workout Recommendation System", page_icon="💪", layout="wide")

@st.cache_data
def load_data(path: str = "megaGymDataset.csv") -> pd.DataFrame:
    # Wrap the ML module's load_data with Streamlit cache
    return base_load_data(path)

# ------------------------------------
# CUSTOM PAGE CSS STYLING
# ------------------------------------
st.markdown("""
<style>
/* Background Gradient */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    color: white;
}

/* Title Styling */
.title {
    font-size: 50px;
    font-weight: 900;
    text-align: center;
    color: #00eaff;
    text-shadow: 2px 2px 4px #000;
}

/* Card style */
.block {
    background-color: rgba(255,255,255,0.1);
    padding: 25px;
    border-radius: 15px;
    margin: 10px auto;
    border: 1px solid #00eaff;
    backdrop-filter: blur(10px);
}

/* Button styling */
div.stButton > button {
    background-color: #00eaff;
    color: black;
    border-radius: 8px;
    font-size: 18px;
    height: 50px;
    width: 250px;
}

/* Download button */
.stDownloadButton {
    background-color: #00eaff !important;
    color: black !important;
    border-radius: 6px;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------
# TITLE
# ------------------------------------
st.markdown('<p class="title">🏋 Workout Recommendation System | ML Powered 💪</p>', unsafe_allow_html=True)
st.write("### Personalized AI-Based Workout Suggestions Based on Body Target, Equipment & Fitness Level")

# Load data
df = load_data("megaGymDataset.csv")

# ------------------------------------
# SIDEBAR: TRAIN MODELS
# ------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1047/1047711.png", width=120)
    st.header("⚙ System Controls")

    if st.button("⚡ Train Models"):
        with st.spinner("Training ML Models... please wait ⏳"):
            pipelines, accuracies, best_model_name, label_enc = train_level_models(df)
            rating_reg = train_rating_regressor(df)

            # Save models
            joblib.dump(pipelines, "pipelines.pkl")
            joblib.dump(rating_reg, "rating_reg.pkl")

            st.success(f"✔ Training Completed — Best Model: {best_model_name}")
            st.write("Model Accuracies:")
            st.json({k: float(v) for k, v in accuracies.items()})

# ------------------------------------
# USER INPUT CARD
# ------------------------------------
st.markdown("### 🎯 Select Your Preferences")
with st.container():
    body_part = st.selectbox("💪 Body Part", ["Any"] + sorted(df["BodyPart"].dropna().unique().tolist()))
    equipment = st.selectbox("🏋 Equipment", ["Any"] + sorted(df["Equipment"].dropna().unique().tolist()))
    level = st.selectbox("📶 Level", ["Any"] + sorted(df["Level"].dropna().unique().tolist()))
    top_n = st.slider("📌 Top-N Recommendations", 1, 10, 5)

if st.button("🔥 Get Recommendations"):
    try:
        rating_reg = joblib.load("rating_reg.pkl")
    except:
        st.warning("Models not found on disk, training regression model now...")
        rating_reg = train_rating_regressor(df)

    results = recommend_exercises(df, rating_reg, body_part, equipment, level, top_n)
    st.markdown("## 🏆 Top Recommended Exercises")
    st.dataframe(results, use_container_width=True)

    if results is not None and not results.empty:
        csv = results.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇ Download CSV",
            data=csv,
            file_name="recommended_exercises.csv",
            mime="text/csv"
        )

# Footer
st.markdown("""
---
### 👨‍💻 Developed By Students of *Bennett University*
*Aditya | Prince Kumar | Harsh*  
Guide: *Tinku Sir*
""")