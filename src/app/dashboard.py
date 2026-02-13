import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime


st.set_page_config(page_title="Cognitive Overload Predictor")

model = joblib.load("outputs/overload_model.pkl")
scaler = joblib.load("outputs/scaler.pkl")

st.title("🧠 Cognitive Overload Prediction System")

st.sidebar.header("User Behavior Inputs")

#user details
st.subheader("👤 User Info")
user_name = st.text_input("Enter your name", value="Guest")


typing_speed = st.sidebar.slider("Typing Speed", 30, 120, 60)
error_rate = st.sidebar.slider("Error Rate (%)", 0, 20, 5)
task_time = st.sidebar.slider("Task Time (minutes)", 5, 60, 20)
idle_time = st.sidebar.slider("Idle Time (minutes)", 0, 15, 2)
tab_switches = st.sidebar.slider("Tab Switches", 0, 20, 4)
focus_score = st.sidebar.slider("Focus Score", 20, 100, 75)
fatigue_trend = st.sidebar.slider("Fatigue Trend", -3.0, 3.0, 0.0)

input_df = pd.DataFrame([{
    "typing_speed": typing_speed,
    "error_rate": error_rate,
    "task_time": task_time,
    "idle_time": idle_time,
    "tab_switches": tab_switches,
    "focus_score": focus_score,
    "fatigue_trend": fatigue_trend
}])

# SCALE INPUT
input_scaled = scaler.transform(input_df)

prediction = model.predict(input_scaled)[0]
confidence = model.predict_proba(input_scaled).max()

# -------- Save Session --------
save_session = st.button("💾 Save Session")

#--------- save session logic --------
if save_session:
    session_data = input_df.copy()
    session_data.insert(0, "user_name", user_name)

    session_data["prediction"] = prediction
    session_data["confidence"] = confidence
    session_data["timestamp"] = datetime.now()

    file_path = "data/user_sessions.csv"

    if os.path.exists(file_path):
        session_data.to_csv(file_path, mode="a", header=False, index=False)
    else:
        session_data.to_csv(file_path, index=False)

    st.success("Session saved successfully!")


#gauge for confidence
st.subheader("📊 Prediction Confidence")

percent = int(confidence * 100)

st.metric(label="AI Confidence", value=f"{percent}%")

st.progress(confidence)

if percent > 80:
    st.success("Very confident prediction")
elif percent > 60:
    st.warning("Moderate confidence")
else:
    st.error("Low confidence — behavior unclear")

#levels
levels = ["Normal", "Medium", "High"]

st.subheader("Prediction Result")
st.write("### Cognitive Load:", levels[prediction])
st.write("### Confidence:", f"{confidence*100:.2f}%")

if prediction == 2:
    st.error("🚨 High overload detected! Take a break.")
elif prediction == 1:
    st.warning("⚠️ Medium load — slow down.")
else:
    st.success("✅ Normal cognitive load.")


#new feature importance
st.subheader("🔍 Feature Importance (Why overload happens)")

importance_df = pd.read_csv("outputs/feature_importance.csv")

st.bar_chart(
    importance_df.set_index("feature")["importance"]
)