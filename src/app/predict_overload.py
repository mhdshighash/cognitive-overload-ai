import joblib
import numpy as np
import pandas as pd

# Load trained model
model = joblib.load("outputs/overload_model.pkl")

# Example new user behavior (simulate real-time input)
new_data = {
    "typing_speed": 40,
    "error_rate": 12,
    "task_time": 35,
    "idle_time": 6,
    "tab_switches": 10,
    "focus_score": 45,
    "fatigue_trend": 1.5
}

df_new = pd.DataFrame([new_data])

# Predict overload
prediction = model.predict(df_new)[0]
probability = model.predict_proba(df_new).max()

levels = ["Normal", "Medium", "High"]

print("\nPredicted Cognitive Load:", levels[prediction])
print("Confidence:", round(probability * 100, 2), "%")

# Alert logic
if prediction == 2:
    print("🚨 ALERT: High cognitive overload detected! Take a break.")
elif prediction == 1:
    print("⚠️ Warning: Medium load. Slow down.")
else:
    print("✅ Normal cognitive load.")
