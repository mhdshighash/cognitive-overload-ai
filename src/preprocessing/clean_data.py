import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

df = pd.read_csv("data/cognitive_overload_data.csv")

X = df.drop("overload_level", axis=1)
y = df["overload_level"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

df_scaled = pd.DataFrame(X_scaled, columns=X.columns)
df_scaled["overload_level"] = y

# Save scaler
joblib.dump(scaler, "outputs/scaler.pkl")

df_scaled.to_csv("data/cleaned_data.csv", index=False)

print("Cleaned data + scaler saved!")
