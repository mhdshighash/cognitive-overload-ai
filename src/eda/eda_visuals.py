import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/cleaned_data.csv")

# Overload distribution
plt.figure(figsize=(5,4))
df["overload_level"].value_counts().plot(kind="bar")
plt.title("Overload Level Distribution")
plt.xlabel("Level")
plt.ylabel("Count")
plt.show()

# Typing speed vs overload
plt.figure(figsize=(5,4))
plt.scatter(df["typing_speed"], df["overload_level"])
plt.xlabel("Typing Speed")
plt.ylabel("Overload Level")
plt.title("Typing Speed vs Overload")
plt.show()

# Error rate vs overload
plt.figure(figsize=(5,4))
plt.scatter(df["error_rate"], df["overload_level"])
plt.xlabel("Error Rate")
plt.ylabel("Overload Level")
plt.title("Error Rate vs Overload")
plt.show()
