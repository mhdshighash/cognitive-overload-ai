import numpy as np
import pandas as pd

np.random.seed(42)
n = 1500

typing_speed = np.random.uniform(30, 120, n)
error_rate = np.random.uniform(0, 20, n)
task_time = np.random.uniform(5, 60, n)
idle_time = np.random.uniform(0, 15, n)
tab_switches = np.random.randint(0, 20, n)
focus_score = np.random.uniform(20, 100, n)
fatigue_trend = np.random.uniform(-3, 3, n)

overload_score = (
    0.5 * error_rate +
    0.4 * task_time +
    0.6 * idle_time +
    0.4 * tab_switches -
    0.4 * typing_speed -
    0.6 * focus_score +
    2 * fatigue_trend
)

# Percentile thresholds (auto balanced)
low = np.percentile(overload_score, 33)
high = np.percentile(overload_score, 66)

overload_level = []

for score in overload_score:
    if score < low:
        overload_level.append(0)
    elif score < high:
        overload_level.append(1)
    else:
        overload_level.append(2)

df = pd.DataFrame({
    "typing_speed": typing_speed,
    "error_rate": error_rate,
    "task_time": task_time,
    "idle_time": idle_time,
    "tab_switches": tab_switches,
    "focus_score": focus_score,
    "fatigue_trend": fatigue_trend,
    "overload_level": overload_level
})

print("\nClass distribution:")
print(df["overload_level"].value_counts())

df.to_csv("data/cognitive_overload_data.csv", index=False)
