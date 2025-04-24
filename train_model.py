import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# Generate synthetic data
np.random.seed(42)
n_samples = 200

df = pd.DataFrame({
    "task_complexity": np.random.randint(1, 6, n_samples),
    "team_size": np.random.randint(1, 10, n_samples),
    "effective_hours": np.random.uniform(2.0, 50.0, n_samples),
    "experience": np.random.randint(0, 10, n_samples)
})

# Simulate cost
df["cost"] = (
    df["task_complexity"] * 100 +
    df["team_size"] * 50 +
    df["effective_hours"] * 20 +
    df["experience"] * 30 +
    np.random.normal(0, 50, n_samples)  # noise
)

# Train model
X = df[["task_complexity", "team_size", "effective_hours", "experience"]]
y = df["cost"]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Save the model
joblib.dump(model, "model_cost.pkl")
print("✅ Model saved as model_cost.pkl")
