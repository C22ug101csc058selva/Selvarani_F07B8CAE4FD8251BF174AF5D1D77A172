
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Generate synthetic data
np.random.seed(42)
data = pd.DataFrame({
    'temperature': np.random.normal(40, 5, 1000),
    'voltage': np.random.normal(600, 15, 1000),
    'current': np.random.normal(15, 2, 1000),
    'power': np.random.normal(9000, 300, 1000),
    'runtime_hrs': np.random.normal(2500, 800, 1000),
    'failure': np.random.choice([0, 1], size=1000, p=[0.92, 0.08])
})

# Preprocessing
X = data.drop('failure', axis=1)
y = data['failure']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_scaled, y)

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("Model Evaluation:\n", classification_report(y_test, y_pred))

# Predict on new data
new_data = pd.DataFrame({
    'temperature': [46],
    'voltage': [595],
    'current': [16],
    'power': [8600],
    'runtime_hrs': [2800]
})
new_scaled = scaler.transform(new_data)
prediction = model.predict(new_scaled)[0]

print("\nPredicted Failure Status:", "Failure" if prediction == 1 else "OK")
