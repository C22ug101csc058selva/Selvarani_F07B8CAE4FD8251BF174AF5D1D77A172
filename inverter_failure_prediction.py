
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
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
    'failure': np.random.choice([0, 1], size=1000, p=[0.92, 0.08])  # Imbalanced data
})

# Features and target
X = data.drop('failure', axis=1)
y = data['failure']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Address class imbalance with SMOTE
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_scaled, y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train model with hyperparameter tuning
params = {
    'n_estimators': [100, 150],
    'max_depth': [5, 10],
    'min_samples_split': [2, 5]
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), params, cv=3)
grid_search.fit(X_train, y_train)

# Predict and evaluate
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Predict on new sample
sample = pd.DataFrame({
    'temperature': [46],
    'voltage': [595],
    'current': [14],
    'power': [8700],
    'runtime_hrs': [2700]
})
sample_scaled = scaler.transform(sample)
sample_pred = best_model.predict(sample_scaled)
print("\nPredicted Inverter Status (1=Failure, 0=OK):", sample_pred[0])
