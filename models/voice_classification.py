import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# Loading the SVD features
processed_dir = r"C:\Users\jagde\PycharmProjects\AIHealthcareAssistant\data\voice\processed_features"
X = np.load(os.path.join(processed_dir, "svd_features.npy"))
y = np.load(os.path.join(processed_dir, "svd_labels.npy"))

print(f"Data Loaded: {X.shape[0]} samples, {X.shape[1]} features")

mask = np.isfinite(X).all(axis=1)
X, y = X[mask], y[mask]

# Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95)),
    ('svm', SVC(probability=True, class_weight='balanced'))
])

param_grid = {
    'svm__C': [0.1, 1, 10, 100],          # Regularization strength
    'svm__gamma': ['scale', 'auto', 0.01], # Kernel coefficient
    'svm__kernel': ['rbf', 'linear']      # Boundary type
}

print("Running 5-Fold CV Hyperparameter Tuning...")
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=cv_strategy,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

best_idx = grid_search.best_index_
cv_mean = grid_search.cv_results_['mean_test_score'][best_idx]
cv_std = grid_search.cv_results_['std_test_score'][best_idx]


y_pred = best_model.predict(X_test)
y_probs = best_model.predict_proba(X_test)

current_loss = log_loss(y_test, y_probs)

print(f"\nBest Parameters found: {grid_search.best_params_}")
print(f"Mean CV Accuracy: {cv_mean:.4f}")
print(f"CV Standard Deviation (SD): {cv_std:.4f}")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Log-Loss (Error Metric): {current_loss:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Anomaly", "Healthy"]))

save_path = r"C:\Users\jagde\PycharmProjects\AIHealthcareAssistant\models\saved_models"
os.makedirs(save_path, exist_ok=True)

joblib.dump(best_model, os.path.join(save_path, "voice_full_pipeline.pkl"))
print(f"\nOptimized Pipeline saved to {save_path}")
