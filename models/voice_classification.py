import os
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, learning_curve, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             log_loss, roc_curve, auc)
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# Directory and paths setup
results_dir = r"C:\Users\jagde\PycharmProjects\AIHealthcareAssistant\models\voice_results"
os.makedirs(results_dir, exist_ok=True)

processed_dir = r"C:\Users\jagde\PycharmProjects\AIHealthcareAssistant\data\voice\processed_features"
X = np.load(os.path.join(processed_dir, "svd_features.npy"))
y = np.load(os.path.join(processed_dir, "svd_labels.npy"))

mask = np.isfinite(X).all(axis=1)
X, y = X[mask], y[mask]

print(f"Data Loaded: {X.shape[0]} samples. Saving visuals to: {results_dir}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Grid search on hyperparameter
print("\n--- Running Grid Search ---")
param_grid = {'svm__C': [0.1, 1, 10, 50, 100], 'svm__gamma': ['scale', 'auto', 0.01], 'svm__kernel': ['rbf', 'linear']}
pipeline = Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components=0.95)),
                     ('svm', SVC(probability=True, class_weight='balanced', random_state=42))])

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(pipeline, param_grid, cv=cv_strategy, scoring='neg_log_loss', return_train_score=True, n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
results_df = pd.DataFrame(grid_search.cv_results_)
best_params = grid_search.best_params_

# Performance dashboard for CM, ROC, F1
print("Generating Performance Summary...")
y_pred = best_model.predict(X_test)
y_probs = best_model.predict_proba(X_test)

plt.figure(figsize=(18, 5))
# Confusion Matrix
plt.subplot(1, 3, 1)
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', xticklabels=["Anomaly", "Healthy"], yticklabels=["Anomaly", "Healthy"])
plt.title("Confusion Matrix")

# ROC Curve
plt.subplot(1, 3, 2)
fpr, tpr, _ = roc_curve(y_test, y_probs[:, 1])
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {auc(fpr, tpr):.2f}')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.title("ROC Curve"); plt.legend()

# F1 Chart
report_dict = classification_report(y_test, y_pred, target_names=["Anomaly", "Healthy"], output_dict=True)
plt.subplot(1, 3, 3)
plt.bar(['Anomaly', 'Healthy'], [report_dict['Anomaly']['f1-score'], report_dict['Healthy']['f1-score']], color=['#e74c3c', '#2ecc71'])
plt.title("F1-Score Comparison")

plt.tight_layout()
plt.savefig(os.path.join(results_dir, "performance_summary.png"))
plt.show()

# Tuning Curves
print("Generating Tuning Curves...")
plt.figure(figsize=(18, 5))
c_sub = results_df[(results_df['param_svm__gamma'] == best_params['svm__gamma']) & (results_df['param_svm__kernel'] == best_params['svm__kernel'])]
plt.subplot(1, 3, 1)
plt.plot(c_sub['param_svm__C'].astype(float), -c_sub['mean_train_score'], 'g-o', label='Train')
plt.plot(c_sub['param_svm__C'].astype(float), -c_sub['mean_test_score'], 'r--x', label='Val')
plt.xscale('log'); plt.title("Log Loss vs C"); plt.ylabel("Log Loss"); plt.legend()

gamma_sub = results_df[(results_df['param_svm__C'] == best_params['svm__C']) & (results_df['param_svm__kernel'] == best_params['svm__kernel'])]
plt.subplot(1, 3, 2)
plt.plot(range(len(gamma_sub)), -gamma_sub['mean_test_score'], 'b-s')
plt.xticks(range(len(gamma_sub)), gamma_sub['param_svm__gamma'])
plt.title("Log Loss vs Gamma"); plt.ylabel("Log Loss")

kernel_sub = results_df[(results_df['param_svm__C'] == best_params['svm__C']) & (results_df['param_svm__gamma'] == best_params['svm__gamma'])]
plt.subplot(1, 3, 3)
plt.bar(kernel_sub['param_svm__kernel'].astype(str), -kernel_sub['mean_test_score'], color=['#3498db', '#e74c3c'])
plt.title("Kernel Comparison (Log Loss)")

plt.tight_layout()
plt.savefig(os.path.join(results_dir, "tuning_curves.png"))
plt.show()

# Prediction Certainty Visualisation
print("Rendering Prediction Distributions...")
plt.figure(figsize=(8, 6))
label_map = {0: 'Anomaly', 1: 'Healthy'}
actual_names = [label_map[label] for label in y_test]
df_probs = pd.DataFrame({
    'Probability': y_probs[:, 1],
    'Actual': actual_names
})
sns.kdeplot(data=df_probs, x='Probability', hue='Actual',
            fill=True, common_norm=False, palette='coolwarm')
plt.title("Class Probability Density (Model Confidence)")
plt.xlabel("Probability of Healthy Label (0.0=Anomaly, 1.0=Healthy)")
plt.ylabel("Density")
plt.savefig(os.path.join(results_dir, "prediction_certainty.png"))
plt.show()

# Learning Curve: Log Loss
print("Generating Log Loss Learning Curve...")
train_sizes, train_scores, test_scores = learning_curve(
    best_model, X_train, y_train, cv=StratifiedKFold(n_splits=3), n_jobs=-1,
    train_sizes=np.linspace(0.3, 1.0, 4), scoring='neg_log_loss'
)
plt.figure(figsize=(8, 6))
plt.plot(train_sizes, -np.mean(train_scores, axis=1), 'o-', color="r", label="Training Loss")
plt.plot(train_sizes, -np.mean(test_scores, axis=1), 'o-', color="g", label="Validation Loss")
plt.title("Learning Curve (Log Loss Convergence)")
plt.xlabel("Samples"); plt.ylabel("Log Loss"); plt.legend(); plt.grid(True)
plt.savefig(os.path.join(results_dir, "learning_curve_logloss.png"))
plt.show()

# Model Saving
save_path = r"C:\Users\jagde\PycharmProjects\AIHealthcareAssistant\models\saved_models"
os.makedirs(save_path, exist_ok=True)
joblib.dump(best_model, os.path.join(save_path, "voice_full_pipeline.pkl"))
print(f"\nSaved visuals to {results_dir} and model to {save_path}")