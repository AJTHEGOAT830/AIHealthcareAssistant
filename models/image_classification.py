import os
import numpy as np
import joblib
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, learning_curve, validation_curve
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             log_loss, roc_curve, auc)
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize

# Set up directory
results_dir = r"C:\Users\jagde\PycharmProjects\AIHealthcareAssistant\models\image_results"
save_model_path = r"C:\Users\jagde\PycharmProjects\AIHealthcareAssistant\models\saved_models"
os.makedirs(results_dir, exist_ok=True)
os.makedirs(save_model_path, exist_ok=True)

processed_dir = r"C:\Users\jagde\PycharmProjects\AIHealthcareAssistant\data\images\Dermnet\processed"
dataset_dir = r"C:\Users\jagde\PycharmProjects\AIHealthcareAssistant\data\images\Dermnet\dataset_merged\train"

# Loading
print("Loading image data...")
raw_images = np.load(os.path.join(processed_dir, "train_images.npy"))
y = np.load(os.path.join(processed_dir, "train_labels.npy"))

print("Resizing to 64x64 for memory efficiency...")
X_resized = np.array([cv2.resize(img, (64, 64)).flatten() for img in raw_images], dtype='float32')
del raw_images

class_names = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
n_classes = len(class_names)

X_train, X_test, y_train, y_test = train_test_split(X_resized, y, test_size=0.20, random_state=42, stratify=y)

# Pipeline and grid search
img_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=50)),
    ('rf', RandomForestClassifier(random_state=42))
])

param_grid = {
    'rf__n_estimators': [50, 100, 200],
    'rf__max_depth': [10, 20, 30, None],
    'rf__min_samples_split': [2, 5]
}

print(f"\n--- Starting 5-Fold CV Grid Search ---")
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(img_pipeline, param_grid, cv=cv_strategy, scoring='accuracy', n_jobs=-1, return_train_score=True)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

# Extracting Metrics
best_idx = grid_search.best_index_
cv_mean = grid_search.cv_results_['mean_test_score'][best_idx]
cv_std = grid_search.cv_results_['std_test_score'][best_idx]

y_pred = best_model.predict(X_test)
y_probs = best_model.predict_proba(X_test)
test_acc = accuracy_score(y_test, y_pred)
loss_value = log_loss(y_test, y_probs)

print("\n" + "="*40)
print("      IMAGE MODEL FINAL AUDIT REPORT")
print("="*40)
print(f"Optimal Parameters:  {best_params}")
print(f"Mean 5-Fold CV Acc:  {cv_mean:.4f}")
print(f"CV Std Deviation:    {cv_std:.4f}")
print(f"Final Test Accuracy: {test_acc:.4f}")
print(f"Log-Loss (Error):    {loss_value:.4f}")
print("-" * 40)
print(classification_report(y_test, y_pred, target_names=class_names))
print("="*40)

# Performance metrics dashbord (CM, ROC, F1)
print("Plotting Dashboard 1: Performance Summary...")
plt.figure(figsize=(22, 7))

plt.subplot(1, 3, 1)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=class_names, yticklabels=class_names, cbar=False)
plt.xticks(rotation=45, ha='right'); plt.title("Confusion Matrix (Actual Counts)")

plt.subplot(1, 3, 2)
y_test_bin = label_binarize(y_test, classes=range(n_classes))
fpr = dict(); tpr = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_probs[:, i])
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= n_classes
plt.plot(all_fpr, mean_tpr, color='darkgreen', lw=2, label=f'Macro-AUC = {auc(all_fpr, mean_tpr):.2f}')
plt.plot([0, 1], [0, 1], 'k--'); plt.title("ROC Curve (Diagnostic Power)"); plt.legend(); plt.grid(True)

plt.subplot(1, 3, 3)
report = classification_report(y_test, y_pred, output_dict=True)
f1_scores = [report[str(i)]['f1-score'] for i in range(n_classes)]
sns.boxplot(y=f1_scores, color='#2ecc71'); plt.title("F1-Score Stability Across Classes")

plt.tight_layout()
plt.savefig(os.path.join(results_dir, "image_performance_summary.png"))
plt.show()

# Log Loss Tuning error analysis
print("Plotting Dashboard 2: Log-Loss Tuning Curves...")
plt.figure(figsize=(18, 6))

# Depth Tuning
plt.subplot(1, 2, 1)
depth_range = [5, 10, 20, 30, 40]
train_d, test_d = validation_curve(best_model, X_train, y_train, param_name="rf__max_depth", param_range=depth_range, cv=3, scoring="neg_log_loss", n_jobs=-1)
plt.plot(depth_range, -np.mean(train_d, axis=1), 'r-o', label='Train Loss')
plt.plot(depth_range, -np.mean(test_d, axis=1), 'g-o', label='Val Loss')
plt.title("Log-Loss vs. Tree Depth"); plt.xlabel("Max Depth"); plt.ylabel("Log-Loss"); plt.legend(); plt.grid(True)

# Estimator Tuning
plt.subplot(1, 2, 2)
est_range = [50, 100, 200, 300]
train_e, test_e = validation_curve(best_model, X_train, y_train, param_name="rf__n_estimators", param_range=est_range, cv=3, scoring="neg_log_loss", n_jobs=-1)
plt.plot(est_range, -np.mean(train_e, axis=1), 'r-o', label='Train Loss')
plt.plot(est_range, -np.mean(test_e, axis=1), 'g-o', label='Val Loss')
plt.title("Log-Loss vs. Number of Trees"); plt.xlabel("Estimators"); plt.legend(); plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(results_dir, "image_logloss_tuning.png"))
plt.show()

# Data suffiency and PCA plots
print("Plotting Dashboard 3: Methodology Verification...")
plt.figure(figsize=(18, 5))

plt.subplot(1, 2, 1)
train_sizes, train_scores, test_scores = learning_curve(best_model, X_train, y_train, cv=3, n_jobs=-1, train_sizes=np.linspace(0.3, 1.0, 3))
plt.plot(train_sizes, np.mean(train_scores, axis=1), 'r-o', label='Train')
plt.plot(train_sizes, np.mean(test_scores, axis=1), 'g-o', label='Val')
plt.title("Learning Curve: Sample Sufficiency"); plt.legend(); plt.grid(True)

plt.subplot(1, 2, 2)
pca_explained = best_model.named_steps['pca'].explained_variance_ratio_
plt.step(range(len(pca_explained)), np.cumsum(pca_explained), where='mid', color='darkgreen')
plt.title("PCA: Cumulative Explained Variance"); plt.xlabel("Components"); plt.ylabel("Variance Ratio")

plt.tight_layout()
plt.savefig(os.path.join(results_dir, "image_methodology_check.png"))
plt.show()

# Saving Model
joblib.dump(best_model, os.path.join(save_model_path, "rf_image_pipeline.pkl"))
print(f"\nSUCCESS: All results and models saved to {results_dir}")