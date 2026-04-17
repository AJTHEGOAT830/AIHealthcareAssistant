import os
import numpy as np
import joblib
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, learning_curve, validation_curve, \
    cross_val_score
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             log_loss, roc_curve, auc)
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize

results_dir = r"C:\Users\jagde\PycharmProjects\AIHealthcareAssistant\models\image_results"
save_model_path = r"C:\Users\jagde\PycharmProjects\AIHealthcareAssistant\models\saved_models"
processed_dir = r"C:\Users\jagde\PycharmProjects\AIHealthcareAssistant\data\images\Dermnet\processed"
dataset_dir = r"C:\Users\jagde\PycharmProjects\AIHealthcareAssistant\data\images\Dermnet\dataset_merged\train"

os.makedirs(results_dir, exist_ok=True)
os.makedirs(save_model_path, exist_ok=True)

# Data loading and preprocesing
print("Loading image data...")
raw_images = np.load(os.path.join(processed_dir, "train_images.npy"))
y = np.load(os.path.join(processed_dir, "train_labels.npy"))

print("Resizing to 64x64 and flattening...")
X_resized = np.array([cv2.resize(img, (64, 64)).flatten() for img in raw_images], dtype='float32')
del raw_images

class_names = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
n_classes = len(class_names)

X_train, X_test, y_train, y_test = train_test_split(X_resized, y, test_size=0.20, random_state=42, stratify=y)

# Pipeline and parameters
img_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=15)),
    ('rf', RandomForestClassifier(random_state=42, n_jobs=-1, max_samples=0.7))
])

param_grid = {
    'rf__n_estimators': [150],
    'rf__max_depth': [6, 8, 10],
    'rf__min_samples_leaf': [30, 50]
}

print(f"\n--- Starting Fast 3-Fold CV Grid Search ---")
cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
grid_search = GridSearchCV(img_pipeline, param_grid, cv=cv_strategy, scoring='neg_log_loss', n_jobs=-1, return_train_score=True)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

# Evaluation
y_pred = best_model.predict(X_test)
y_probs = best_model.predict_proba(X_test)
test_acc = accuracy_score(y_test, y_pred)
loss_value = log_loss(y_test, y_probs)

cv_results = pd.DataFrame(grid_search.cv_results_)
best_row = cv_results[cv_results['rank_test_score'] == 1].iloc[0]

print("\nCalculating Cross-Validation Accuracy...")
cv_accuracy_scores = cross_val_score(best_model, X_train, y_train, cv=cv_strategy, scoring='accuracy', n_jobs=-1)
mean_cv_acc = np.mean(cv_accuracy_scores)
sd_cv_acc = np.std(cv_accuracy_scores)

print("\n" + "="*50)
print("      IMAGE MODEL FINAL AUDIT REPORT")
print("="*50)
print(f"--- IMAGE MODEL COMPARISON DATA ---")
print(f"Mean CV Accuracy:        {mean_cv_acc:.4f}")
print(f"Standard Deviation (SD): {sd_cv_acc:.4f}")
print(f"Individual Fold Scores:  {cv_accuracy_scores}")
print("-" * 50)
print(f"Best Parameters:  {best_params}")
print(f"Final Test Accuracy: {test_acc:.4f}")
print(f"Log-Loss (Error):    {loss_value:.4f}")
print("-" * 50)
print("\nFull Classification Report (Recall/F1/Precision):")
print(classification_report(y_test, y_pred, target_names=class_names))
print("="*50)

# Performance summary
plt.figure(figsize=(22, 7))

plt.subplot(1, 3, 1)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=class_names, yticklabels=class_names, cbar=False)
plt.xticks(rotation=45, ha='right'); plt.title("Confusion Matrix")

plt.subplot(1, 3, 2)
y_test_bin = label_binarize(y_test, classes=range(n_classes))
fpr, tpr = dict(), dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_probs[:, i])
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= n_classes
plt.plot(all_fpr, mean_tpr, color='darkgreen', lw=2, label=f'Macro-AUC = {auc(all_fpr, mean_tpr):.2f}')
plt.plot([0, 1], [0, 1], 'k--'); plt.title("ROC Curve"); plt.legend(); plt.grid(True)

plt.subplot(1, 3, 3)
report = classification_report(y_test, y_pred, output_dict=True)
f1_scores = [report[str(i)]['f1-score'] for i in range(n_classes)]
sns.boxplot(y=f1_scores, color='#2ecc71'); plt.title("F1-Score Stability")

plt.tight_layout()
plt.savefig(os.path.join(results_dir, "image_performance_summary.png"))
plt.show()

# Tuning validation curves
print("Plotting Dashboard 2...")
plt.figure(figsize=(18, 6))

plt.subplot(1, 2, 1)
depth_range = [4, 6, 8, 10, 12]
train_d, test_d = validation_curve(best_model, X_train, y_train, param_name="rf__max_depth",
                                   param_range=depth_range, cv=2, scoring="neg_log_loss", n_jobs=-1)
plt.plot(depth_range, -np.mean(train_d, axis=1), 'r-o', label='Train')
plt.plot(depth_range, -np.mean(test_d, axis=1), 'g-o', label='Val')
plt.title("Log Loss vs. Tree Depth"); plt.legend(); plt.grid(True)

plt.subplot(1, 2, 2)
est_range = [50, 150, 250]
train_e, test_e = validation_curve(best_model, X_train, y_train, param_name="rf__n_estimators",
                                   param_range=est_range, cv=2, scoring="neg_log_loss", n_jobs=-1)
plt.plot(est_range, -np.mean(train_e, axis=1), 'r-o', label='Train')
plt.plot(est_range, -np.mean(test_e, axis=1), 'g-o', label='Val')
plt.title("Log Loss vs. Estimators"); plt.legend(); plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(results_dir, "image_accuracy_tuning.png"))
plt.show()

# Learning curves
print("Plotting Dashboard 3...")
plt.figure(figsize=(18, 5))

plt.subplot(1, 2, 1)
train_sizes, train_scores, test_scores = learning_curve(best_model, X_train, y_train, cv=2,
                                                        scoring="neg_log_loss", n_jobs=-1,
                                                        train_sizes=np.linspace(0.3, 1.0, 4))
plt.plot(train_sizes, -np.mean(train_scores, axis=1), 'r-o', label='Train')
plt.plot(train_sizes, -np.mean(test_scores, axis=1), 'g-o', label='Val')
plt.title("Sample Sufficiency (Log Loss)"); plt.legend(); plt.grid(True)

#PCA plot
plt.subplot(1, 2, 2)
pca_explained = best_model.named_steps['pca'].explained_variance_ratio_
plt.step(range(len(pca_explained)), np.cumsum(pca_explained), where='mid', color='darkgreen')
plt.title("PCA Explained Variance"); plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(results_dir, "image_methodology_check.png"))
plt.show()

# Saving model
joblib.dump(best_model, os.path.join(save_model_path, "rf_image_pipeline.pkl"))
print(f"\nSUCCESS: Results saved to {results_dir}")