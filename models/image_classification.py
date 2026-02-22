import os
import numpy as np
import joblib
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Paths
processed_dir = r"C:\Users\jagde\PycharmProjects\AIHealthcareAssistant\data\images\Dermnet\processed"
dataset_dir = r"C:\Users\jagde\PycharmProjects\AIHealthcareAssistant\data\images\Dermnet\dataset_merged\train"
save_model_path = r"C:\Users\jagde\PycharmProjects\AIHealthcareAssistant\models\saved_models"

# Loading and Resizing
print("Loading data...")
raw_images = np.load(os.path.join(processed_dir, "train_images.npy"))
y = np.load(os.path.join(processed_dir, "train_labels.npy"))

print("Resizing images to 64x64 to prevent Memory Errors...")
X_resized = []
for img in raw_images:
    # Shrinking from 224x224 to 64x64 to reduce memory usage
    res = cv2.resize(img, (64, 64))
    X_resized.append(res.flatten())

X = np.array(X_resized, dtype='float32')
del raw_images # Free up RAM immediately

class_names = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

# Pipeline
img_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=50)), # Using 50 components for speed/stability
    ('rf', RandomForestClassifier(random_state=42))
])

# Hyperparameter Tuning
param_grid = {
    'rf__n_estimators': [100, 200],
    'rf__max_depth': [10, 20, None],
    'rf__min_samples_split': [2, 5]
}

print("Starting Hyperparameter Tuning (5-Fold CV)...")
grid_search = GridSearchCV(img_pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

#Extracting CV Metrics
best_idx = grid_search.best_index_
cv_mean = grid_search.cv_results_['mean_test_score'][best_idx]
cv_std = grid_search.cv_results_['std_test_score'][best_idx]

#Evaluation and Loss Function
y_pred = best_model.predict(X_test)
y_probs = best_model.predict_proba(X_test)

# Calculate Log-Loss
loss_value = log_loss(y_test, y_probs)

print(f"\nBest Parameters: {grid_search.best_params_}")
print(f"Mean CV Accuracy: {cv_mean:.4f}")
print(f"CV Standard Deviation (SD): {cv_std:.4f}")
print(f"Final Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Log-Loss (Error): {loss_value:.4f}")

print("\nFull Classification Report (Recall/F1/Precision):")
print(classification_report(y_test, y_pred, target_names=class_names))

# Saving
os.makedirs(save_model_path, exist_ok=True)
joblib.dump(best_model, os.path.join(save_model_path, "rf_image_pipeline.pkl"))
np.save(os.path.join(save_model_path, "class_names.npy"), class_names)
print("\nOptimized Pipeline saved successfully.")

