import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA

# Loading the SVD features
processed_dir = r"C:\Users\jagde\PycharmProjects\AIHealthcareAssistant\data\voice\processed_features"
X = np.load(os.path.join(processed_dir, "svd_features.npy"))
y = np.load(os.path.join(processed_dir, "svd_labels.npy"))

print(f"Data Loaded: {X.shape[0]} samples, {X.shape[1]} features")

# Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# PCA
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
print(f"PCA reduced features to: {X_train_pca.shape[1]}")

# Training SVM with optimised parameters
model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
model.fit(X_train_pca, y_train)

# Evaluation
y_pred = model.predict(X_test_pca)
print("\n--- RESULTS ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Anomaly", "Healthy"]))

# Save models for the Flask API
save_path = r"C:\Users\jagde\PycharmProjects\AIHealthcareAssistant\models\saved_models"
os.makedirs(save_path, exist_ok=True)

joblib.dump(model, os.path.join(save_path, "svm_voice_classifier.pkl"))
joblib.dump(scaler, os.path.join(save_path, "voice_scaler.pkl"))
joblib.dump(pca, os.path.join(save_path, "voice_pca.pkl"))

print(f"\nModel and Scaler saved to {save_path}")