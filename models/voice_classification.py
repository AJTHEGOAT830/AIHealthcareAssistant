import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
import joblib

# Load preprocessed features
processed_dir = r"C:\Users\jagde\PycharmProjects\AIHealthcareAssistant\data\voice\processed_features"

print("Loading feature dataset...")
X = np.load(os.path.join(processed_dir, "coswara_features.npy"))
y = np.load(os.path.join(processed_dir, "coswara_labels.npy"))

print("Shape:", X.shape, y.shape)

# Train/Val/Test Split (70/15/15)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"\nData split:")
print(f"Train: {len(X_train)}")
print(f"Val:   {len(X_val)}")
print(f"Test:  {len(X_test)}")


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# OPTIONAL: PCA
USE_PCA = True

if USE_PCA:
    pca = PCA(n_components=0.95)  # keep 95% variance
    X_train_scaled = pca.fit_transform(X_train_scaled)
    X_val_scaled = pca.transform(X_val_scaled)
    X_test_scaled = pca.transform(X_test_scaled)

    print(f"\nPCA applied. New shape: {X_train_scaled.shape}")
else:
    pca = None

# Apply SMOTE only to train for balancing the dataset
print("\nApplying SMOTE...")
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)

print("Balanced training shape:", X_train_bal.shape, y_train_bal.shape)


# Training SVM
print("\nTraining SVM with RBF kernel...")
svm_model = SVC(kernel="rbf", C=10, gamma="scale", probability=True)
svm_model.fit(X_train_bal, y_train_bal)

# Evaluation on the test set
y_pred = svm_model.predict(X_test_scaled)

print("\nFINAL TEST ACCURACY:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=["Anomaly", "Healthy"]))


# Saving the model + scaler + PCA
save_path = r"C:\Users\jagde\PycharmProjects\AIHealthcareAssistant\models\saved_models"
os.makedirs(save_path, exist_ok=True)

joblib.dump(svm_model, os.path.join(save_path, "svm_voice_classifier.pkl"))
joblib.dump(scaler, os.path.join(save_path, "voice_scaler.pkl"))

if pca:
    joblib.dump(pca, os.path.join(save_path, "voice_pca.pkl"))

print("\nModel, scaler, and PCA saved successfully!")
