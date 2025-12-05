import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

processed_dir = r"C:\Users\jagde\PycharmProjects\AIHealthcareAssistant\data\images\Dermnet\processed"
dataset_dir = r"C:\Users\jagde\PycharmProjects\AIHealthcareAssistant\data\images\Dermnet\dataset_merged\train"
save_model_path = r"C:\Users\jagde\PycharmProjects\AIHealthcareAssistant\models\saved_models"

# Load feature data
train_images = np.load(os.path.join(processed_dir, "train_images.npy"))
train_labels = np.load(os.path.join(processed_dir, "train_labels.npy"))

# Autoload class names
class_names = sorted([
    d for d in os.listdir(dataset_dir)
    if os.path.isdir(os.path.join(dataset_dir, d))
])

print("Detected class names:", class_names)

# Flatten images for Random Forest
X = train_images.reshape(len(train_images), -1)
y = train_labels

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Train Random Forest
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# -------------------------
# Evaluate model
# -------------------------
preds = model.predict(X_test)

print("\nImage Classification Accuracy:", accuracy_score(y_test, preds))

print("\nClassification Report:\n")
print(classification_report(y_test, preds, target_names=class_names))

print("\nConfusion Matrix:\n")
cm = confusion_matrix(y_test, preds)

# Pretty print confusion matrix with labels
print(f"{'':15}{class_names}")
for i, row in enumerate(cm):
    print(f"{class_names[i]:15}{row}")

# -------------------------
# Save model
# -------------------------
os.makedirs(save_model_path, exist_ok=True)
model_path = os.path.join(save_model_path, "rf_image_classifier.pkl")
joblib.dump(model, model_path)

print(f"\nRandom Forest image model saved to: {model_path}")
