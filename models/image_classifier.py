import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

processed_dir = r"C:\Users\jagde\PycharmProjects\AIHealthcareAssistant\data\images\Dermnet\processed"
save_model_path = r"C:\Users\jagde\PycharmProjects\AIHealthcareAssistant\models\saved_models"

train_images = np.load(os.path.join(processed_dir, "train_images.npy"))
train_labels = np.load(os.path.join(processed_dir, "train_labels.npy"))

X = train_images.reshape(len(train_images), -1)
y = train_labels

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=200)
model.fit(X_train, y_train)

preds = model.predict(X_test)
print("Image Classification Accuracy:", accuracy_score(y_test, preds))

os.makedirs(save_model_path, exist_ok=True)
model_path = os.path.join(save_model_path, "rf_image_classifier.pkl")
joblib.dump(model, model_path)

print(f"Random Forest image model saved to: {model_path}")
