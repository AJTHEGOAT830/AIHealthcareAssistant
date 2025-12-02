import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

save_model_path = r"C:\Users\jagde\PycharmProjects\AIHealthcareAssistant\models\saved_models"
train_images = np.load(r"C:\Users\jagde\PycharmProjects\AIHealthcareAssistant\data\images\Dermnet\processed\train_images.npy")
train_labels = np.load(r"C:\Users\jagde\PycharmProjects\AIHealthcareAssistant\data\images\Dermnet\processed\train_labels.npy")

X = train_images.reshape(len(train_images), -1)  # flatten images
y = train_labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=200)
model.fit(X_train, y_train)

preds = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))

os.makedirs(save_model_path, exist_ok=True)
joblib.dump(model, os.path.join(save_model_path, "rf_image_classifier.pkl"))
print("Random Forest model saved!")