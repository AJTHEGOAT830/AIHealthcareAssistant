from flask import Flask, request, jsonify
import numpy as np
import cv2
import librosa
import parselmouth
import joblib
import os
import uuid
import logging
from pydub import AudioSegment


# Logging
logging.basicConfig(level=logging.ERROR)

# App init
app = Flask(__name__)

MODEL_DIR = r"C:\Users\jagde\PycharmProjects\AIHealthcareAssistant\models\saved_models"

VOICE_MODEL = joblib.load(os.path.join(MODEL_DIR, "svm_voice_classifier.pkl"))
VOICE_SCALER = joblib.load(os.path.join(MODEL_DIR, "voice_scaler.pkl"))
VOICE_PCA = joblib.load(os.path.join(MODEL_DIR, "voice_pca.pkl"))

IMAGE_MODEL = joblib.load(os.path.join(MODEL_DIR, "rf_image_classifier.pkl"))
CLASS_NAMES = np.load(os.path.join(MODEL_DIR, "class_names.npy"), allow_pickle=True)

sr_target = 16000
n_mfcc = 13

# Chatbot logic
def chatbot_reply(user_msg: str) -> str:
    if not user_msg:
        return (
            "Hello — I can help with general health guidance. "
            "Describe your symptoms, or try the image or voice analysis tools. "
            "This system does not provide medical diagnoses."
        )

    msg = user_msg.lower()

    # Urgent Symptoms
    urgent_signs = [
        "chest pain", "severe chest", "crushing chest",
        "shortness of breath", "can't breathe", "hard to breathe",
        "fainting", "passed out",
        "slurred speech", "face drooping", "stroke",
        "seizure",
        "heavy bleeding", "bleeding a lot",
        "head injury", "loss of consciousness",
        "fever 104", "fever 40"
    ]

    if any(term in msg for term in urgent_signs):
        return (
            "Some of the symptoms you described may be serious. "
            "I cannot diagnose conditions, but it would be safest to seek urgent medical attention "
            "or contact emergency services now. If you are unsure, it is better to be cautious."
        )

    # Respiratory Symptoms
    if "cough" in msg or "breath" in msg or "wheezing" in msg:
        return (
            "Cough and breathing symptoms may come from infections, allergies, or irritation. "
            "Stay hydrated and avoid smoke or irritants. Seek care if symptoms worsen, "
            "last more than a few days, or are associated with chest pain or fever. "
            "You can also try the voice tool — it may help flag possible anomalies, "
            "but it is not diagnostic."
        )

    # Fever/infection symptoms
    if "fever" in msg or "temperature" in msg or "flu" in msg:
        return (
            "Fever is often related to infection. Rest, fluids, and monitoring usually help. "
            "Seek medical advice if it lasts beyond three days, becomes very high, "
            "or is accompanied by confusion, severe weakness, or chest pain."
        )

    # Skin/Rash symptoms
    if "rash" in msg or "skin" in msg or "spots" in msg:
        return (
            "Rashes may be due to irritation, infection, or allergies. "
            "Keep the area clean and avoid scratching. "
            "You may upload an image for analysis — it may offer supportive insight, "
            "but it does not replace medical evaluation."
        )

    # If Pain in user imput
    if "pain" in msg or "ache" in msg or "hurts" in msg:
        return (
            "Pain varies widely in cause. Resting and staying hydrated may help. "
            "Seek medical evaluation if pain is severe, persistent, worsening, "
            "or linked to injury, fever, numbness, or weakness."
        )

    # If medication in response
    if "medicine" in msg or "medication" in msg or "take" in msg:
        return (
            "I cannot provide medication dosing or prescribing guidance. "
            "Those decisions depend on medical history and allergies. "
            "A clinician or pharmacist is best placed to advise."
        )

    # default response
    return (
        "I can provide general health guidance, but I cannot diagnose conditions. "
        "You can also upload an image or record your voice for additional analysis. "
        "Tell me more about what you are experiencing."
    )

def extract_voice_features(path):
    try:
        # Load and Standardize
        y, _ = librosa.load(path, sr=16000)
        if y.size == 0: return None
        y = y / (np.max(np.abs(y)) + 1e-6)

        # Extracting 13 MFCCs
        mfcc = librosa.feature.mfcc(y=y, sr=16000, n_mfcc=13)
        mfcc_mean = np.mean(mfcc.T, axis=0)

        # Extracting Jitter, Shimmer, and HNR
        sound = parselmouth.Sound(path)
        pitch = sound.to_pitch()
        pulses = parselmouth.praat.call([sound, pitch], "To PointProcess (cc)")

        jitter = parselmouth.praat.call(pulses, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)
        shimmer = parselmouth.praat.call([sound, pulses], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

        harmonicity = sound.to_harmonicity()
        hnr = parselmouth.praat.call(harmonicity, "Get mean", 0, 0)

        # Combining to match the 16 features the model expects
        extra = np.array([jitter, shimmer, hnr])
        return np.concatenate([mfcc_mean, extra])
    except Exception as e:
        print(f"Extraction Error: {e}")
        return None

# /voice-analysis
@app.route("/voice-analysis", methods=["POST"])
def voice_analysis():
    if "audio" not in request.files:
        return jsonify({"error": "No audio uploaded"}), 400

    raw_audio = request.files["audio"]
    temp_input = f"temp_{uuid.uuid4().hex}"
    temp_wav = f"{temp_input}.wav"

    try:
        raw_audio.save(temp_input)

        # Converting anything to WAV
        # Forcing parameters during export to match training data
        AudioSegment.from_file(temp_input).set_frame_rate(16000).set_channels(1).export(temp_wav, format="wav")

        feat = extract_voice_features(temp_wav)
        if feat is None:
            return jsonify({
                "result": "Could not analyse voice",
                "disclaimer": "This is not a medical diagnosis."
            })

        feat = feat.reshape(1, -1)
        scaled = VOICE_SCALER.transform(feat)
        reduced = VOICE_PCA.transform(scaled)
        pred = VOICE_MODEL.predict(reduced)[0]

        label = "Clear Vocal Profile: Your vocal patterns appear steady and clear." if pred == 1 else "Possible Vocal Anomaly: We noticed some minor vocal irregularities. This is common with fatigue or illness symptoms."

        return jsonify({
            "result": label,
            "disclaimer": "This is not a medical diagnosis."
        })

    finally:
        for p in [temp_input, temp_wav]:
            if os.path.exists(p):
                os.remove(p)

# /image-analysis
@app.route("/image-analysis", methods=["POST"])
def image_analysis():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    img_file = request.files["image"]
    img_path = f"temp_{uuid.uuid4().hex}.jpg"

    try:
        img_file.save(img_path)

        img = cv2.imread(img_path)
        if img is None:
            return jsonify({"error": "Invalid image uploaded"}), 400

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img_flat = img.reshape(1, -1)

        pred = IMAGE_MODEL.predict(img_flat)[0]
        label = str(CLASS_NAMES[pred])

        return jsonify({
            "result": label,
            "disclaimer": "This is not a medical diagnosis."
        })

    finally:
        if os.path.exists(img_path):
            os.remove(img_path)

# /chatbot
@app.route("/chatbot", methods=["POST"])
def chatbot():
    data = request.get_json(silent=True) or {}
    user_msg = (data.get("message") or "").lower()

    response = chatbot_reply(user_msg)

    return jsonify({
        "response": response,
        "disclaimer": "This is not a medical diagnosis."
    })

# /multimodal
@app.route("/multimodal", methods=["POST"])
def multimodal():
    text = request.form.get("text", "")
    image = request.files.get("image")
    audio = request.files.get("audio")

    responses = []

    # Image
    if image:
        img_path = f"temp_{uuid.uuid4().hex}.jpg"
        try:
            image.save(img_path)

            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (224, 224))
                img_flat = img.reshape(1, -1)

                pred = IMAGE_MODEL.predict(img_flat)[0]
                responses.append(f"Image suggests: {CLASS_NAMES[pred]}")
        finally:
            if os.path.exists(img_path):
                os.remove(img_path)

    # Audio
    if audio:
        audio_path = f"temp_{uuid.uuid4().hex}.wav"
        try:
            audio.save(audio_path)

            feat = extract_voice_features(audio_path)
            if feat is not None:
                feat = feat.reshape(1, -1)
                scaled = VOICE_SCALER.transform(feat)
                reduced = VOICE_PCA.transform(scaled)
                pred = VOICE_MODEL.predict(reduced)[0]

                responses.append(
                    "Voice suggests healthy"
                    if pred == 1
                    else "Voice suggests possible anomaly"
                )
        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)

    # Chatbot
    responses.append(chatbot_reply(text))

    return jsonify({
        "response": " ".join(responses),
        "disclaimer": "This system provides non-diagnostic guidance only."
    })


# Run server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
