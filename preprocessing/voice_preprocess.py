import os
import numpy as np
import librosa
import json
import parselmouth
import warnings
import wfdb  # pip install wfdb
import soundfile as sf  # pip install soundfile

warnings.filterwarnings("ignore", category=RuntimeWarning)

warnings.filterwarnings("ignore")

SR_TARGET = 16000
N_MFCC = 13

def extract_features_v2(file_path):
    try:
        # Load for MFCCs
        y, _ = librosa.load(file_path, sr=SR_TARGET)
        if y.size == 0: return None
        y = y / (np.max(np.abs(y)) + 1e-6)

        mfcc = librosa.feature.mfcc(y=y, sr=SR_TARGET, n_mfcc=N_MFCC)
        mfcc_mean = np.mean(mfcc.T, axis=0)

        # Parselmouth for Jitter, Shimmer, and HNR
        sound = parselmouth.Sound(file_path)
        pitch = sound.to_pitch()
        pulses = parselmouth.praat.call([sound, pitch], "To PointProcess (cc)")

        # Jitter (Instability)
        jitter = parselmouth.praat.call(pulses, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)
        # Shimmer (Volume instability)
        shimmer = parselmouth.praat.call([sound, pulses], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        # 3. HNR
        harmonicity = sound.to_harmonicity()
        hnr = parselmouth.praat.call(harmonicity, "Get mean", 0, 0)

        extra = np.array([jitter, shimmer, hnr])
        return np.concatenate([mfcc_mean, extra])
    except Exception as e:
        return None


def process_svd_kaggle(base_dir):
    """Processes the Normal, Laryngozele, and Vox senilis folders."""
    features, labels = [], []

    #Normal = 1Healthy, Others -> 0 (Anomaly)
    categories = {
        "Normal": 1,
        "Laryngozele": 0,
        "Vox senilis": 0
    }

    for folder, label in categories.items():
        folder_path = os.path.join(base_dir, folder)
        if not os.path.exists(folder_path):
            print(f"Warning: Folder {folder} not found at {folder_path}")
            continue

        print(f"Processing {folder}...")
        for file in os.listdir(folder_path):
            if file.endswith(".wav"):
                feat = extract_features_v2(os.path.join(folder_path, file))
                if feat is not None:
                    features.append(feat)
                    labels.append(label)

    return np.array(features), np.array(labels)


if __name__ == "__main__":
    DATA_PATH = r"C:\Users\jagde\PycharmProjects\AIHealthcareAssistant\data\voice\SVD Dataset\patient-vocal-dataset\patient-vocal-dataset"
    SAVE_PATH = r"C:\Users\jagde\PycharmProjects\AIHealthcareAssistant\data\voice\processed_features"

    X, y = process_svd_kaggle(DATA_PATH)
    os.makedirs(SAVE_PATH, exist_ok=True)
    np.save(os.path.join(SAVE_PATH, "svd_features.npy"), X)
    np.save(os.path.join(SAVE_PATH, "svd_labels.npy"), y)
    print(f"Done! Saved {len(X)} samples. Healthy: {sum(y)}, Anomaly: {len(y) - sum(y)}")
