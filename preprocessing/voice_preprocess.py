import os
import numpy as np
import librosa
import json
import parselmouth
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

sr_target = 16000
n_mfcc = 13

def extract_features(file_path):
    """Extract MFCC + pitch + jitter + shimmer features from an audio file."""
    try:
        # Loading audio
        y, _ = librosa.load(file_path, sr=sr_target)

        # 1. Handling silent/empty arrays early
        if y.size == 0 or np.max(np.abs(y)) == 0:
            raise ValueError("Audio signal is empty or entirely silent.")

        # Normalize
        y = y / (np.max(np.abs(y)) + 1e-6)

        # MFCCs
        mfcc = librosa.feature.mfcc(y=y, sr=sr_target, n_mfcc=n_mfcc)

        if mfcc.shape[1] < 2:
            raise ValueError("MFCC array has too few frames for reliable mean calculation.")

        mfcc_mean = np.mean(mfcc.T, axis=0)

        # jitter shimmer
        sound = parselmouth.Sound(file_path)
        pitch_obj = sound.to_pitch()
        pulses = parselmouth.praat.call([sound, pitch_obj], "To PointProcess (cc)")

        f0_mean = parselmouth.praat.call(pitch_obj, "Get mean", 0, 0, "Hertz")
        f0_std = parselmouth.praat.call(pitch_obj, "Get standard deviation", 0, 0, "Hertz")

        jitter = parselmouth.praat.call(pulses, "Get jitter (ddp)",
                                        0, 0, 0.0001, 0.02, 1.3)

        shimmer = parselmouth.praat.call([sound, pulses], "Get shimmer (apq3)",
                                         0, 0, 0.0001, 0.02, 1.3, 1.6)

        extra = np.array([f0_mean, f0_std, jitter, shimmer])

        # Combined feature vector
        final_vector = np.concatenate([mfcc_mean, extra])

        # Final check for NaNs that might arise from edge-case Parselmouth failure
        if not np.isfinite(final_vector).all():
            raise ValueError("Feature vector contains NaNs after extraction.")

        return final_vector

    except Exception as e:
        return None


def process_coswara(coswara_dir):
    features = []
    labels = []
    total_files_attempted = 0

    print("Processing Coswara dataset...")

    for root, dirs, files in os.walk(coswara_dir):

        if "metadata.json" not in files:
            continue

        metadata_path = os.path.join(root, "metadata.json")

        try:
            meta = json.load(open(metadata_path))
        except:
            print(f"Skipping invalid metadata: {metadata_path}")
            continue

        covid_status = meta.get("covid_status", "").lower()
        label = 1 if covid_status == "healthy" else 0

        # Filtering out corrupt file names
        audio_files = [f for f in files if f.endswith(".wav") and not f.startswith("._")]

        if len(audio_files) == 0:
            continue

        for f in audio_files:
            file_path = os.path.join(root, f)
            total_files_attempted += 1

            feat = extract_features(file_path)

            if feat is not None:
                features.append(feat)
                labels.append(label)

    features = np.array(features)
    labels = np.array(labels)

    total_usable = len(features)
    skipped_count = total_files_attempted - total_usable

    print(f"Finished processing.")
    print(f"Total files attempted: {total_files_attempted}")
    print(f"Total usable audio files: {total_usable}")
    print(f"Files skipped due to error/corruption: {skipped_count}")
    print(f"Feature shape: {features.shape}")
    print(f"Label shape: {labels.shape}")

    return features, labels

if __name__ == "__main__":
    coswara_dir = r"C:\Users\jagde\PycharmProjects\AIHealthcareAssistant\data\voice\coswara\Coswara-Data"
    save_dir = r"C:\Users\jagde\PycharmProjects\AIHealthcareAssistant\data\voice\processed_features"

    os.makedirs(save_dir, exist_ok=True)

    X, y = process_coswara(coswara_dir)

    np.save(os.path.join(save_dir, "coswara_features.npy"), X)
    np.save(os.path.join(save_dir, "coswara_labels.npy"), y)

    print("\nSaved preprocessed features to:")
    print(save_dir)