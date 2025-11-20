import os
import numpy as np
import librosa
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

sr_target = 16000  # target sampling rate
n_mfcc = 13  # number of MFCC features

def is_valid_audio(file_path, sr=sr_target):
    """Check if an audio file is loadable and finite."""
    try:
        y, _ = librosa.load(file_path, sr=sr)
        if not np.isfinite(y).all() or np.max(np.abs(y)) == 0:
            return False
        return True
    except Exception:
        return False

def extract_features(file_path, sr=sr_target, n_mfcc=n_mfcc):
    """Load audio, normalize, and extract MFCC features."""
    y, _ = librosa.load(file_path, sr=sr)
    y = y / np.max(np.abs(y))  # normalize
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfccs.T, axis=0)  # average over time frames


def get_audio_files(directory, extensions=('.wav', '.mp3')):
    """Recursively get all audio files in a directory."""
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(extensions):
                yield os.path.join(root, file)

def process_dataset(dataset_name, dataset_dir):
    """Process a dataset and extract MFCC features for valid audio files."""
    print(f"Processing {dataset_name} dataset...")
    features = []
    valid_count = 0
    total_files = 0

    for file_path in get_audio_files(dataset_dir):
        total_files += 1
        if is_valid_audio(file_path):
            try:
                feat = extract_features(file_path)
                features.append(feat)
                valid_count += 1
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        else:
            print(f"Skipping invalid audio: {file_path}")

    features = np.array(features)
    print(f"{dataset_name} features shape: {features.shape}")
    print(f"Valid files processed: {valid_count}/{total_files}\n")
    return features

if __name__ == "__main__":
    # Coswara dataset
    coswara_dir = r"C:\Users\jagde\PycharmProjects\AIHealthcareAssistant\data\voice\coswara\Coswara-Data"
    coswara_features = process_dataset("Coswara", coswara_dir)

    # Common Voice dataset
    commonvoice_dir = r"C:\Users\jagde\PycharmProjects\AIHealthcareAssistant\data\voice\cv-corpus-22.0\en\clips"
    commonvoice_features = process_dataset("Common Voice", commonvoice_dir)

    save_dir = r"C:\Users\jagde\PycharmProjects\AIHealthcareAssistant\data\voice\processed_features"
    os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist

    np.save(os.path.join(save_dir, 'coswara_mfcc_features.npy'), coswara_features)
    np.save(os.path.join(save_dir, 'commonvoice_mfcc_features.npy'), commonvoice_features)

    print(f"Features saved to {save_dir}")
    print("Voice preprocessing complete!")
