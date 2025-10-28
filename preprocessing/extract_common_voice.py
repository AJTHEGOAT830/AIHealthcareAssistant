import tarfile
import os

file_path = r"C:\Users\jagde\Downloads\cv-corpus-22.0-delta-2025-06-20-en.tar.gz"
extract_path = r"/data/voice"
os.makedirs(extract_path, exist_ok=True)
with tarfile.open(file_path, "r:gz") as tar:
    tar.extractall(path=extract_path)

print("Extraction complete! Check your data/voice folder.")
