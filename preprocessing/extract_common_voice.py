import tarfile
import os

# Path to your downloaded Common Voice .tar.gz file
file_path = r"C:\Users\jagde\Downloads\cv-corpus-22.0-delta-2025-06-20-en.tar.gz"

# Destination folder inside your project
extract_path = r"/data/voice"

# Make sure the destination folder exists
os.makedirs(extract_path, exist_ok=True)

# Extract the .tar.gz
with tarfile.open(file_path, "r:gz") as tar:
    tar.extractall(path=extract_path)

print("Extraction complete! Check your data/voice folder.")
