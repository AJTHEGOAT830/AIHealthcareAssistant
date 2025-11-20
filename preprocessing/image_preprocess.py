import os
import cv2
import numpy as np
from tqdm import tqdm

image_size = (224, 224)  # resize images to 224x224
data_dir = r"C:\Users\jagde\PycharmProjects\AIHealthcareAssistant\data\images\Dermnet\dataset_merged"
save_dir = r"C:\Users\jagde\PycharmProjects\AIHealthcareAssistant\data\images\Dermnet\processed"

def process_folder(folder_path):
    images = []
    labels = []
    class_names = sorted(os.listdir(folder_path))

    for idx, class_name in enumerate(class_names):
        class_path = os.path.join(folder_path, class_name)
        if not os.path.isdir(class_path):
            continue

        for file_name in tqdm(os.listdir(class_path), desc=f"Processing {class_name}"):
            file_path = os.path.join(class_path, file_name)
            try:
                img = cv2.imread(file_path)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert to RGB
                img = cv2.resize(img, image_size)
                images.append(img)
                labels.append(idx)
            except Exception as e:
                print(f"Failed to process {file_path}: {e}")

    images = np.array(images, dtype=np.uint8)
    labels = np.array(labels, dtype=np.int32)
    return images, labels, class_names

if __name__ == "__main__":
    os.makedirs(save_dir, exist_ok=True)

    for split in ['train', 'test']:
        split_path = os.path.join(data_dir, split)
        images, labels, class_names = process_folder(split_path)

        np.save(os.path.join(save_dir, f"{split}_images.npy"), images)
        np.save(os.path.join(save_dir, f"{split}_labels.npy"), labels)
        print(f"{split.capitalize()} set saved: {images.shape[0]} images, {len(class_names)} classes")

    print("Image preprocessing complete!")
