import os
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json
# Paths
DATASET_CSV = r"C:\Users\singh\OneDrive\Desktop\handwritten_ocr\data\raw\A_Z Handwritten Data.csv"
RAW_TRAIN = "data/raw/train"
RAW_TEST = "data/raw/test"

os.makedirs(RAW_TRAIN, exist_ok=True)
os.makedirs(RAW_TEST, exist_ok=True)

# Load CSV
print("Loading dataset...")
df = pd.read_csv(DATASET_CSV, header=None)
labels = df.iloc[:, 0].values
images = df.iloc[:, 1:].values.reshape(-1, 28, 28)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2, random_state=42, stratify=labels
)

def save_images(images, labels, out_dir):
    for i, (img, lbl) in enumerate(tqdm(zip(images, labels), total=len(images))):
        img_path = os.path.join(out_dir, f"{i}.jpg")
        json_path = os.path.join(out_dir, f"{i}.json")

        cv2.imwrite(img_path, img)

        # Save label in JSON format
        with open(json_path, "w") as f:
            json.dump({"label": int(lbl)}, f)

print("Saving training set...")
save_images(X_train, y_train, RAW_TRAIN)

print("Saving test set...")
save_images(X_test, y_test, RAW_TEST)

print("✅ Dataset prepared successfully!")
