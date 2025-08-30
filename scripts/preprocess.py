import os
import numpy as np
import pandas as pd
import cv2 
from tqdm import tqdm

def process_csv(input_csv, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading {input_csv}...")
    data = pd.read_csv(input_csv).values

    for i, row in tqdm(enumerate(data), total=len(data)):
        label = row[0]  # first column is the letter index
        pixels = row[1:].reshape(28, 28).astype(np.uint8)

        label_dir = os.path.join(output_dir, str(label))
        os.makedirs(label_dir, exist_ok=True)

        filename = os.path.join(label_dir, f"{i}.png")
        cv2.imwrite(filename, pixels)

    print(f"✅ Processed {len(data)} images into {output_dir}")

if __name__ == "__main__":
    process_csv("data/raw/A_Z Handwritten Data.csv", "data/processed/train")
