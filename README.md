# 📝 Handwritten Word Recognition (OCR)

A machine learning project to recognize handwritten words from images using a deep learning model.  
This repository contains preprocessing, training, and inference scripts for building an end-to-end OCR (Optical Character Recognition) system.

---

## 🚀 Features
- Preprocessing pipeline for raw image + annotation data  
- Training script for OCR model  
- Inference script for running predictions on new handwritten images  
- Modular and extensible codebase  

---

## ⚡ Quick Setup 

```bash
# Clone repo
git clone https://github.com/Avinash-Singh-13/Handwritten-word-recognition.git
cd Handwritten-word-recognition

# Create & activate virtual environment
# Linux / macOS
python3 -m venv venv && source venv/bin/activate
# Windows (PowerShell)
python -m venv venv; .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Preprocess dataset
python scripts/preprocess.py

# Train the model
python scripts/train.py

# Run inference on a test image
python scripts/infer.py --image path/to/image.jpg
```

### 📂 Project Structure
``` bash
Handwritten-word-recognition/
│── data/                 # Dataset folder (not included in repo)
│   ├── raw/train/        # Training images (.jpg) + annotations (.json)
│   ├── raw/test/         # Test images (.jpg) + annotations (.json)
│
│── scripts/              # Python scripts
│   ├── preprocess.py     # Step 1: Preprocess raw data
│   ├── train.py          # Step 2: Train the OCR model
│   ├── infer.py          # Step 3: Run inference on new images
│
│── requirements.txt      # Python dependencies
│── README.md             # Documentation
│── .gitignore            # Ignored files (venv, cache, datasets)
```

### 🛠️ Usage
``` bash
Step 1: Prepare dataset

Create folders:
data/raw/train/
data/raw/test/

Place your dataset into the above folders.
Each sample must include:

A .jpg image
A .json annotation file

Step 2: Run preprocessing
python scripts/preprocess.py

Step 3: Train the model
python scripts/train.py

Step 4: Run inference on new images
python scripts/infer.py --image path/to/image.jpg
```
### 🤝 Contributing
``` bash
1. Fork the repository

2. Create a new branch
    git checkout -b feature-name

3. Commit your changes
    git commit -m "Add feature"

4. Push to your branch
    git push origin feature-name

5. Open a Pull Request
```
