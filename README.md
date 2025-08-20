# Handwritten OCR Project

## Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Steps
1. Put dataset under `data/raw/train` and `data/raw/test` with `.jpg` and `.json` pairs.
2. Run preprocessing:
```bash
python scripts/preprocess.py
```
3. Train model:
```bash
python scripts/train.py
```
4. Run inference:
```bash
python scripts/infer.py
```
