import os
import torch
import pandas as pd
from dataclasses import dataclass
from torch.utils.data import Dataset
from transformers import (VisionEncoderDecoderModel, TrOCRProcessor,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments,
                          default_data_collator)
import evaluate
from PIL import Image

@dataclass(frozen=True)
class Config:
    data_root = "data/processed"
    model_name = "microsoft/trocr-small-handwritten"
    epochs = 1
    batch_size = 4
    learning_rate = 5e-5
    output_dir = "models/trocr_handwritten"

class OCRDataset(Dataset):
    def __init__(self, csv_path, image_dir, processor):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(os.path.join(self.image_dir, row['image'])).convert("RGB")
        inputs = self.processor(image, return_tensors="pt")
        labels = self.processor.tokenizer(row['text'], return_tensors="pt",
                                          padding="max_length", truncation=True).input_ids
        inputs["labels"] = labels.squeeze()
        return inputs

def compute_cer(eval_pred):
    processor = TrOCRProcessor.from_pretrained(Config.model_name)
    cer = evaluate.load("cer")
    labels = eval_pred.label_ids
    preds = eval_pred.predictions
    preds_str = processor.batch_decode(preds, skip_special_tokens=True)
    labels[labels == -100] = processor.tokenizer.pad_token_id
    labels_str = processor.batch_decode(labels, skip_special_tokens=True)
    return {"cer": cer.compute(predictions=preds_str, references=labels_str)}

def main():
    proc = TrOCRProcessor.from_pretrained(Config.model_name)
    model = VisionEncoderDecoderModel.from_pretrained(Config.model_name)
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = OCRDataset("data/train.csv", "data/processed/train", proc)
    valid_dataset = OCRDataset("data/test.csv", "data/processed/test", proc)

    training_args = Seq2SeqTrainingArguments(
        output_dir=Config.output_dir,
        num_train_epochs=Config.epochs,
        per_device_train_batch_size=Config.batch_size,
        per_device_eval_batch_size=Config.batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        save_total_limit=2,
        learning_rate=Config.learning_rate,
        report_to="none"
    )

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=proc.feature_extractor,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=default_data_collator,
        compute_metrics=compute_cer
    )
    trainer.train()

if __name__ == "__main__":
    main()
