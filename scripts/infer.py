import os
import torch
from PIL import Image
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
def infer(image_path, model_dir, model_name="microsoft/trocr-small-handwritten"):
    proc = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_dir)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    img = Image.open(image_path).convert("RGB")
    pixel_values = proc(img, return_tensors="pt").pixel_values.to(model.device)
    outputs = model.generate(pixel_values)
    return proc.batch_decode(outputs, skip_special_tokens=True)[0]
if __name__ == "__main__":
    img_path = "path/to/sample.jpg"
    print("Prediction:", infer(img_path, "models/trocr_handwritten/checkpoint-xxx"))
