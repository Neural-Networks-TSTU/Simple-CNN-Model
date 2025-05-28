import os
import pandas as pd
import torch
from predict import load_model, preprocess, predict
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, classes = load_model("checkpoints/best.pth", device, "cnnv2")


df = pd.read_csv("test/labels.csv")
correct = total = 0

for _, row in df.iterrows():
    filename = row["filename"]
    expected = row["label"]
    img_path = Path("test/images") / filename

    tensor = preprocess(str(img_path)).to(device)
    label, prob = predict(model, tensor, device, classes)

    total += 1
    if label == expected:
        correct += 1
    else:
        print(f"[FAIL] {filename}: predicted '{label}', expected '{expected}'")

accuracy = correct / total
print(f"\nAccuracy: {correct}/{total} ({accuracy:.2%})")

if accuracy < 1.0:
    raise SystemExit("Some predictions failed!")
