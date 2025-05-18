import os
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
import albumentations as A

SRC_DIR     = "data"
DST_DIR     = "data_split"
TRAIN_RATIO = 0.8
IMG_SIZE    = 96

transform = A.Compose([
    A.LongestMaxSize(max_size=IMG_SIZE),
    A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE, border_mode=0, value=0.0)
])

os.makedirs(f"{DST_DIR}/train", exist_ok=True)
os.makedirs(f"{DST_DIR}/val",   exist_ok=True)

for class_name in os.listdir(SRC_DIR):
    class_path = Path(SRC_DIR) / class_name
    if not class_path.is_dir():
        continue

    images = [p.name for p in class_path.iterdir() if p.suffix.lower() in ('.png','.jpg','.jpeg','.bmp')]
    train_imgs, val_imgs = train_test_split(images, train_size=TRAIN_RATIO, random_state=42)

    for split, split_imgs in (("train", train_imgs), ("val", val_imgs)):
        for img_name in split_imgs:
            src_path = class_path / img_name
            img = cv2.imread(str(src_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Cannot read this file: {src_path}")
                continue

            augmented = transform(image=img)["image"]

            dst_dir = Path(DST_DIR) / split / class_name
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst_path = dst_dir / img_name

            cv2.imwrite(str(dst_path), augmented)

print("Success")
