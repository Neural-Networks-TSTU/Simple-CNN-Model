import os
import cv2
import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import albumentations as A
import random

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

def generate_trash_image():
    img = np.full((IMG_SIZE, IMG_SIZE), 255, dtype=np.uint8)
    
    for _ in range(random.randint(1, 3)):
        shape_type = random.choice(['line', 'circle', 'rectangle'])
        color = random.randint(0, 75)
        
        if shape_type == 'line':
            cv2.line(img, 
                    (random.randint(5, IMG_SIZE-5), random.randint(5, IMG_SIZE-5)),
                    (random.randint(5, IMG_SIZE-5), random.randint(5, IMG_SIZE-5)),
                    color, random.randint(1, 3))
        
        elif shape_type == 'circle':
            cv2.circle(img,
                      (random.randint(20, IMG_SIZE-20), random.randint(20, IMG_SIZE-20)),
                      random.randint(5, 25),
                      color, random.choice([-1, 1, 2]))
        
        elif shape_type == 'rectangle':
            x1 = random.randint(5, IMG_SIZE//2)
            y1 = random.randint(5, IMG_SIZE//2)
            x2 = random.randint(x1+10, IMG_SIZE-5)
            y2 = random.randint(y1+10, IMG_SIZE-5)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, random.choice([-1, 1, 2]))
    
    return img

os.makedirs(f"{DST_DIR}/train/TRASH", exist_ok=True)
os.makedirs(f"{DST_DIR}/val/TRASH", exist_ok=True)

print("Start")

for class_name in os.listdir(SRC_DIR):
    class_path = Path(SRC_DIR) / class_name
    if not class_path.is_dir():
        continue

    images = [p.name for p in class_path.iterdir() 
              if p.suffix.lower() in ('.png','.jpg','.jpeg','.bmp')]
    train_imgs, val_imgs = train_test_split(images, train_size=TRAIN_RATIO, random_state=42)

    for split, split_imgs in (("train", train_imgs), ("val", val_imgs)):
        for img_name in split_imgs:
            src_path = class_path / img_name
            img = cv2.imread(str(src_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Cannot read this file: {src_path}")
                continue

            augmented = transform(image=img)["image"]
            h, w = augmented.shape[:2]

            ys, xs = np.where(augmented > 0)
            if len(xs) > 0:
                x_min, x_max = int(xs.min()), int(xs.max())
                y_min, y_max = int(ys.min()), int(ys.max())
            else:
                x_min, y_min, x_max, y_max = 0, 0, w-1, h-1

            dst_dir = Path(DST_DIR) / split / class_name
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst_path = dst_dir / img_name
            cv2.imwrite(str(dst_path), augmented)

            annotation = {
                "filename": img_name,
                "class": class_name,
                "processed_size": {"width": w, "height": h},
                "bbox": {
                    "x_min": x_min,
                    "y_min": y_min,
                    "x_max": x_max,
                    "y_max": y_max
                },
                "path": str(dst_path.resolve())
            }

            ann_path = dst_path.with_suffix('.json')
            with open(ann_path, 'w') as f:
                json.dump(annotation, f, indent=4)

for split, count in [("train", 160), ("val", 40)]:
    for i in range(count):
        img_name = f"trash_{i:04d}.png"
        img = generate_trash_image()
        
        augmented = transform(image=img)["image"]
        h, w = augmented.shape[:2]

        ys, xs = np.where(augmented > 0)
        if len(xs) > 0:
            x_min, x_max = int(xs.min()), int(xs.max())
            y_min, y_max = int(ys.min()), int(ys.max())
        else:
            x_min, y_min, x_max, y_max = 0, 0, w-1, h-1

        dst_dir = Path(DST_DIR) / split / "trash"
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_path = dst_dir / img_name
        cv2.imwrite(str(dst_path), augmented)

        annotation = {
            "filename": img_name,
            "class": "trash",
            "processed_size": {"width": w, "height": h},
            "bbox": {
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max
            },
            "path": str(dst_path.resolve())
        }

        ann_path = dst_path.with_suffix('.json')
        with open(ann_path, 'w') as f:
            json.dump(annotation, f, indent=4)

print("Done!") 
