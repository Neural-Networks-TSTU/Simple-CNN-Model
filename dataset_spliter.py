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
IMG_SIZE    = 128
TARGET_COUNT = 400

transform = A.Compose([
    A.LongestMaxSize(max_size=IMG_SIZE),
    A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE, border_mode=0, value=0.0)
])

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
        else:
            x1 = random.randint(5, IMG_SIZE//2)
            y1 = random.randint(5, IMG_SIZE//2)
            x2 = random.randint(x1+10, IMG_SIZE-5)
            y2 = random.randint(y1+10, IMG_SIZE-5)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, random.choice([-1, 1, 2]))
    return img

for split in ("train", "val"):
    (Path(DST_DIR) / split).mkdir(parents=True, exist_ok=True)

print("Start balancing and splitting data")

for class_name in os.listdir(SRC_DIR):
    class_path = Path(SRC_DIR) / class_name
    if not class_path.is_dir():
        continue

    images = [p.name for p in class_path.iterdir() if p.suffix.lower() in ('.png','.jpg','.jpeg','.bmp')]
    total = len(images)
    imgs = []

    if total < TARGET_COUNT:
        for img_name in images:
            img = cv2.imread(str(class_path / img_name), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            imgs.append((img_name, img))
        to_gen = TARGET_COUNT - len(imgs)
        for i in range(to_gen):
            name, orig = random.choice(imgs)
            aug = transform(image=orig)["image"]
            imgs.append((f"{class_name}_aug_{i:03d}.png", aug))
    else:
        for img_name in random.sample(images, TARGET_COUNT):
            img = cv2.imread(str(class_path / img_name), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            imgs.append((img_name, img))

    names = [n for n, _ in imgs]
    train_names, val_names = train_test_split(names, train_size=TRAIN_RATIO, random_state=42)

    def save_images(names_list, split):
        for name in names_list:
            img = next(im for nm, im in imgs if nm == name)
            aug = transform(image=img)["image"]
            h, w = aug.shape

            ys, xs = np.where(aug > 0)
            if len(xs):
                x_min, x_max = xs.min(), xs.max()
                y_min, y_max = ys.min(), ys.max()
            else:
                x_min, y_min, x_max, y_max = 0, 0, w-1, h-1

            out_dir = Path(DST_DIR) / split / class_name
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / name
            cv2.imwrite(str(out_path), aug)

            ann = {
                "filename": name,
                "class": class_name,
                "processed_size": {"width": w, "height": h},
                "bbox": {"x_min": int(x_min), "y_min": int(y_min), "x_max": int(x_max), "y_max": int(y_max)},
                "path": str(out_path.resolve())
            }
            with open(out_path.with_suffix('.json'), 'w') as f:
                json.dump(ann, f, indent=4)

    save_images(train_names, "train")
    save_images(val_names, "val")

for split in ("train", "val"):
    texts = []
    for i in range(TARGET_COUNT):
        img = generate_trash_image()
        texts.append((f"trash_{i:04d}.png", img))

    # Сплит
    names = [n for n, _ in texts]
    train_names, val_names = train_test_split(names, train_size=TRAIN_RATIO, random_state=42)

    def save_trash(names_list, split):
        for name in names_list:
            img = next(im for nm, im in texts if nm == name)
            aug = transform(image=img)["image"]
            h, w = aug.shape

            ys, xs = np.where(aug > 0)
            if len(xs):
                x_min, x_max = xs.min(), xs.max()
                y_min, y_max = ys.min(), ys.max()
            else:
                x_min, y_min, x_max, y_max = 0, 0, w-1, h-1

            out_dir = Path(DST_DIR) / split / "TRASH"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / name
            cv2.imwrite(str(out_path), aug)

            ann = {
                "filename": name,
                "class": "trash",
                "processed_size": {"width": w, "height": h},
                "bbox": {"x_min": int(x_min), "y_min": int(y_min), "x_max": int(x_max), "y_max": int(y_max)},
                "path": str(out_path.resolve())
            }
            with open(out_path.with_suffix('.json'), 'w') as f:
                json.dump(ann, f, indent=4)

    save_trash(train_names, "train")
    save_trash(val_names,   "val")

print("Done!")
