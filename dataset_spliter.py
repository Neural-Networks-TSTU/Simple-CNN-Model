import os
import shutil
from sklearn.model_selection import train_test_split

SRC_DIR = "data"
DST_DIR = "data_split"
TRAIN_RATIO = 0.8

os.makedirs(f"{DST_DIR}/train", exist_ok=True)
os.makedirs(f"{DST_DIR}/val", exist_ok=True)

for class_name in os.listdir(SRC_DIR):
    class_path = os.path.join(SRC_DIR, class_name)
    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)
    train_imgs, val_imgs = train_test_split(images, train_size=TRAIN_RATIO, random_state=42)

    for img in train_imgs:
        src = os.path.join(class_path, img)
        dst = os.path.join(DST_DIR, "train", class_name)
        os.makedirs(dst, exist_ok=True)
        shutil.copy(src, os.path.join(dst, img))

    for img in val_imgs:
        src = os.path.join(class_path, img)
        dst = os.path.join(DST_DIR, "val", class_name)
        os.makedirs(dst, exist_ok=True)
        shutil.copy(src, os.path.join(dst, img))
