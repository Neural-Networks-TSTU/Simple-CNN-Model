import os
import json
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import logging
import argparse
from tqdm import tqdm
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import Counter

def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

DATA_DIR       = "data_split"
BATCH_SIZE     = 32
NUM_EPOCHS     = 20
LEARNING_RATE  = 1e-3
PATIENCE       = 5 
DELTA          = 0.005
INPUT_SIZE     = 128
NUM_CLASSES    = 7
SAVE_DIR       = "checkpoints"
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(SAVE_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(SAVE_DIR, "training.log"), encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class BBoxMaskDataset(Dataset):
    def __init__(self, root: str, classes: list, transform=None):
        self.transform = transform
        self.samples = []
        for class_idx, cls in enumerate(classes):
            cls_dir = Path(root) / cls
            for img_path in cls_dir.iterdir():
                if img_path.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.bmp'):
                    continue
                json_path = img_path.with_suffix('.json')
                if not json_path.exists():
                    raise FileNotFoundError(f"Нет .json-аннотации для {img_path}")
                self.samples.append((img_path, json_path, class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, ann_path, label = self.samples[idx]
        img = Image.open(img_path).convert('L')
        img_np = np.array(img, dtype=np.uint8)

        with open(ann_path, 'r') as f:
            ann = json.load(f)
        bb = ann['bbox']
        x_min, y_min = bb['x_min'], bb['y_min']
        x_max, y_max = bb['x_max'], bb['y_max']

        h, w = img_np.shape
        x_min, x_max = max(0, x_min), min(w-1, x_max)
        y_min, y_max = max(0, y_min), min(h-1, y_max)
        if x_max <= x_min or y_max <= y_min:
            logging.warning(f"Некорректный bounding box в {img_path}")
            x_min, y_min, x_max, y_max = 0, 0, w-1, h-1

        mask = np.zeros_like(img_np, dtype=np.uint8)
        mask[y_min:y_max+1, x_min:x_max+1] = 1
        masked_np = img_np * mask

        masked_img = Image.fromarray(masked_np)
        if self.transform:
            masked_img = self.transform(masked_img)

        return masked_img, label

class Trainer:
    def __init__(self, model, device, classes, optimizer_type='sgd'):
        self.model   = model.to(device)
        self.device  = device
        self.classes = classes
        self.crit    = nn.CrossEntropyLoss()
        if optimizer_type.lower() == 'adam':
            self.opt = optim.Adam(self.model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        elif optimizer_type.lower() == 'sgd':
            self.opt = optim.SGD(self.model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=1e-4)
        else:
            raise ValueError("Неподдерживаемый оптимизатор")
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.opt, mode='min', factor=0.5, patience=2)
        self.metric_acc = MulticlassAccuracy(num_classes=NUM_CLASSES).to(device)
        self.metric_f1 = MulticlassF1Score(num_classes=NUM_CLASSES).to(device)
        self.best_acc = 0.0
        self.epochs_since_impr = 0
        self.history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "train_f1": [], "val_f1": []}

    def train_epoch(self, loader):
        self.model.train()
        running_loss = 0
        self.metric_acc.reset()
        self.metric_f1.reset()
        pbar = tqdm(loader, desc="  train", leave=False)
        for imgs, labels in pbar:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            self.opt.zero_grad()
            logits = self.model(imgs)
            loss = self.crit(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.opt.step()
            running_loss += loss.item()
            preds = logits.argmax(dim=1)
            self.metric_acc.update(preds, labels)
            self.metric_f1.update(preds, labels)
            pbar.set_postfix({"loss": f"{loss.item():.3f}", "acc": f"{self.metric_acc.compute().item():.3f}", "f1": f"{self.metric_f1.compute().item():.3f}"})
        return running_loss/len(loader), self.metric_acc.compute().item(), self.metric_f1.compute().item()

    def validate(self, loader):
        self.model.eval()
        running_loss = 0
        self.metric_acc.reset()
        self.metric_f1.reset()
        all_preds, all_labels = [], []
        with torch.no_grad():
            pbar = tqdm(loader, desc="  valid", leave=False)
            for imgs, labels in pbar:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                logits = self.model(imgs)
                loss = self.crit(logits, labels)
                running_loss += loss.item()
                preds = logits.argmax(dim=1)
                self.metric_acc.update(preds, labels)
                self.metric_f1.update(preds, labels)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                pbar.set_postfix({"loss": f"{loss.item():.3f}", "acc": f"{self.metric_acc.compute().item():.3f}", "f1": f"{self.metric_f1.compute().item():.3f}"})
        acc = self.metric_acc.compute().item()
        f1 = self.metric_f1.compute().item()
        logging.info(f"  → Val loss: {running_loss/len(loader):.4f}, acc: {acc:.4f}, f1: {f1:.4f}")
        return running_loss/len(loader), acc, f1, all_preds, all_labels

    def plot_history(self):
        epochs = range(1, len(self.history["train_loss"])+1)
        plt.figure(figsize=(12,5))
        plt.subplot(1,3,1)
        plt.plot(epochs, self.history["train_loss"], label='Train Loss')
        plt.plot(epochs, self.history["val_loss"], label='Val Loss')
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
        plt.subplot(1,3,2)
        plt.plot(epochs, self.history["train_acc"], label='Train Acc')
        plt.plot(epochs, self.history["val_acc"], label='Val Acc')
        plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()
        plt.subplot(1,3,3)
        plt.plot(epochs, self.history["train_f1"], label='Train F1')
        plt.plot(epochs, self.history["val_f1"], label='Val F1')
        plt.xlabel("Epoch"); plt.ylabel("F1 Score"); plt.legend()
        plt.tight_layout()
        fig_path = Path(SAVE_DIR) / "training_curves.png"
        plt.savefig(fig_path); plt.close()
        logging.info(f"Saved training curves → {fig_path}")

    def save(self, epoch, is_best=False):
        fn = Path(SAVE_DIR) / ("best.pth" if is_best else f"epoch_{epoch}.pth")
        torch.save({'model_state': self.model.state_dict(),
                    'classes': self.classes}, fn)

    def fit(self, train_loader, val_loader):
        for epoch in range(1, NUM_EPOCHS+1):
            logging.info(f"Epoch {epoch}/{NUM_EPOCHS}")
            t_loss, t_acc, t_f1 = self.train_epoch(train_loader)
            logging.info(f"  → Train loss: {t_loss:.4f}, acc: {t_acc:.4f}, f1: {t_f1:.4f}")
            v_loss, v_acc, v_f1, v_preds, v_labels = self.validate(val_loader)
            self.scheduler.step(v_loss)
            self.history["train_loss"].append(t_loss)
            self.history["train_acc"].append(t_acc)
            self.history["train_f1"].append(t_f1)
            self.history["val_loss"].append(v_loss)
            self.history["val_acc"].append(v_acc)
            self.history["val_f1"].append(v_f1)
            if v_acc > self.best_acc + DELTA:
                self.best_acc = v_acc
                self.epochs_since_impr = 0
                self.save(epoch, is_best=True)
            else:
                self.epochs_since_impr += 1
                logging.info(f"  No improvement for {self.epochs_since_impr} epochs")
                if self.epochs_since_impr >= PATIENCE:
                    logging.info("Early stopping triggered.")
                    break
            self.save(epoch, is_best=False)
        logging.info(f"Training complete. Best validation accuracy: {self.best_acc:.4f}")
        self.plot_history()
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(v_labels, v_preds)
        logging.info("Confusion Matrix:")
        logging.info("\n" + str(cm))

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def main():
    parser = argparse.ArgumentParser(description='Train CNN w/ BBox masking')
    parser.add_argument('--model-type', choices=['lightcnn', 'cnn', 'cnnv2', 'resnet18'], required=True)
    parser.add_argument('--optimizer', choices=['adam', 'sgd'], default='adam')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained weights for ResNet18')
    args = parser.parse_args()

    if args.model_type == 'resnet18':
        input_size = 224
    else:
        input_size = INPUT_SIZE

    train_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dir = os.path.join(DATA_DIR, "train")
    classes = sorted(d for d in os.listdir(train_dir)
                     if os.path.isdir(os.path.join(train_dir, d)))

    train_ds = BBoxMaskDataset(train_dir, classes, transform=train_transform)
    val_ds   = BBoxMaskDataset(os.path.join(DATA_DIR, "val"), classes, transform=val_transform)

    train_labels = [label for _, label in train_ds]
    train_class_counts = Counter(train_labels)
    logging.info("Class distribution in training set:")
    for cls_idx, count in train_class_counts.items():
        logging.info(f"  {classes[cls_idx]}: {count}")
    val_labels = [label for _, label in val_ds]
    val_class_counts = Counter(val_labels)
    logging.info("Class distribution in validation set:")
    for cls_idx, count in val_class_counts.items():
        logging.info(f"  {classes[cls_idx]}: {count}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    if args.model_type == 'resnet18':
        if args.pretrained:
            model = models.resnet18(pretrained=True)
            weight = model.conv1.weight.data.mean(dim=1, keepdim=True)
            model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            model.conv1.weight.data = weight
        else:
            model = models.resnet18(pretrained=False)
            model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    elif args.model_type == 'lightcnn':
        from models.lite_cnn_model import LightCNNClassifier
        model = LightCNNClassifier()
    elif args.model_type in ['cnn', 'cnnv2']:
        from models.cnn_model import CNNClassifier
        model = CNNClassifier()
    else:
        raise ValueError("Invalid model type")

    if not (args.model_type == 'resnet18' and args.pretrained):
        model.apply(init_weights)
    else:
        init_weights(model.fc)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Model has {num_params} trainable parameters")

    trainer = Trainer(model, DEVICE, classes, optimizer_type=args.optimizer)
    trainer.fit(train_loader, val_loader)

if __name__ == "__main__":
    main()