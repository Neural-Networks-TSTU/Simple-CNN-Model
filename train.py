import os
import logging
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy
from tqdm import tqdm
from cnn_model import CNNClassifier
import matplotlib.pyplot as plt

DATA_DIR       = "data_split"           
BATCH_SIZE     = 32
NUM_EPOCHS     = 25
LEARNING_RATE  = 1e-3
PATIENCE       = 5 
DELTA          = 1e-4 
INPUT_SIZE     = 96              
NUM_CLASSES    = 7
SAVE_DIR       = "checkpoints"
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")


os.makedirs(SAVE_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")



class Trainer:
    def __init__(self, model, device, classes):
        self.model   = model.to(device)
        self.device  = device
        self.classes = classes
        self.crit    = nn.CrossEntropyLoss()
        self.opt     = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.opt, mode='max', factor=0.5, patience=2)
        self.metric = MulticlassAccuracy(num_classes=NUM_CLASSES).to(device)

        self.best_acc = 0.0
        self.epochs_since_impr = 0

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": []
        }

    def train_epoch(self, loader):
        self.model.train()
        running_loss = 0.0
        self.metric.reset()

        pbar = tqdm(loader, desc="  train", leave=False)
        for imgs, labels in pbar:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            self.opt.zero_grad()

            logits = self.model(imgs)
            loss   = self.crit(logits, labels)
            loss.backward()
            self.opt.step()

            running_loss += loss.item()
            preds = logits.argmax(dim=1)
            self.metric.update(preds, labels)
            pbar.set_postfix({
                "loss": f"{loss.item():.3f}",
                "acc":  f"{self.metric.compute().item():.3f}"
            })

        return running_loss/len(loader), self.metric.compute().item()

    def validate(self, loader):
        self.model.eval()
        running_loss = 0.0
        self.metric.reset()

        with torch.no_grad():
            pbar = tqdm(loader, desc="  valid", leave=False)
            for imgs, labels in pbar:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                logits = self.model(imgs)
                loss   = self.crit(logits, labels)

                running_loss += loss.item()
                preds = logits.argmax(dim=1)
                self.metric.update(preds, labels)
                pbar.set_postfix({
                    "loss": f"{loss.item():.3f}",
                    "acc":  f"{self.metric.compute().item():.3f}"
                })

        acc = self.metric.compute().item()
        logging.info(f"  → Val loss: {running_loss/len(loader):.4f}, acc: {acc:.4f}")
        return running_loss/len(loader), acc
    
    def plot_history(self):
        epochs = range(1, len(self.history["train_loss"]) + 1)
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.history["train_loss"], label='Train Loss')
        plt.plot(epochs, self.history["val_loss"], label='Val Loss')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss Over Epochs")
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.history["train_acc"], label='Train Acc')
        plt.plot(epochs, self.history["val_acc"], label='Val Acc')
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy Over Epochs")
        plt.legend()

        plt.tight_layout()
        fig_path = Path(SAVE_DIR) / "training_curves.png"
        plt.savefig(fig_path)
        plt.close()
        logging.info(f"Saved training curves → {fig_path}")

    def save(self, epoch, is_best=False):
        fn = Path(SAVE_DIR) / ("best.pth" if is_best else f"epoch_{epoch}.pth")
        torch.save({
            'model_state': self.model.state_dict(),
            'classes': self.classes
        }, fn)
        logging.info(f"Saved model with classes: {self.classes} → {fn}")

    def fit(self, train_loader, val_loader):
        for epoch in range(1, NUM_EPOCHS+1):
            logging.info(f"Epoch {epoch}/{NUM_EPOCHS}")
            train_loss, train_acc = self.train_epoch(train_loader)
            logging.info(f"  → Train loss: {train_loss:.4f}, acc: {train_acc:.4f}")

            val_loss, val_acc = self.validate(val_loader)
            self.scheduler.step(val_acc)

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            if val_acc > self.best_acc + DELTA:
                self.best_acc = val_acc
                self.epochs_since_impr = 0
                self.save(epoch, is_best=True)
            else:
                self.epochs_since_impr += 1
                logging.info(f"  No improvement for {self.epochs_since_impr} epochs")
                if self.epochs_since_impr >= PATIENCE:
                    logging.info("Early stopping triggered.")
                    break

        self.save(epoch, is_best=False)
        logging.info("Training complete.")
        self.plot_history()


def main():
    tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor()
    ])

    train_dir = os.path.join(DATA_DIR, "train")
    classes = sorted([
        d for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d))
    ])

    train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=tf)
    val_ds   = datasets.ImageFolder(os.path.join(DATA_DIR, "val"),   transform=tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model   = CNNClassifier()
    trainer = Trainer(model, DEVICE, classes)
    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    main()
