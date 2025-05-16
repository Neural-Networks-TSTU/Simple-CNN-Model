import torch.nn as nn
import torch

INPUT_SIZE     = 96              
NUM_CLASSES    = 7

class CNNClassifier(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool  = nn.MaxPool2d(2,2)
        self.drop  = nn.Dropout(0.25)
        self.fc1   = nn.Linear(64 * (INPUT_SIZE//4) * (INPUT_SIZE//4), 128)
        self.fc2   = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # -> (32, INPUT/2, INPUT/2)
        x = self.pool(torch.relu(self.conv2(x)))  # -> (64, INPUT/4, INPUT/4)
        x = x.flatten(1)
        x = self.drop(torch.relu(self.fc1(x)))
        return self.fc2(x)