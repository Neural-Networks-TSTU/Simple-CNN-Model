import torch
import torch.nn as nn
import torch.nn.functional as F

class SepConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, padding=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=kernel, padding=padding, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        return F.relu(self.bn(x))

class SEBlock(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        self.fc1 = nn.Linear(ch, ch // r, bias=False)
        self.fc2 = nn.Linear(ch // r, ch, bias=False)

    def forward(self, x):
        s = x.mean(dim=(-2, -1)) 
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s)).unsqueeze(-1).unsqueeze(-1)
        return x * s


class LightCNNClassifier(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        # SepConv: 1→32→64→128
        self.sep1 = SepConv(1,  32)
        self.sep2 = SepConv(32, 64)
        self.sep3 = SepConv(64,128)
        self.pool = nn.MaxPool2d(2)
        self.se1   = SEBlock(32)
        self.se2   = SEBlock(64)
        self.se3   = SEBlock(128)
        self.gap   = nn.AdaptiveAvgPool2d(1)
        self.drop  = nn.Dropout(0.2)
        self.fc    = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(self.sep1(x)); x = self.se1(x)
        x = self.pool(self.sep2(x)); x = self.se2(x)
        x = self.pool(self.sep3(x)); x = self.se3(x)
        x = self.gap(x).view(x.size(0), -1)
        x = self.drop(x)
        return self.fc(x)
