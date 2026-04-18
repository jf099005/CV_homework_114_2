# ============================================================================
# File: model.py
# Date: 2026-03-27
# Author: TA
# Description: Model architecture.
# ============================================================================

import torch
import torch.nn as nn
import torchvision.models as models

# class MyNet(nn.Module): 
class MyNet(nn.Module): 
    def __init__(self):
        super(MyNet, self).__init__()
        
        # 特徵提取層 (Feature Extractor)
        self.features = nn.Sequential(
            # 第一段：32x32 -> 16x16
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第二段：16x16 -> 8x8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 第三段：8x8 -> 4x4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 分類層 (Classifier)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.features(x)
        # Flatten: (Batch, 256, 4, 4) -> (Batch, 256*4*4)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()

        super(ResNet18, self).__init__()

        self.resnet = models.resnet18( weights=models.ResNet18_Weights.DEFAULT )
        # self.resnet = models.resnet18(weights=None)

        # ✅ 修改第一層 conv (原本 kernel=7, stride=2 → 太大)
        self.resnet.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        # ✅ 移除 maxpool（避免過早 downsample）
        self.resnet.maxpool = nn.Identity()

        # FC layer 改成 CIFAR-10
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 10)

    def forward(self, x):
        return self.resnet(x)
    
if __name__ == '__main__':
    model = ResNet18()
    print(model)
