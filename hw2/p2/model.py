# ============================================================================
# File: model.py
# Date: 2026-03-27
# Author: TA
# Description: Model architecture.
# ============================================================================

import torch
import torch.nn as nn
import torchvision.models as models

class MyNet(nn.Module): 
    def __init__(self):
        super(MyNet, self).__init__()
        
        ################################################################
        # TODO:                                                        #
        # Define your CNN model architecture. Note that the first      #
        # input channel is 3, and the output dimension is 10 (class).  #
        ################################################################

        pass

    def forward(self, x):

        ##########################################
        # TODO:                                  #
        # Define the forward path of your model. #
        ##########################################

        pass
    
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
    def forward(self, x):
        return self.resnet(x)
    
if __name__ == '__main__':
    model = ResNet18()
    print(model)
