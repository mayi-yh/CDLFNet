import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from model.RefineNet.ResNet import ResNet


class ResidualConvUnit(nn.ModuleList):
    def __init__(self, in_channels):
        super(ResidualConvUnit, self).__init__()
        for i in range(4):
            self.append(
                nn.Sequential(
                    nn.ReLU(),
                    nn.Conv2d(in_channels // (2 ** (3 - i)), in_channels // (2 ** (3 - i)), 3, padding=1, bias=False),
                    nn.BatchNorm2d(in_channels // (2 ** (3 - i))),
                    nn.ReLU(),
                    nn.Conv2d(in_channels // (2 ** (3 - i)), in_channels // (2 ** (3 - i)), 3, padding=1, bias=False),
                )
            )

    def forward(self, x):
        outs = []
        for index, module in enumerate(self):
            x1 = module(x[index])
            x1 = x[index] + x1
            outs.append(x1)
        return outs


def un_pool(input, scale):
    return F.interpolate(input, scale_factor=scale, mode='bilinear', align_corners=True)


class MultiResolutionFusion(nn.ModuleList):
    def __init__(self, in_channels, out_channels, scale_factors=[1, 2, 4, 8]):
        super(MultiResolutionFusion, self).__init__()
        self.scale_factors = scale_factors

        for index, scale in enumerate(scale_factors):
            self.append(
                nn.Sequential(
                    nn.Conv2d(in_channels // 2 ** (len(scale_factors) - index - 1), out_channels, kernel_size=3,
                              padding=1)
                )
            )

    def forward(self, x):
        outputs = []
        for index, module in enumerate(self):
            xi = module(x[index])
            xi = un_pool(xi, scale=self.scale_factors[index])
            outputs.append(xi)
        return outputs[0] + outputs[1] + outputs[2] + outputs[3]


class ChainedResidualPool(nn.ModuleList):
    def __init__(self, in_channels, blocks=4):
        super(ChainedResidualPool, self).__init__()
        self.in_channels = in_channels
        self.blocks = blocks
        self.relu = nn.ReLU()
        for i in range(blocks):
            self.append(
                nn.Sequential(
                    nn.MaxPool2d(kernel_size=5, stride=1, padding=2),
                    nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1, bias=False),
                )
            )

    def forward(self, x):
        x = self.relu(x)
        path = x
        for index, CRP in enumerate(self):
            path = CRP(path)
            x = x + path
        return x


class RefineNet(nn.Module):
    def __init__(self, num_classes=3):
        super(RefineNet, self).__init__()
        self.backbone = ResNet.resnet50()
        self.final = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, num_classes, kernel_size=3, padding=1, bias=False),
        )
        self.ResidualConvUnit = ResidualConvUnit(2048)
        self.MultiResolutionFusion = MultiResolutionFusion(2048, 256)
        self.ChainedResidualPool = ChainedResidualPool(256)

    def forward(self, x):
        x = self.backbone(x)
        x = self.ResidualConvUnit(x)
        x = self.MultiResolutionFusion(x)
        x = self.ChainedResidualPool(x)
        x = un_pool(x, scale=4)
        x = self.final(x)
        return x