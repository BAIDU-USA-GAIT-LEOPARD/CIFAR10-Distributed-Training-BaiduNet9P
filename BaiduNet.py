
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np

class ConvBN(nn.Module):
  def __init__(self, c_in, c_out):
    super(ConvBN, self).__init__()
    self.conv = nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn = nn.BatchNorm2d(c_out)

  def forward(self, x):
    return F.relu(self.bn(self.conv(x)))

class Residual(nn.Module):
  def __init__(self, c_in, c_out):
    super(Residual, self).__init__()
    self.pre = ConvBN(c_in, c_out)
    self.conv_bn1 = ConvBN(c_out, c_out)
    self.conv_bn2 = ConvBN(c_out, c_out)

  def forward(self, x):
    x = self.pre(x)
    x = F.max_pool2d(x, 2)
    return self.conv_bn2(self.conv_bn1(x)) + x

class BaiduNet(nn.Module):
    def __init__(self, num_classes=10):
        super(BaiduNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.layer1 = Residual(64, 128)
        self.layer2 = ConvBN(128, 256)
        self.layer3 = Residual(256, 384)
        self.fc = nn.Linear(384, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.max_pool2d(out, 8)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def BaiduNet9P():

    return BaiduNet()





















