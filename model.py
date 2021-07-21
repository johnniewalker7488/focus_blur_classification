import torch
import torch.nn as nn


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1):
        super(CNNBlock, self).__init__()
        
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.relu(self.bn2(self.conv(self.bn1(x))))
    

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ResBlock, self).__init__()
        self.stride = stride
        self.conv1 = CNNBlock(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = CNNBlock(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.downscale = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)
                
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        x = self.downscale(x)
        return out + x
    

class ResNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ResNet, self).__init__()
        
        self.conv1 = CNNBlock(in_channels, 8, kernel_size=7, stride=1, padding=3)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.res1 = ResBlock(8, 16, stride=2)
        self.res2 = ResBlock(16, 32, stride=2)
#         self.res3 = ResBlock(32, 64, stride=2)
#         self.res4 = ResBlock(64, 128, stride=2)
        
        self.pool = nn.AdaptiveMaxPool2d(4)
        self.bn = nn.BatchNorm1d(512)
#         self.fc1 = nn.Linear(2048, 512)
#         self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        x = self.bn(x)
        x = self.fc2(x)
        return x