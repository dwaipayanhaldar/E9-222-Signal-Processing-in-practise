import torch
import torch.nn as nn

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)

class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNReLU(ch, ch, 3, 1, 1),
            ConvBNReLU(ch, ch, 3, 1, 1),
        )
    def forward(self, x):
        return x + self.block(x)

class ResNet9(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.prep = ConvBNReLU(3, 64, 3, 1, 1) 
        self.layer1 = nn.Sequential(
            ConvBNReLU(64, 128, 3, 1, 1),                     
            nn.MaxPool2d(2),                                  
        )
        self.res1 = ResidualBlock(128)
        self.layer2 = nn.Sequential(
            ConvBNReLU(128, 256, 3, 1, 1),                    
            nn.MaxPool2d(2),                                  
        )
        self.layer3 = nn.Sequential(
            ConvBNReLU(256, 512, 3, 1, 1),                     
            nn.MaxPool2d(2),                                   
        )
        self.res2 = ResidualBlock(512)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.prep(x)
        x = self.layer1(x)
        x = self.res1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.res2(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)