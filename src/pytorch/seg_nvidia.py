from torch import nn
import torch

class CNNSeg(nn.Module):

    def __init__(self):
        super().__init__()

        # (w - k + 2p) / s + 1
        self.conv_seg = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2), #3x120x160
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(16, 32, kernel_size=5, padding=2), #16x60x80
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),

            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2), #
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2)
        )

        #self.act = nn.ReLU(inplace=True)


    def forward(self, x):

        x = self.conv_seg(x)

        return x

class SegNvidia(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv_seg = CNNSeg()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=5, padding=2, stride=2), #3x120x160
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 36, kernel_size=5, padding=2, stride=2), #24x60x80
            nn.ReLU(inplace=True),
            nn.Conv2d(36, 48, kernel_size=3, padding=2, stride=2), #36x30x40
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 64, kernel_size=3, padding=2, stride=2), #48x15x20
        )

        self.linear1 = nn.Linear(64*9*12, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 128)
        self.linear4 = nn.Linear(128, 2)

        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.5)

        self.act = nn.ReLU(inplace=True)
        self.act2 = nn.Tanh()


    def forward(self, x):

        x = self.conv_seg(x)
        x = self.conv(x)

        b, c, h, w = x.shape
        x = x.view(b, c * h * w) #Flatten

        x = self.dropout1(x)

        x = self.act(self.linear1(x))
        x = self.dropout2(x)
        x = self.act(self.linear2(x))
        x = self.act2(self.linear3(x))
        x = self.linear4(x)

        return x






