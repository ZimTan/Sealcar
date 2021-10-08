from torch import nn
import torch

class CNNSeg(nn.Module):

    def __init__(self):
        super().__init__()

        # (w - k + 2p) / s + 1
        self.conv_seg = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2, stride=1), #3x120x160
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=5, padding=2, stride=1), #24x60x80
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1, stride=1), #36x30x40
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1, stride=1), #48x15x20
            nn.ReLU(inplace=True),
        )

        #self.act = nn.ReLU(inplace=True)


    def forward(self, x):

        x = self.conv_seg(x)

        return x




