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




