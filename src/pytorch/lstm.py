import torch
from torch import nn
from torchvision.models import squeezenet1_1

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


class LSTM(nn.Module):

    def __init__(self):
        super(LSTM, self).__init__()

        # (w - k + 2p) / s + 1
        self.cnn = squeezenet1_1()
        self.cnn.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2))
        self.cnn.classifier[1] = nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))

        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 2)

        self.act1 = nn.ReLU(inplace=True)


    def forward(self, x):

        x = self.cnn(x)

        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)

        return x
