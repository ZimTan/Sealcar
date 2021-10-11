import torch
from torch import nn
from torchvision.models import squeezenet1_1

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


class LSTM(nn.Module):

    def __init__(self):
        super(LSTM, self).__init__()

        # Load SqueezeNet v1.1 CNN as backbone:
        self.cnn = squeezenet1_1()
        self.cnn.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2))
        self.cnn.classifier[1] = nn.Conv2d(512, 300, kernel_size=(1, 1), stride=(1, 1))

        # Concatenate 3 layers of LSTM to CNN backbone:
        self.lstm = nn.LSTM(300, 256, num_layers=1, batch_first=True)

        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 2)

        self.act1 = nn.ReLU(inplace=True)


    def forward(self, x):
        # x have a diferent shape than in other CNN, the shape is (batch_size, seq_length, channel, height, width)

        hidden = None

        for t in range(x.size(1)):
            with torch.no_grad():
                x_cnn = self.cnn(x[:, t, :, :, :])

            out, hidden = self.lstm(x_cnn.unsqueeze(0), hidden)

        x_lstm = self.fc1(out[-1, :, :])
        x_lstm = self.act1(x_lstm)
        x_lstm = self.fc2(x_lstm)

        return x_lstm
