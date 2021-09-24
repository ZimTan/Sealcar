import torch
from torch import nn

class LSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()

        self.hidden_dim = hidden_dim # Hidden dimensions
        self.layer_dim = layer_dim # Number of hidden layers


        # RNN LSTM model building:
        # ouput size formula: (w - k + 2p) / s + 1
        self.cnn = nn.Sequential(
                nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True),
                nn.Dropout(0.2),

        )

        self.dense1 = nn.Linear(hidden_dim, output_dim) 

        self.act = nn.ReLU(inplace=True)
        

    def forward(self, x):

        return x

