import torch
from torch import nn


class LSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTM, self).__init__()

        self.hidden_dim = hidden_dim # Hidden dimensions
        self.layer_dim = layer_dim # Number of hidden layers


        # RNN LSTM model building:
        # ouput size formula: (w - k + 2p) / s + 1
        self.lstm = nn.Sequential(
                    nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True),
                    nn.Dropout(0.2),
                )

        self.dense1 = nn.Linear(hidden_dim, output_dim)

        self.act = nn.ReLU(inplace=True)


    def forward(self, x):

        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # 28 time steps
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        # out.size() --> 100, 28, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states! 
        out = self.fc(out[:, -1, :]) 
        # out.size() --> 100, 10

        return out




