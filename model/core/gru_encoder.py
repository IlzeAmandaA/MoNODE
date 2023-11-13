import torch
import torch.nn as nn
from torch.nn.modules.rnn import GRU

class GRUEncoder(nn.Module):
    def __init__(self, output_dim, input_dim, rnn_output_size=20, H=50, act='relu'):
        super(GRUEncoder, self).__init__()
        self.input_dim  = input_dim
        self.output_dim = output_dim
        self.rnn_output_size = rnn_output_size # number of outputs per output_dim
        self.rnn_hidden_to_latent = nn.Sequential(nn.Linear(self.rnn_output_size, H), nn.ReLU(True), nn.Linear(H, output_dim))
        self.gru = GRU(self.input_dim, self.rnn_output_size)

    def forward(self, data, run_backwards=True):
        data = data.permute(1,0,2)  # (N, T, D) -> (T, N, D)
        if run_backwards:
            data = torch.flip(data, [0])  # (T, N, D)

        outputs, _ = self.gru(data)  # (T, N, K)
        return self.rnn_hidden_to_latent(outputs[-1]) # N,q
