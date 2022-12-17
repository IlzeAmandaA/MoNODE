import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules.rnn import GRU
import torch.nn.functional as F

# import os, sys
# sys.path.append("..") # Adds higher directory to python modules path.
from model.core.mlp import MLP

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

# class GRUEncoder(nn.Module):
#     def __init__(self, output_dims, input_dim, rnn_output_size=20, H=50):
#         super(GRUEncoder, self).__init__()
#         self.is_list_output = type(output_dims) == list
#         if not self.is_list_output:
#             output_dims = [output_dims]
#         self.input_dim   = input_dim
#         self.output_dims = output_dims
#         self.rnn_output_size = rnn_output_size # number of outputs per output_dim

#         rnn_hidden_to_latent_nets = [nn.Sequential(nn.Linear(self.rnn_output_size, H), nn.ReLU(True), nn.Linear(H, d))
#                                     for d in self.output_dims]
        
#         self.rnn_hiddens_to_latents = nn.ModuleList(rnn_hidden_to_latent_nets)
#         self.gru = GRU(self.input_dim, self.rnn_output_size*len(output_dims))

#     def forward(self, data, run_backwards=True):
#         data = data.permute(1,0,2)  # (N, T, D) -> (T, N, D)
#         if run_backwards:
#             data = torch.flip(data, [0])  # (T, N, D)

#         outputs, _ = self.gru(data)  # (T, N, K)

#         q_outputs,idx = [],0
#         for net in self.rnn_hiddens_to_latents:
#             output  = outputs[-1][:, idx:idx+self.rnn_output_size]
#             idx    += self.rnn_output_size
#             net_out = net(output)
#             q_outputs.append(net_out)
        
#         if not self.is_list_output:
#             q_outputs = q_outputs[0]

#         return q_outputs