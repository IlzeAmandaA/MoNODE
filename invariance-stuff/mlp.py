# Copyright (c) 2022 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# Author: Cagatay Yildiz, cagatay.yildiz1@gmail.com

import torch
import torch.nn as nn
import numpy as np

def get_act(act="relu"):
    if act=="relu":         return nn.ReLU()
    elif act=="elu":        return nn.ELU()
    elif act=="celu":       return nn.CELU()
    elif act=="leaky_relu": return nn.LeakyReLU()
    elif act=="sigmoid":    return nn.Sigmoid()
    elif act=="tanh":       return nn.Tanh()
    elif act=="sin":        return torch.sin
    elif act=="linear":     return nn.Identity()
    elif act=='softplus':   return nn.modules.activation.Softplus()
    elif act=='swish':      return lambda x: x*torch.sigmoid(x)
    elif act=='lipswish':   return lambda x: 0.909 * torch.nn.functional.silu(x)
    else:                   return None


class BatchMLP(nn.Module):
    def __init__(self, B, n_in, n_out, L=2, H=100, act='relu'):
        super().__init__()
        layers_ins  = [n_in] + L*[H]
        layers_outs = L*[H] + [n_out]
        self.B      = B
        self.H      = H
        self.L      = L
        self.n_in   = n_in
        self.n_out  = n_out
        self.weights= nn.ParameterList([])
        self.biases = nn.ParameterList([])
        self.acts   = nn.ModuleList([])
        for i,(n_in,n_out) in enumerate(zip(layers_ins,layers_outs)):
            self.weights.append(nn.Parameter(torch.Tensor(B, n_in, n_out)))
            self.biases.append(nn.Parameter(torch.Tensor(B, 1, n_out)))
            self.acts.append(get_act(act) if i<L else get_act('linear')) # no act. in final layer
        self.reset_parameters()

    @property
    def device(self):
        return next(self.layers[0].parameters()).device

    def reset_parameters(self,gain=1.0):
        for i,(weight,bias) in enumerate(zip(self.weights,self.biases)):
            nn.init.xavier_uniform_(weight,gain)
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(bias, -bound, bound)

    def kl(self):
        return torch.zeros(1).to(self.device)

    def draw_f(self):
        return self

    def forward(self, x):
        for i,(act,w,b) in enumerate(zip(self.acts, self.weights, self.biases)):
            x = act(x@w + b)
        return x


class MLP(nn.Module):
    def __init__(self, n_in, n_out, L=2, H=100, act='relu', skip_con=False, dropout_rate=0.0):
        super().__init__()
        layers_ins  = [n_in] + L*[H]
        layers_outs = L*[H] + [n_out]
        self.H      = H
        self.L      = L
        self.n_in   = n_in
        self.n_out  = n_out
        self.layers = nn.ModuleList([])
        self.acts   = nn.ModuleList([])
        for i,(n_in,n_out) in enumerate(zip(layers_ins,layers_outs)):
            self.layers.append(nn.Linear(n_in,n_out))
            self.acts.append(get_act(act) if i<L else get_act('linear')) # no act. in final layer
        self.skip_con = skip_con
        self.reset_parameters()

    @property
    def device(self):
        return next(self.layers[0].parameters()).device

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def kl(self):
        return torch.zeros(1).to(self.device)

    def forward(self, x):
        for i,(act,layer) in enumerate(zip(self.acts,self.layers)):
            h = layer(x)
            h = act(h)
            x = x+h if self.skip_con and 0<i and i<self.n_hid_layers else h
        return x
    
    def draw_f(self):
        return self
