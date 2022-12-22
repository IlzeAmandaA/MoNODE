import torch
import torch.nn as nn
from model.core.vae import EncoderCNN, EncoderRNN

class INV_ENC(nn.Module):
    def __init__(self, task, last_layer_gp=None, n_filt=8, inv_latent_dim=10, rnn_hidden=10, device='cpu'):
        super(INV_ENC, self).__init__()
        self.last_layer_gp = last_layer_gp
        if task=='rot_mnist' or task=='mov_mnist':
            self.inv_encoder = InvariantEncoderCNN(task, 'dirac', inv_latent_dim, n_filt).to(device)
        elif task=='sin':
            data_dim = 1
            self.inv_encoder = InvariantEncoderRNN(data_dim, Tin=10, rnn_hidden=rnn_hidden, enc_out_dim=inv_latent_dim, out_distr='dirac').to(device)
    
    @property
    def is_last_layer_gp(self):
        return self.last_layer_gp is not None

    def kl(self):
        if self.is_last_layer_gp:
            return self.last_layer_gp.kl()
        else:
            return torch.zeros(1) * 0.0

    def forward(self, X, L=1):
        ''' 
            X is [N,T,nc,d,d] or [N,T,q] 
            returns [L,N,T,q]
        '''
        c = self.inv_encoder(X) # N,Tinv,q
        if self.is_last_layer_gp:
            [N_,T_,q_] = c.shape
            c = c.reshape(N_*T_,q_)
            cL = torch.stack([self.last_layer_gp(c) for _ in range(L)]) # 
            return cL.reshape(L,N_,T_,q_)
        else:
            return c.repeat([L,1,1,1]) # L,N,T,q


class InvariantEncoderCNN(EncoderCNN):
    def __init__(self, task, out_distr='dirac', enc_out_dim=16, n_filt=8, n_in_channels=1):
        super().__init__(task, out_distr=out_distr, enc_out_dim=enc_out_dim, n_filt=n_filt, n_in_channels=n_in_channels)
    def forward(self,X):
        [N,T,nc,d,d] = X.shape
        X = X.reshape(N*T,nc,d,d)
        X_out = super().forward(X) # N*T,_
        return X_out.reshape(N,T,self.enc_out_dim)

class InvariantEncoderRNN(EncoderRNN):
    def __init__(self, input_dim, Tin=None, rnn_hidden=10, enc_out_dim=16, out_distr='dirac'):
        super(InvariantEncoderRNN, self).__init__(input_dim, Tin=Tin, rnn_hidden=rnn_hidden, enc_out_dim=enc_out_dim, out_distr=out_distr)
    def forward(self, X, ns=5):
        [N,T,d] = X.shape
        Tin = T//2 if self.Tin is None else self.Tin
        Tin = min(Tin,T)
        X   = X.repeat([ns,1,1])
        t0s = torch.randint(0,T-Tin+1,[ns*N]) 
        X   = torch.stack([X[n,t0:t0+Tin] for n,t0 in enumerate(t0s)]) # ns*N,T//2,d
        X_out = super().forward(X) # ns*N,enc_out_dim
        return X_out.reshape(ns,N,self.enc_out_dim).permute(1,0,2) # N,ns,enc_out_dim
        