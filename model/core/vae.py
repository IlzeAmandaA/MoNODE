import torch
import torch.nn as nn
from torch.distributions import Normal
from torchsummary import summary
from model.misc.torch_utils import Flatten, UnFlatten
from model.core.gru_encoder import GRUEncoder
from model.core.mlp import MLP
from model.core.conv import *
import numpy as np

EPSILON = 1e-5

def build_rot_mnist_cnn_enc(n_in_channels, n_filt):
    cnn =   nn.Sequential(
            nn.Conv2d(n_in_channels, n_filt, kernel_size=5, stride=2, padding=(2,2)), # 14,14
            nn.BatchNorm2d(n_filt),
            nn.ReLU(),
            nn.Conv2d(n_filt, n_filt*2, kernel_size=5, stride=2, padding=(2,2)), # 7,7
            nn.BatchNorm2d(n_filt*2),
            nn.ReLU(),
            nn.Conv2d(n_filt*2, n_filt*4, kernel_size=5, stride=2, padding=(2,2)),
            nn.BatchNorm2d(n_filt*4), # this is new
            nn.ReLU(),
            Flatten()
        )
    out_features = n_filt*4 * 4*4 # encoder output is [4*n_filt,4,4]
    return cnn, out_features

def build_rot_mnist_cnn_dec(n_filt, n_in):
    out_features = n_filt*4 * 4*4 # encoder output is [4*n_filt,4,4]
    cnn = nn.Sequential(
        nn.Linear(n_in, out_features),
        UnFlatten(4),
        nn.ConvTranspose2d(out_features//16, n_filt*8, kernel_size=3, stride=1, padding=(0,0)),
        nn.BatchNorm2d(n_filt*8),
        nn.ReLU(),
        nn.ConvTranspose2d(n_filt*8, n_filt*4, kernel_size=5, stride=2, padding=(1,1)),
        nn.BatchNorm2d(n_filt*4),
        nn.ReLU(),
        nn.ConvTranspose2d(n_filt*4, n_filt*2, kernel_size=5, stride=2, padding=(1,1), output_padding=(1,1)),
        nn.BatchNorm2d(n_filt*2),
        nn.ReLU(),
        nn.ConvTranspose2d(n_filt*2, 1, kernel_size=5, stride=1, padding=(2,2)),
        nn.Sigmoid(),
    )
    return cnn

def build_mov_mnist_cnn_enc(n_in_channels, n_filt):
	cnn = nn.Sequential(
            nn.Conv2d(n_in_channels, n_filt, kernel_size=5, stride=2, padding=(2,2)), # 32,32
            nn.BatchNorm2d(n_filt),
            nn.ReLU(),
            nn.Conv2d(n_filt, n_filt*2, kernel_size=5, stride=2, padding=(2,2)), # 16,16
            nn.BatchNorm2d(n_filt*2),
            nn.ReLU(),
            nn.Conv2d(n_filt*2, n_filt*4, kernel_size=5, stride=2, padding=(2,2)), # 8,8
            nn.BatchNorm2d(n_filt*4),
            nn.ReLU(),
            nn.Conv2d(n_filt*4, n_filt*8, kernel_size=5, stride=2, padding=(2,2)), # 4,4
            nn.BatchNorm2d(n_filt*8),
            nn.ReLU(),
            Flatten()
        )
	out_features = n_filt*8 * 4*4 
	return cnn, out_features

def build_mov_mnist_cnn_dec(n_filt, n_in):
    out_features = n_filt*8 * 4*4 # encoder output is [4*n_filt,4,4]
    cnn = nn.Sequential(
            nn.Linear(n_in, out_features),
            UnFlatten(4),
            nn.ConvTranspose2d(out_features//16, n_filt*8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(n_filt*8),
            nn.ReLU(),
            nn.ConvTranspose2d(n_filt*8, n_filt*4, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm2d(n_filt*4),
            nn.ReLU(),
            nn.ConvTranspose2d(n_filt*4, n_filt*2, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm2d(n_filt*2),
            nn.ReLU(),
            nn.ConvTranspose2d(n_filt*2, n_filt, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm2d(n_filt),
            nn.ReLU(),
            nn.ConvTranspose2d(n_filt, 1, kernel_size=6, stride=1, padding=2),
            nn.Sigmoid(),
        )
    return cnn


class VAE(nn.Module):
    def __init__(self, task, v_frames=1, n_filt=8, H=100, rnn_hidden=10, dec_act='relu', ode_latent_dim=8, inv_latent_dim=0, T_in=10, device='cpu', order=1, cnn_arch='cnn'):
        super(VAE, self).__init__()

        # task, out_distr='normal', enc_out_dim=16, n_filt=8, n_in_channels=1
        ### build encoder
        if task=='rot_mnist' or task=='mov_mnist':
            lhood_distribution = 'bernoulli'
            if cnn_arch == 'dcgan':
                self.encoder = encoder_factory('dcgan',nx=64, nc=1, nh=128, nf=n_filt, enc_out_dim=ode_latent_dim//order, T_in=T_in)
                self.decoder = decoder_factory('dcgan',nx=64, nc=1, ny=ode_latent_dim//order+inv_latent_dim, nf=64, skip=None)
            elif cnn_arch == 'cnn':
                self.encoder = PositionEncoderCNN(task=task, out_distr='normal', enc_out_dim=ode_latent_dim//order, n_filt=n_filt, T_in=T_in).to(device)
                self.decoder = Decoder(task, ode_latent_dim//order+inv_latent_dim, n_filt=n_filt, distribution=lhood_distribution).to(device)
            else:
                raise SystemExit('Invalid encoder/decoder selected')
            if order==2:
                self.encoder_v   = VelocityEncoderCNN(v_frames, task, 'normal', ode_latent_dim//order, n_filt).to(device)

        elif task=='sin' or task=='spiral' or task =='lv':
            lhood_distribution = 'normal'
            data_dim = 1 if task=='sin' else 2
            self.encoder = EncoderRNN(data_dim, Tin=T_in, rnn_hidden=rnn_hidden, enc_out_dim=ode_latent_dim, out_distr='normal').to(device)
            self.decoder = Decoder(task, ode_latent_dim, H=H, distribution=lhood_distribution, dec_out_dim=data_dim, act=dec_act).to(device)
            if order==2:
                self.encoder_v = EncoderRNN(data_dim, rnn_hidden=rnn_hidden, enc_out_dim=ode_latent_dim, out_distr='normal').to(device)


        self.prior = Normal(torch.zeros(ode_latent_dim).to(device), torch.ones(ode_latent_dim).to(device))
        
        self.ode_latent_dim = ode_latent_dim
        self.order = order

    def reset_parameters(self):
        modules = [self.encoder, self.decoder]
        if self.order==2:
            modules += [self.encoder_v]
        for module in modules:
            for layer in module.children():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()

    def print_summary(self):
        """Print the summary of both the models: encoder and decoder"""
        summary(self.encoder, (1, *(28,28)))
        summary(self.decoder, (1, self.ode_latent_dim))
        if self.order==2:
            summary(self.encoder_v, (1,*(28,28)))

    def save(self, encoder_path=None, decoder_path=None):
        """Save the VAE model. Both encoder and decoder and saved in different files."""
        torch.save(self.encoder.state_dict(), encoder_path)
        torch.save(self.decoder.state_dict(), decoder_path)

    def test(self, x):
        """Test the VAE model on data x. First x is encoded using encoder model, a sample is produced from then latent
        distribution and then it is passed through the decoder model."""
        self.encoder.eval()
        self.decoder.eval()
        enc_m, enc_log_var = self.encoder(x)
        z = self.encoder.sample(enc_m, enc_log_var)
        y = self.decoder(z)
        return y


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.sp = nn.Softplus()
    
    @property
    def device(self):
        return self.sp.device

    def sample(self, mu, std, L=1):
        ''' mu,std  - [N,q]
            returns - [L,N,q] if L>1 else [N,q]'''
        if std is None:
            return mu
        eps = torch.randn([L,*std.shape]).to(mu.device).to(mu.dtype).squeeze(0) # [N,q] or [L,N,q]
        return mu + std*eps

    def q_dist(self, mu_s, std_s, mu_v=None, std_v=None):
        if mu_v is not None:
            means = torch.cat((mu_s,mu_v), dim=-1)
            stds  = torch.cat((std_s,std_v), dim=-1)
        else:
            means = mu_s
            stds  = std_s

        return Normal(means, stds) #N,q

    @property
    def device(self):
        return self.sp.device


class EncoderCNN(AbstractEncoder):
    def __init__(self, task, out_distr='normal', enc_out_dim=16, n_filt=8, n_in_channels=1):
        super(EncoderCNN, self).__init__()
        self.enc_out_dim = enc_out_dim
        self.out_distr  = out_distr
        if task=='rot_mnist':
            self.cnn, in_features = build_rot_mnist_cnn_enc(n_in_channels, n_filt)
        elif task=='mov_mnist':
            self.cnn, in_features = build_mov_mnist_cnn_enc(n_in_channels, n_filt)
        else:
            raise ValueError(f'Unknown task {task}')
        self.fc1 = nn.Linear(in_features, enc_out_dim)
        if out_distr=='normal':
            self.fc2 = nn.Linear(in_features, enc_out_dim)
        
    def forward(self, X):
        ''' X - [N,T,nc,w,w] '''
        h = self.cnn(X)
        z0_mu = self.fc1(h)
        if self.out_distr=='normal':
            z0_log_sig = self.fc2(h) # N,q & N,q
            z0_log_sig = self.sp(z0_log_sig)
            return z0_mu, z0_log_sig
        else:
            return z0_mu

class PositionEncoderCNN(EncoderCNN):
    def __init__(self, task, out_distr='normal', enc_out_dim=16, n_filt=8, n_in_channels=1, T_in=1):
        super().__init__(task, out_distr=out_distr, enc_out_dim=enc_out_dim, n_filt=n_filt, n_in_channels=n_in_channels*T_in)
        self.T_in = T_in
    def forward(self,X):
        [N,T,nc,d,d] = X.shape
        X_in = X[:,:self.T_in].reshape(N, self.T_in*nc, d, d)
        return super().forward(X_in)

class VelocityEncoderCNN(EncoderCNN):
    def __init__(self, num_frames, task, out_distr='normal', enc_out_dim=16, n_filt=8, n_in_channels=1):
        super().__init__(task, out_distr=out_distr, enc_out_dim=enc_out_dim, n_filt=n_filt, n_in_channels=n_in_channels)
        self.num_frames = num_frames
    def forward(self,X):
        [N,T,nc,d,d] = X.shape
        X_in = X[:,:self.num_frames].reshape(N, self.num_frames*nc, d, d)
        return super().forward(X_in)

class EncoderRNN(AbstractEncoder):
    def __init__(self, input_dim, Tin=None, rnn_hidden=10, enc_out_dim=16, out_distr='normal'):
        super(EncoderRNN, self).__init__()
        self.enc_out_dim = enc_out_dim
        self.out_distr   = out_distr
        self.Tin         = Tin
        enc_out_dim      = enc_out_dim + enc_out_dim*(out_distr=='normal')
        self.gru = GRUEncoder(enc_out_dim, input_dim, rnn_output_size=rnn_hidden, H=50)
        
    def forward(self, x):
        x = x[:,:self.Tin]
        outputs = self.gru(x)
        if self.out_distr=='normal':
            z0_mu, z0_log_sig = outputs[:,:self.enc_out_dim,], outputs[:,self.enc_out_dim:]
            z0_log_sig = self.sp(z0_log_sig)
            return z0_mu, z0_log_sig
        return outputs

class Decoder(nn.Module):
    def __init__(self, task, dec_inp_dim, n_filt=8, H=100, distribution='bernoulli', dec_out_dim=None, act='relu'):
        super(Decoder, self).__init__()
        self.distribution = distribution
        if task=='rot_mnist':
            self.net = build_rot_mnist_cnn_dec(n_filt, dec_inp_dim)
        elif task=='mov_mnist':
            self.net = build_mov_mnist_cnn_dec(n_filt, dec_inp_dim)
        elif task=='sin' or task=='spiral' or task=='lv':
            self.net = MLP(dec_inp_dim, dec_out_dim, L=2, H=H, act=act)
            self.out_logsig = torch.nn.Parameter(torch.zeros(dec_out_dim)*0.0)
            self.sp = nn.Softplus()
        else:
            raise ValueError('Unknown task {task}')

    def forward(self, z, dims):
        inp  = z.contiguous().view([np.prod(list(z.shape[:-1])),z.shape[-1]])  # L*N*T,q  
        Xrec = self.net(inp)
        return Xrec.view(dims) # L,N,T,...
    
    @property
    def device(self):
        return next(self.parameters()).device

    def log_prob(self, X, Xhat, L=1):
        '''
        x - input [N,T,nc,d,d]   or [N,T,d]
        z - preds [L,N,T,nc,d,d] or [L,N,T,d]
        '''
        XL = X.repeat([L]+[1]*X.ndim) # L,N,T,nc,d,d or L,N,T,d
        if self.distribution == 'bernoulli':
            try:
                log_p = torch.log(Xhat)*XL + torch.log(1-Xhat)*(1-XL) # L,N,T,nc,d,d
            except:
                log_p = torch.log(EPSILON+Xhat)*XL + torch.log(EPSILON+1-Xhat)*(1-XL) # L,N,T,nc,d,d
        elif self.distribution == 'normal':
            std = self.sp(self.out_logsig)
            log_p = torch.distributions.Normal(XL,std).log_prob(Xhat)
        else:
            raise ValueError('Currently only bernoulli dist implemented')

        return log_p