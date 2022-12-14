import torch
import torch.nn as nn
from torch.distributions import Normal
from torchsummary import summary
from model.misc.torch_utils import Flatten, UnFlatten
from model.core.gru_encoder import GRUEncoder
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
    def __init__(self, task, frames=1, n_filt=8, ode_latent_dim=8, inv_latent_dim=0, device='cpu', order=1, distribution='bernoulli'):
        super(VAE, self).__init__()

        # task, out_distr='normal', enc_out_dim=16, n_filt=8, n_in_channels=1
        ### build encoder
        if task=='rot_mnist' or task=='mov_mnist':
            self.encoder = CNNEncoder(task, 'normal', ode_latent_dim//order, n_filt).to(device)
            if inv_latent_dim>0:
                self.inv_encoder = CNNEncoder(task, 'dirac', inv_latent_dim, n_filt).to(device)
            if order==2:
                self.encoder_v   = CNNEncoder(task, 'normal', ode_latent_dim//order, n_filt, frames).to(device)
        else:
            self.encoder = RNNEncoder('normal', task, ode_latent_dim//order, inv_latent_dim, n_filt).to(device)
            if inv_latent_dim>0:
                self.inv_encoder = RNNEncoder('dirac', task, ode_latent_dim//order, inv_latent_dim, n_filt).to(device)
            if order==2:
                self.encoder_v   = RNNEncoder('normal', task, ode_latent_dim//order, inv_latent_dim, n_filt).to(device)

        ### build decoder
        self.decoder = Decoder(task, ode_latent_dim//order+ inv_latent_dim, n_filt, distribution).to(device)

        self.prior =  Normal(torch.zeros(ode_latent_dim).to(device), torch.ones(ode_latent_dim).to(device))
        
        self.ode_latent_dim = ode_latent_dim
        self.order = order

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

    def sample(self, mu, std):
        if std is None:
            return mu
        eps = torch.randn_like(std)
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


class CNNEncoder(AbstractEncoder):
    def __init__(self, task, out_distr='normal', enc_out_dim=16, n_filt=8, n_in_channels=1):
        super(CNNEncoder, self).__init__()
        self.out_distr = out_distr
        if task=='rot_mnist':
            self.cnn, in_features = build_rot_mnist_cnn_enc(n_in_channels, n_filt)
        elif task=='mov_mnist':
            self.cnn, in_features = build_mov_mnist_cnn_enc(n_in_channels, n_filt)
        else:
            raise ValueError(f'Unknown task {task}')
        self.fc1 = nn.Linear(in_features, enc_out_dim)
        if out_distr=='normal':
            self.fc2 = nn.Linear(in_features, enc_out_dim)
        
    def forward(self, x):
        h = self.cnn(x)
        z0_mu = self.fc1(h)
        if self.out_distr=='normal':
            z0_log_sig = self.fc2(h) # N,q & N,q
            z0_log_sig = self.sp(z0_log_sig)
            return z0_mu, z0_log_sig
        else:
            return z0_mu

class RNNEncoder(AbstractEncoder):
    def __init__(self, input_dim, enc_out_dim=16, out_distr='normal'):
        super(RNNEncoder, self).__init__()
        out_dims = enc_out_dim if out_distr=='normal' else [enc_out_dim,enc_out_dim]
        self.net = GRUEncoder(out_dims, input_dim, rnn_output_size=20, H=50)
        
    def forward(self, x):
        if self.out_distr=='normal':
            z0_mu, z0_log_sig = self.net(x)
            z0_log_sig = self.sp(z0_log_sig)
            return z0_mu, z0_log_sig
        else:
            z0 = self.net(x)
            return z0



class Decoder(nn.Module):
    def __init__(self, task, dec_inp_dim, n_filt=8, distribution='bernoulli'):
        super(Decoder, self).__init__()
        self.distribution = distribution
        if task=='rot_mnist':
            self.fc_cnn = build_rot_mnist_cnn_dec(n_filt, dec_inp_dim)
        elif task=='mov_mnist':
            self.fc_cnn = build_mov_mnist_cnn_dec(n_filt, dec_inp_dim)
        else:
            raise ValueError(f'Unknown task {task}')

    def forward(self, x):
        #L,N,T,q = x.shape
        #s = self.fc(x.contiguous().view([L*N*T,q]) ) # N*T,q
        inp = x.contiguous().view([np.prod(list(x.shape[:-1])),x.shape[-1]])     
        return self.fc_cnn(inp)
    
    @property
    def device(self):
        return next(self.parameters()).device


    def log_prob(self, x,z, L=1):
        '''
        x           - input images [N,T,1,nc,nc]
        z           - reconstructions [L,N,T,1,nc,nc]
        '''
        XL = x.repeat([L,1,1,1,1,1]) # L,N,T,nc,d,d 
        if self.distribution == 'bernoulli':
            try:
                log_p = torch.log(z)*XL + torch.log(1-z)*(1-XL) # L,N,T,nc,d,d
            except:
                log_p = torch.log(EPSILON+z)*XL + torch.log(EPSILON+1-z)*(1-XL) # L,N,T,nc,d,d
        else:
            raise ValueError('Currently only bernoulli dist implemented')

        return log_p
