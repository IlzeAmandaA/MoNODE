import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data

from scipy.io import loadmat
import numpy as np

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# prepare dataset
class Dataset(data.Dataset):
    def __init__(self, Xtr):
        self.Xtr = Xtr # N,16,784
    def __len__(self):
        return len(self.Xtr)
    def __getitem__(self, idx):
        return self.Xtr[idx]

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def __init__(self,w):
        super().__init__()
        self.w = w
    def forward(self, input):
        nc = input[0].numel()//(self.w**2)
        return input.view(input.size(0), nc, self.w, self.w)

def build_encoder(n_filt):
	return nn.Sequential(
            nn.Conv2d(1, n_filt, kernel_size=5, stride=2, padding=(2,2)), # 14,14
            nn.BatchNorm2d(n_filt),
            nn.ReLU(),
            nn.Conv2d(n_filt, n_filt*2, kernel_size=5, stride=2, padding=(2,2)), # 7,7
            nn.BatchNorm2d(n_filt*2),
            nn.ReLU(),
            nn.Conv2d(n_filt*2, n_filt*4, kernel_size=5, stride=2, padding=(2,2)),
            nn.ReLU(),
            Flatten()
        )

def build_decoder(h_dim, n_filt):
	return nn.Sequential(
            UnFlatten(4),
            nn.ConvTranspose2d(h_dim//16, n_filt*8, kernel_size=3, stride=1, padding=(0,0)),
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

def get_dataset(num_workers=0, batch_size=25, N=500, T=16, augment_data=False, device=torch.device('cpu')):
	X = loadmat(os.path.join('data','rot-3s.mat'))['X'].squeeze() # (N, 16, 784)
	Xtr   = torch.tensor(X[:N],dtype=torch.float32, device=device).view([N,T,1,28,28])
	Xtest = torch.tensor(X[N:],dtype=torch.float32, device=device).view([-1,T,1,28,28])
	if augment_data:
		Xtr   = torch.cat([Xtr,Xtr[:,1:]],1) # N,2T,1,d,d
		Xtest = torch.cat([Xtest,Xtest[:,1:]],1) # N,2T,1,d,d
		t0s_tr   = torch.randint(0,T,[Xtr.shape[0]])
		t0s_test = torch.randint(0,T,[Xtest.shape[0]])
		Xtr   = torch.stack([Xtr[i,t0:t0+T]   for i,t0 in enumerate(t0s_tr)])
		Xtest = torch.stack([Xtest[i,t0:t0+T] for i,t0 in enumerate(t0s_test)])

	# Data generators
	if num_workers>0:
	    from multiprocessing import Process, freeze_support
	    torch.multiprocessing.set_start_method('spawn', force="True")

	params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': num_workers}
	trainset = Dataset(Xtr)
	trainset = data.DataLoader(trainset, **params)
	testset  = Dataset(Xtest)
	testset  = data.DataLoader(testset, **params)
	return trainset, testset

def plot_rot_mnist(X, Xrec, show=False, fname='rot_mnist.png'):
    N = min(X.shape[0],10)
    Xnp = X.detach().cpu().numpy()
    Xrecnp = Xrec.detach().cpu().numpy()
    T,Trec = X.shape[1],Xrec.shape[1]
    Tmax   = max(T,Trec)
    plt.figure(2,(Tmax,3*N))
    for i in range(N):
        for t in range(T):
            plt.subplot(2*N,Tmax,i*Tmax*2+t+1)
            plt.imshow(np.reshape(Xnp[i,t],[28,28]), cmap='gray')
            plt.xticks([]); plt.yticks([])
        for t in range(Trec):
            plt.subplot(2*N,Tmax,i*Tmax*2+t+Tmax+1)
            plt.imshow(np.reshape(Xrecnp[i,t],[28,28]), cmap='gray')
            plt.xticks([]); plt.yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join('figs',fname))
    if show is False:
        plt.close()

def plot_latent_traj(Q, Nplot=10, show=False, fname='rot_mnist_latents.png'):
	[N,T,q] = Q.shape 
	if q>2:
		Q = Q.reshape(N*T,q)
		U,S,V = torch.pca_lowrank(Q, q=min(q,10))
		Qpca = Q @ V[:,:2] 
		Qpca = Qpca.reshape(N,T,2).detach().cpu().numpy() # N,T,2
		S = S / S.sum()
	else:
		Qpca = Q.detach().cpu().numpy()
	plt.figure(1,(5,5))
	for n in range(Nplot):
		plt.plot(Qpca[n,:,0], Qpca[n,:,1], '*-', markersize=6)
		plt.plot(Qpca[n,0,0], Qpca[n,0,1], '+', markersize=10)
	if q>2:
		plt.xlabel('PCA-1  ({:.2f})'.format(S[0]),fontsize=15)
		plt.ylabel('PCA-2  ({:.2f})'.format(S[1]),fontsize=15)
	plt.tight_layout()
	plt.savefig(os.path.join('figs',fname))
	if show is False:
		plt.close()


