import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from pathlib import Path

import matplotlib.colors as mcolors
palette = list(mcolors.TABLEAU_COLORS.keys())

def plot_results(plotter, args, ztl_tr, tr_rec, trainset, ztl_te, te_rec, \
    testset, elbo_meter, nll_meter, kl_z0_meter, inducing_kl_meter, mse_meter):

    plotter.plot_fit(trainset, tr_rec, 'tr')
    plotter.plot_fit(testset,  te_rec, 'test')

    plotter.plot_latent(ztl_tr, 'tr')
    plotter.plot_latent(ztl_te, 'test')

    plot_trace(elbo_meter, nll_meter, kl_z0_meter, inducing_kl_meter, mse_meter, args) # logpL_meter, logztL_meter, args)

class Plotter:
	def __init__(self, root, task_name):
		self.task_name    = task_name
		self.path_prefix  = root
		if self.task_name=='rot_mnist':
			self.plot_fit_fnc    = plot_mnist
		if self.task_name=='mov_mnist':
			self.plot_fit_fnc = plot_mnist
		if self.task_name=='sin':
			self.plot_fit_fnc = plot_sin
		self.plot_latent_fnc = plot_latent_traj

	def plot_fit(self, X, Xrec, fname=''):
		fname = self.task_name + '_fit_' + fname + '.png'
		fname = os.path.join(self.path_prefix, fname)
		self.plot_fit_fnc(X, Xrec, fname=fname)

	def plot_latent(self, z, fname=''):
		fname = self.task_name + '_latents_' + fname + '.png'
		fname = os.path.join(self.path_prefix, fname)
		self.plot_latent_fnc(z, fname=fname)

def plot_sin(X, Xrec, show=False, fname='predictions.png', N=None, D=None):
    ''' X    - [N,T,d] 
        Xrec - [L,N,Ttest,d]
    '''
    if N is None:
        N = min(X.shape[0],6)
    if D is None:
        D = min(X.shape[-1],3)
    Xnp    = X.detach().cpu().numpy()
    Xrecnp = Xrec.detach().cpu().numpy()
    nc,nr = D, N
    fig, axes = plt.subplots(nr, nc, figsize=(nc*10,nr*2), squeeze=False)
    for n in range(N):
        for d in range(D):
            axes[n,d].plot(Xrecnp[:,n,:,d].T, '-', color='tab:gray', lw=0.7, alpha=0.5)
            axes[n,d].plot(Xnp[n,:,d].T, '-', color='tab:blue', lw=2)
    if show:
        plt.show()
    else:
        plt.savefig(fname)
        plt.close()


def plot_mnist(X, Xrec, show=False, fname='predictions.png', N=None):
    if Xrec.ndim > X.ndim:
        Xrec = Xrec[0]
    if N is None:
        N = min(X.shape[0],10)
    Xnp    = X.detach().cpu().numpy()
    Xrecnp = Xrec.detach().cpu().numpy()
    Tdata  = X.shape[1]
    Tpred  = Xrec.shape[1]
    T = max(Tpred,Tdata)
    c = Xnp.shape[-1]
    plt.figure(2,(T,3*N))
    for i in range(N):
        for t in range(Tdata):
            plt.subplot(2*N,T,i*T*2+t+1)
            plt.imshow(np.reshape(Xnp[i,t],[c,c]), cmap='gray')
            plt.xticks([]); plt.yticks([])
        for t in range(Tpred):
            plt.subplot(2*N,T,i*T*2+t+T+1)
            plt.imshow(np.reshape(Xrecnp[i,t],[c,c]), cmap='gray')
            plt.xticks([]); plt.yticks([])
    if show:
        plt.show()
    else:
        plt.savefig(fname)
        plt.close()


def plot_latent_traj(Q, Nplot=10, show=False, fname='latents.png'): #TODO adjust for 2nd ordder (dont think it is right atm)
    [N,T,q] = Q.squeeze(0).shape 
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
        plt.plot(Qpca[n,:,0], Qpca[n,:,1], '*-', markersize=6, color=palette[n])
        plt.plot(Qpca[n,0,0], Qpca[n,0,1], '*', markersize=15, color=palette[n])
    if q>2:
        plt.xlabel('PCA-1  ({:.2f})'.format(S[0]),fontsize=15)
        plt.ylabel('PCA-2  ({:.2f})'.format(S[1]),fontsize=15)
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig(fname)
        plt.close()

def plot_trace(elbo_meter, nll_meter,  kl_z0_meter, inducing_kl_meter, mse_meter, args, make_plot=False): 
    fig, axs = plt.subplots(5, 1, figsize=(10, 10))

    titles = ["Loss (-elbo)", "Obs NLL", "KL-z0", "KL-U", "MSE"]
    meters = [elbo_meter, nll_meter,  kl_z0_meter, inducing_kl_meter, mse_meter]
    for ax,title,meter in zip(axs,titles,meters):
        ax.plot(meter.iters, meter.vals)
        ax.set_title(title)
        ax.grid()

    fig.subplots_adjust()
    if make_plot:
        plt.show()
    else:
        fig.savefig(os.path.join(args.save, 'plots/optimization_trace.png'), dpi=160)
                    # bbox_inches='tight', pad_inches=0.01)
        plt.close(fig)
        np.save(os.path.join(args.save, 'elbo.npy'), np.stack((elbo_meter.iters, elbo_meter.vals), axis=1))
        np.save(os.path.join(args.save, 'nll.npy'), np.stack((nll_meter.iters, nll_meter.vals), axis=1))
        np.save(os.path.join(args.save, 'zkl.npy'), np.stack((kl_z0_meter.iters, kl_z0_meter.vals), axis=1))
        np.save(os.path.join(args.save, 'inducingkl.npy'), np.stack((inducing_kl_meter.iters,inducing_kl_meter.vals), axis=1))
