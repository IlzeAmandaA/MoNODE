import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from pathlib import Path

import matplotlib.colors as mcolors
from model.misc import io_utils
palette = list(mcolors.TABLEAU_COLORS.keys())


def plot_results(plotter, args, ztl_tr, tr_rec, trainset, ztl_te, te_rec, \
    testset, elbo_meter, nll_meter, kl_z0_meter, inducing_kl_meter, \
        tr_mse_meter, test_mse_meter, test_elbo_meter):

    plotter.plot_fit(trainset, tr_rec, 'tr')
    plotter.plot_fit(testset,  te_rec, 'test')

    plotter.plot_latent(ztl_tr, 'tr')
    plotter.plot_latent(ztl_te, 'test')

    plot_trace(args, elbo_meter, nll_meter, kl_z0_meter, inducing_kl_meter, tr_mse_meter, test_mse_meter, test_elbo_meter) # logpL_meter, logztL_meter, args)


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
        if self.task_name == 'lv' or self.task_name =='spiral':
            self.plot_fit_fnc = plot_2d
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
    '''
         X    - [N,T,d] 
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

def plot_2d(X, Xrec, show=False, fname='predictions.png', N=None, D=None, C=2, L=None):
    ''' 
        For spiral and lv dataset (d=2)
        X    - [N,T,d] 
        Xrec - [L,N,Ttest,d]
    '''
    palette_t = palette[1:]
    if N is None:
        N = min(X.shape[0],3)
    if D is None:
        D = min(X.shape[-1],3)
    if L is None:
        L = Xrec.shape[0]
    Xnp    = X.detach().cpu().numpy()
    Xrecnp = Xrec.detach().cpu().numpy()
    fig, axs = plt.subplots(N, C, figsize=(9, 9))
    nidx = 0
    for n in range(N):
        for c in range(C):
            axs[n,c].plot(Xnp[nidx,:,0], Xnp[nidx,:,1], '-', color='tab:blue')
            for l in range(L):
                axs[n,c].plot(Xrecnp[l,nidx,:,0],Xrecnp[l,nidx,:,1], '--', color=palette_t[l])
            nidx +=1
    
    for ax in axs.flat:
        ax.label_outer()
    if show:
        plt.show()
    else:
        plt.savefig(fname)
        plt.close()

def plot_2d_origin(X, show=False, fname='predictions.png', D=None, N=None):
    ''' 
        For spiral and lv dataset (d=2)
        X    - [N,T,d] 
        Xrec - [L,N,Ttest,d]
    '''
    if N is None:
        N = min(X.shape[0],10)
    if D is None:
        D = min(X.shape[-1],3)
    
    Xnp    = X.detach().cpu().numpy()
    plt.figure(1,figsize=(9, 9))
    for n in range(N):
        plt.plot(Xnp[n,:,0], Xnp[n,:,1], '*-', color=palette[n])

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
    [L,N,T,q] = Q.shape 
    if q>2:
        Q = Q.reshape(L*N*T,q)
        U,S,V = torch.pca_lowrank(Q, q=min(q,10))
        Qpca = Q @ V[:,:2] 
        Qpca = Qpca.reshape(L,N,T,2).detach().cpu().numpy() # L,N,T,2
        S = S / S.sum()
    else:
        Qpca = Q.detach().cpu().numpy()
    plt.figure(1,(5,5))
    for n in range(Nplot):
        for l in range(L):
            plt.plot(Qpca[l,n,:,0], Qpca[l,n,:,1], '*-', markersize=2, lw=0.5, color=palette[n])
            plt.plot(Qpca[l,n,0,0], Qpca[l,n,0,1], '*', markersize=15, color=palette[n])
    if q>2:
        plt.xlabel('PCA-1  ({:.2f})'.format(S[0]),fontsize=15)
        plt.ylabel('PCA-2  ({:.2f})'.format(S[1]),fontsize=15)
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig(fname)
        plt.close()


def plot_trace(args, elbo_meter=None, nll_meter=None,  kl_z0_meter=None, inducing_kl_meter=None, tr_mse_meter=None, test_mse_meter=None, test_elbo_meter=None, make_plot=False, data_dir='log_files'): 
    fig, axs = plt.subplots(5, 1, figsize=(10, 10))

    titles = ["Loss (-elbo)", "Obs NLL", "KL-z0", "KL-U", "Train MSE"] #, "Test MSE", "Test ELBO"]
    meters = [elbo_meter, nll_meter,  kl_z0_meter, inducing_kl_meter, tr_mse_meter] #, test_mse_meter, test_elbo_meter]
    for ax,title,meter in zip(axs,titles,meters):
        if meter is not None:
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
        io_utils.makedirs(os.path.join(args.save, data_dir))
        np.save(os.path.join(args.save,data_dir,'tr_elbo.npy'), np.stack((elbo_meter.iters, elbo_meter.vals), axis=1))
        np.save(os.path.join(args.save,data_dir, 'nll.npy'), np.stack((nll_meter.iters, nll_meter.vals), axis=1))
        np.save(os.path.join(args.save,data_dir,'zkl.npy'), np.stack((kl_z0_meter.iters, kl_z0_meter.vals), axis=1))
        np.save(os.path.join(args.save,data_dir,'te_elbo.npy'), np.stack((test_elbo_meter.iters, test_elbo_meter.vals), axis=1))
        np.save(os.path.join(args.save,data_dir,'te_mse.npy'), np.stack((test_mse_meter.iters, test_mse_meter.vals), axis=1))
        if inducing_kl_meter is not None:
            np.save(os.path.join(args.save,data_dir,'inducingkl.npy'), np.stack((inducing_kl_meter.iters,inducing_kl_meter.vals), axis=1))
