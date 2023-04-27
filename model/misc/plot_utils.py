import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import torch
import os
from pathlib import Path

import matplotlib.colors as mcolors
from model.misc import io_utils
palette = list(mcolors.CSS4_COLORS.keys())
palette_tab = list(mcolors.TABLEAU_COLORS.keys())



def plot_results(plotter, \
                 tr_rec, trainset, vl_rec, validset, trace_params, \
                 ztl_tr=None, ztl_vl=None, C_tr=None, C_vl=None):

    plotter.plot_fit(trainset, tr_rec, 'tr', trace_params['iteration'])
    plotter.plot_fit(validset,  vl_rec, 'valid', trace_params['iteration'])

    if ztl_tr is not None:
        plotter.plot_latent(ztl_tr, 'tr', trace_params['iteration'])
    if ztl_vl is not None:
        plotter.plot_latent(ztl_vl, 'valid', trace_params['iteration'])

    if C_tr is not None:
        plotter.plot_C(C_tr, 'tr')
    if C_vl is not None:
        plotter.plot_C(C_vl, 'valid')

    plotter.plot_trace(trace_params)


# def plot_results_sonode(plotter, args, \
#                  tr_rec, trainset, vl_rec, validset, \
#                  trace_params):

#     plotter.plot_fit(trainset, tr_rec.unsqueeze(0), 'tr', trace_params['iteration'])
#     plotter.plot_fit(validset,  vl_rec.unsqueeze(0), 'valid', trace_params['iteration'])

#     plotter.plot_trace(args, trace_params)

class Plotter:
    def __init__(self, root, task_name):
        self.task_name    = task_name
        self.path_root  = root
        self.path_fit = os.path.join(self.path_root, 'fit')
        self.path_latents = os.path.join(self.path_root, 'latents')
        if self.task_name in ['rot_mnist', 'rot_mnist_ou', 'mov_mnist', 'bb']:
            self.plot_fit_fnc    = plot_mnist
        if self.task_name=='sin':
            self.plot_fit_fnc = plot_sin
        if self.task_name == 'lv':
            self.plot_fit_fnc = plot_2d
        self.plot_latent_fnc = plot_latent_traj


    def plot_fit(self, X, Xrec, fname='', ep=0):
        fname = self.task_name + '_fit_' + fname + '_' + str(ep) + '.png'
        fname = os.path.join(self.path_fit, fname)
        self.plot_fit_fnc(X, Xrec, fname=fname)

    def plot_latent(self, z, fname='', ep=0):
        fname = self.task_name + '_latents_' + fname + '_' + str(ep) + '.png'
        fname = os.path.join(self.path_latents, fname)
        self.plot_latent_fnc(z, fname=fname)

    def plot_trace(self, trace_params): 
        plot_training_objectives(self.path_root, trace_params)

    
    def plot_C(self, C, fname=''):
        if C is None:
            return
        else:
            fname = self.task_name + '_C_' + fname + '.png'
            fname = os.path.join(self.path_root, fname)
            C = C.mean(0) if C.ndim==4 else C
            C = C / C.pow(2).sum(-1,keepdim=True).sqrt() # N,Tinv,q
            N_,T_,q_ = C.shape
            C = C.reshape(N_*T_,q_) # NT,q
            C = (C.unsqueeze(0) * C.unsqueeze(1)).sum(-1) # NT, NT
            plt.figure(1,(12,10))
            plt.imshow(C.detach().cpu().numpy())
            plt.colorbar()
            plt.xticks([])
            plt.yticks([])
            plt.savefig(fname)
            plt.close()
        

def plot_sin_gt(X, show=False, fname='predictions.png', N=None, D=None):
    '''
         X    - [N,T,d] 
    '''
    if N is None:
        N = min(X.shape[0],6)
    if D is None:
        D = min(X.shape[-1],3)
    Xnp    = X.detach().cpu().numpy()
    print(Xnp.shape)
    nc,nr = D, N
    fig, axes = plt.subplots(nr, nc, figsize=(nc*10,nr*2), squeeze=False)
    for n in range(N):
        for d in range(D):
            axes[n,d].plot([t for t in range(X.shape[1])], Xnp[n,:,d].T,'o','', color="green", markersize=5, alpha=0.7)
    if show:
        plt.show()
    else:
        plt.savefig(fname)
        plt.close()


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
            axes[n,d].plot(Xrecnp[:,n,:,d].T, '-', color="darkblue", lw=1.2, alpha=0.9)
            axes[n,d].plot([t for t in range(Xnp.shape[1])],Xnp[n,:,d].T, 'o','', color="green", markersize=5, alpha=0.7)
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
    palette_t = palette_tab[1:]
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
        plt.plot(Xnp[n,:,0], Xnp[n,:,1], '*-', color=palette_tab[n])

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

def plot_bb(X, fname=None):
    plt.figure(1,(8,8))
    for n in range(X.shape[2]):
        plt.plot(X[0,:,n,0], X[0,:,n,1], 'o')
    plt.savefig(fname,dpi=200)
    plt.close()

def plot_bb_V(V, N=3,fname=None):
    print(V.shape)
    '''
    V: N,T,dim,dim 
    '''
    c = V.shape[-1]
    T = V.shape[1]
    plt.figure(2,(T,2*N))
    for i in range(N):
        for t in range(T):
            plt.subplot(2*N,T,i*T+t+1)
            plt.imshow(np.reshape(V[i,t],[c,c]), cmap='gray')
            plt.xticks([]); plt.yticks([])
    plt.savefig(fname)


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
            plt.plot(Qpca[l,n,:,0], Qpca[l,n,:,1], '*-', markersize=2, lw=0.5, color=palette_tab[n])
            plt.plot(Qpca[l,n,0,0], Qpca[l,n,0,1], '*', markersize=15, color=palette_tab[n])
    if q>2:
        plt.xlabel('PCA-1  ({:.2f})'.format(S[0]),fontsize=15)
        plt.ylabel('PCA-2  ({:.2f})'.format(S[1]),fontsize=15)
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig(fname)
        plt.close()


def plot_training_objectives(root, trace_params=''): 
    
    fig, axs = plt.subplots(len(trace_params["plot"]), 1, figsize=(10, 10))
    for ax,title,meter in zip(axs,trace_params["plot"].keys(),trace_params["plot"].values()):
        if meter is not None:
            ax.plot(meter.iters, meter.vals)
            ax.set_title(title)
            ax.grid()
            
            np.save(os.path.join(root, title + '.npy'), np.stack((meter.iters, meter.vals), axis=1))

    for title, meter in trace_params.items():
        if title not in ['plot', 'iteration']:
            np.save(os.path.join(root, title + '.npy'), np.stack((meter.iters, meter.vals), axis=1))

    fig.subplots_adjust()
    fname = os.path.join(root, 'optimization_trace.png')
    fig.savefig(fname, dpi=160)
    plt.close(fig) 
