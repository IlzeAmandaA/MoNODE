import os, time, numpy as np
import torch
from torch.distributions import kl_divergence as kl
from torchdiffeq import odeint
import sys

from model.core.svgp import SVGP_Layer
from model.core.mlp import MLP
from model.core.flow import Flow
from model.core.vae import VAE
from model.core.inv_enc import INV_ENC
from model.core.invodevae import INVODEVAE

from model.misc import log_utils 
from model.misc.plot_utils import plot_results


def build_model(args, device, dtype):
    """
    Builds a model object of odevaegp.ODEVAEGP based on training sequence

    @param args: model setup arguments
    @return: an object of ODEVAEGP class
    """

    #differential function
    aug = (args.task=='sin' or args.task=='spiral' or args.task=='lv') and args.inv_latent_dim>0
    Nobj = 1 #TODO maybe also make parser variable that you can change if needed
    if aug: # augmented dynamics
        D_in  = args.ode_latent_dim + args.inv_latent_dim
        D_out = int(args.ode_latent_dim / args.order)
    else:
        if args.task == 'mov_mnist': #multiple objects with shared dynamics
            Nobj = 2
            D_in = args.ode_latent_dim// Nobj
            D_out = args.ode_latent_dim// Nobj
        else:
            D_in  = args.ode_latent_dim
            D_out = int(D_in / args.order)

    if args.de == 'SVGP':
        de = SVGP_Layer(D_in=D_in, 
                        D_out=D_out, #2q, q
                        M=args.num_inducing,
                        S=args.num_features,
                        dimwise=args.dimwise,
                        q_diag=args.q_diag,
                        device=device,
                        dtype=dtype,
                        kernel = args.kernel)

        de.initialize_and_fix_kernel_parameters(lengthscale_value=args.lengthscale, variance_value=args.variance, fix=False) #1.25, 0.5, 0.65 0.25
    
    elif args.de == 'MLP':
        de = MLP(D_in, D_out, L=args.num_layers, H=args.num_hidden, act='softplus') 

    if args.inv_latent_dim>0:
        if args.inv_fnc == 'SVGP':
            last_layer_gp = SVGP_Layer(D_in=args.inv_latent_dim, 
                        D_out=args.inv_latent_dim, #2q, q
                        M=args.num_inducing_inv,
                        S=args.num_features,
                        dimwise=args.dimwise,
                        q_diag=args.q_diag,
                        device=device,
                        dtype=dtype,
                        kernel = args.kernel)
        else:
            last_layer_gp = None
        
        inv_enc = INV_ENC(task=args.task, last_layer_gp=last_layer_gp, inv_latent_dim=args.inv_latent_dim,
            n_filt=args.n_filt, rnn_hidden=10, T_inv=args.T_inv, device=device).to(dtype)

    else:
        inv_enc = None

    # latent ode 
    flow = Flow(diffeq=de, order=args.order, solver=args.solver, use_adjoint=args.use_adjoint)

    # encoder & decoder
    vae = VAE(task=args.task, v_frames=args.frames, n_filt=args.n_filt, ode_latent_dim=args.ode_latent_dim, 
            dec_act=args.dec_act, rnn_hidden=args.rnn_hidden, H=args.decoder_H, 
            inv_latent_dim=args.inv_latent_dim, T_in=args.T_in, order=args.order, cnn_arch=args.cnn_arch, device=device).to(dtype)

    #full model
    inodevae = INVODEVAE(flow = flow,
                        vae = vae,
                        inv_enc = inv_enc,
                        num_observations = args.Ntrain,
                        order = args.order,
                        steps = args.frames,
                        dt  = args.dt,
                        aug = aug,
                        nobj=Nobj)

    return inodevae

def elbo(model, X, Xrec, s0_mu, s0_logv, v0_mu, v0_logv,L):
    ''' Input:
            qz_m        - latent means [N,2q]
            qz_logv     - latent logvars [N,2q]
            X           - input images [L,N,T,nc,d,d]
            Xrec        - reconstructions [L,N,T,nc,d,d]
        Returns:
            likelihood
            kl terms
    '''
    # KL reg
    q = model.vae.encoder.q_dist(s0_mu, s0_logv, v0_mu, v0_logv)
    kl_z0 = kl(q, model.vae.prior).sum(-1) #N

    #Reconstruction log-likelihood
    lhood = model.vae.decoder.log_prob(X,Xrec,L) #L,N,T,d,nc,nc
    idx   = list(np.arange(X.ndim+1)) # 0,1,2,...
    lhood = lhood.sum(idx[2:]).mean(0) #N

    # KL inudcing 
    if model.flow.odefunc.diffeq.type == 'SVGP':
        kl_gp = model.flow.kl()
    else:
        kl_gp = 0.0 * torch.zeros(1).to(X.device)

    if model.inv_enc is not None:
        kl_gp_2 = model.inv_enc.kl().to(X.device) + kl_gp
    else:
        kl_gp_2 = kl_gp

    return lhood.mean(), kl_z0.mean(), kl_gp_2 


def contrastive_loss(C):
    ''' 
    C - invariant embeddings [N,T,q] or [L,N,T,q] 
    '''
    C = C.mean(0) if C.ndim==4 else C
    C = C / C.pow(2).sum(-1,keepdim=True).sqrt() # N,Tinv,q
    N_,T_,q_ = C.shape
    C = C.reshape(N_*T_,q_) # NT,q
    Z   = (C.unsqueeze(0) * C.unsqueeze(1)).sum(-1) # NT, NT
    idx = torch.meshgrid(torch.arange(T_),torch.arange(T_))
    idxset0 = torch.cat([idx[0].reshape(-1)+ n*T_ for n in range(N_)])
    idxset1 = torch.cat([idx[1].reshape(-1)+ n*T_ for n in range(N_)])
    pos = Z[idxset0,idxset1].sum()
    return -pos, Z


def compute_loss(model, data, L, contr_loss=False, T_valid=None):
    """
    Compute loss for optimization
    @param model: a odegpvae objectb 
    @param data: true observation sequence 
    @param L: number of MC samples
    @param contr_loss: whether to compute contrastive loss or not
    @return: loss, nll, regularizing_kl, inducing_kl
    """
    T = data.shape[1]
    #in_data = data if T_valid==None else data[:,:T_valid] #for validation reconstruction on half the sequence length, see forecasting for the other half
    Xrec, ztL, (s0_mu, s0_logv), (v0_mu, v0_logv), C = model(data, L, T_custom=T)
    lhood, kl_z0, kl_gp = elbo(model, data, Xrec, s0_mu, s0_logv, v0_mu, v0_logv,L)
    if contr_loss:
        contr_learn_loss, Z_matrix = contrastive_loss(C)
    else:
        contr_learn_loss = torch.zeros_like(lhood)
        Z_matrix = torch.zeros((2,2))
        
    lhood = lhood * model.num_observations
    kl_z0 = kl_z0 * model.num_observations
    loss  = - lhood + kl_z0 + kl_gp + contr_learn_loss
    mse   = torch.mean((Xrec-data)**2)
    return loss, -lhood, kl_z0, kl_gp, Xrec, ztL, mse, contr_learn_loss, Z_matrix


def freeze_pars(par_list):
    for par in par_list:
        try:
            par.requires_grad = False
        except:
            print('something wrong!')
            raise ValueError('This is not a parameter!?')


def train_model(args, invodevae, plotter, trainset, validset, logger, freeze_dyn=False):
    inducing_kl_meter = log_utils.CachedRunningAverageMeter(0.97)
    elbo_meter  = log_utils.CachedRunningAverageMeter(0.97)
    nll_meter   = log_utils.CachedRunningAverageMeter(0.97)
    kl_z0_meter = log_utils.CachedRunningAverageMeter(0.97)
    tr_mse_meter   = log_utils.CachedRunningAverageMeter(0.97)
    contr_meter = log_utils.CachedRunningAverageMeter(0.97)
    te_mse_meter = log_utils.CachedRunningAverageMeter(0.97)
    test_elbo_meter  = log_utils.CachedRunningAverageMeter(0.97)

    logger.info('********** Started Training **********')

    if freeze_dyn:
        freeze_pars(invodevae.flow.parameters())

    ############## build the optimizer
    gp_pars  = [par for name,par in invodevae.named_parameters() if 'SVGP' in name]
    rem_pars = [par for name,par in invodevae.named_parameters() if 'SVGP' not in name]
    assert len(gp_pars)+len(rem_pars) == len(list(invodevae.parameters()))
    optimizer = torch.optim.Adam([
                    {'params': rem_pars, 'lr': args.lr},
                    {'params': gp_pars, 'lr': args.lr*10}
                    ],lr=args.lr)
    begin = time.time()
    global_itr = 0
    
    for ep in range(args.Nepoch):
        L = 1 if ep<args.Nepoch//2 else 5 
        for itr,local_batch in enumerate(trainset):
            tr_minibatch = local_batch.to(invodevae.device) # N,T,...
            if args.task=='sin' or args.task=='spiral' or args.task=='lv': #slowly increase sequence length
                [N,T] = tr_minibatch.shape[:2]
                if args.task == 'sin': #T is 50 keep sequence length short
                    ep_inc = T //args.Nepoch + 10
                    T_  = min(T, ep//ep_inc+5)
                elif args.task == 'spiral': #T is 1000 increase seqence length more
                    ep_inc = T // args.Nepoch + 1 
                    T_ = min(T, ep//ep_inc+20)
                elif args.task == 'lv': #T is 100
                    ep_inc = T//args.Nepoch + 3
                    T_ = min(T, ep//ep_inc+10)
                if T_ < T:
                    N_  = int(N*(T//T_))
                    t0s = torch.randint(0,T-T_,[N_])  #select a random initial point from the sequence
                    tr_minibatch = tr_minibatch.repeat([N_,1,1])
                    tr_minibatch = torch.stack([tr_minibatch[n,t0:t0+T_] for n,t0 in enumerate(t0s)]) # N*ns,T//2,d
            loss, nlhood, kl_z0, kl_u, Xrec_tr, ztL_tr, tr_mse, contr_learn_cost, tr_Z_matrix = \
                compute_loss(invodevae, tr_minibatch, L, contr_loss=args.contr_loss)

            optimizer.zero_grad()
            loss.backward() 
            optimizer.step()

            #store values 
            elbo_meter.update(loss.item(), global_itr)
            nll_meter.update(nlhood.item(), global_itr)
            kl_z0_meter.update(kl_z0.item(), global_itr)
            tr_mse_meter.update(tr_mse.item(), global_itr)
            contr_meter.update(contr_learn_cost.item(), global_itr)
            inducing_kl_meter.update(kl_u.item(), global_itr)
            global_itr +=1

        with torch.no_grad():
            torch.save(invodevae.state_dict(), os.path.join(args.save, 'invodevae.pth'))
            test_elbos,test_mses,lhoods = [],[],[]
            for itr_test,valid_batch in enumerate(validset):
                valid_batch = valid_batch.to(invodevae.device)
                test_elbo, nlhood, kl_z0, kl_gp, Xrec_te, ztL_te, test_mse, _, te_Z_matrix = compute_loss(invodevae, valid_batch, L=1) #, T_valid=valid_batch.shape[1]//2)
                test_elbos.append(test_elbo.item())
                test_mses.append(test_mse.item())
                lhoods.append(nlhood.item())
            test_elbo, test_mse = np.mean(np.array(test_elbos)),np.mean(np.array(test_mses))

            #update test loggers
            te_mse_meter.update(test_mse.item(), ep)
            test_elbo_meter.update(test_elbo.item(),ep)

            logger.info('Epoch:{:4d}/{:4d} | tr_elbo:{:8.2f}({:8.2f}) | test_elbo {:5.3f} | test_mse:{:5.3f} | contr_loss:{:5.3f}'.\
                format(ep, args.Nepoch, elbo_meter.val, elbo_meter.avg, test_elbo, test_mse, contr_meter.avg))   

            if ep % args.plot_every==0:
                Xrec_tr, ztL_tr = invodevae(tr_minibatch, L=args.plotL, T_custom=args.forecast_tr*tr_minibatch.shape[1])[:2]
                Xrec_te, ztL_te = invodevae(valid_batch,   L=args.plotL, T_custom=args.forecast_te*valid_batch.shape[1])[:2]

                plot_results(plotter, args, ztL_tr, Xrec_tr, tr_minibatch, ztL_te, \
                    Xrec_te, valid_batch, elbo_meter, nll_meter, kl_z0_meter, inducing_kl_meter, \
                        tr_mse_meter, te_mse_meter, test_elbo_meter, te_Z_matrix)
