import os, time, numpy as np
import torch
from torch.distributions import kl_divergence as kl
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
    aug = args.task=='sin' and args.inv_latent_dim>0
    if aug: # augmented dynamics
        D_in  = args.ode_latent_dim + args.inv_latent_dim
        D_out = int(args.ode_latent_dim / args.order)
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
            n_filt=args.n_filt, rnn_hidden=10, device=device).to(dtype)

    else:
        inv_enc = None

    # latent ode 
    flow = Flow(diffeq=de, order=args.order, solver=args.solver, use_adjoint=args.use_adjoint)

    # encoder & decoder
    vae = VAE(task=args.task, v_frames=args.frames, n_filt=args.n_filt, ode_latent_dim=args.ode_latent_dim, 
            dec_act=args.dec_act, rnn_hidden=args.rnn_hidden, H=args.decoder_H, 
            inv_latent_dim=args.inv_latent_dim, order=args.order, cnn_arch=args.cnn_arch, device=device).to(dtype)

    #full model
    inodevae = INVODEVAE(flow = flow,
                        vae = vae,
                        inv_enc = inv_enc,
                        num_observations = args.Ntrain,
                        order = args.order,
                        steps = args.frames,
                        dt  = args.dt,
                        aug = aug)

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
    ''' C - invariant embeddings [N,T,q] or [L,N,T,q] '''
    C = C.mean(0) if C.ndim==4 else C
    C = C / C.pow(2).sum(-1,keepdim=True).sqrt() # N,Tinv,q
    N_,T_,q_ = C.shape
    C = C.reshape(N_*T_,q_) # NT,q
    Z   = (C.unsqueeze(0) * C.unsqueeze(1)).sum(-1) # NT, NT
    idx = torch.meshgrid(torch.arange(T_),torch.arange(T_))
    idxset0 = torch.cat([idx[0].reshape(-1)+ n*T_ for n in range(N_)])
    idxset1 = torch.cat([idx[1].reshape(-1)+ n*T_ for n in range(N_)])
    pos = Z[idxset0,idxset1].sum()
    # Z[idxset0,idxset1] *= 0
    # neg = Z.sum() * 0.0
    # contr_learn_loss = neg-pos
    return -pos


def compute_loss(model, data, L, seed=None, contr_loss=False):
    """
    Compute loss for optimization
    @param model: a odegpvae objectb 
    @param data: true observation sequence 
    @param L: number of MC samples
    @param contr_loss: whether to compute contrastive loss or not
    @return: loss, nll, regularizing_kl, inducing_kl
    """
    T = data.shape[1]
    in_data = data if seed==None else data[:,:seed]
    Xrec, ztL, (s0_mu, s0_logv), (v0_mu, v0_logv), C = model(in_data, L, T_custom=T)
    lhood, kl_z0, kl_gp = elbo(model, data, Xrec, s0_mu, s0_logv, v0_mu, v0_logv,L)
    if contr_loss:
        contr_learn_loss = contrastive_loss(C)
    else:
        contr_learn_loss = torch.zeros_like(lhood)
    lhood = lhood * model.num_observations
    kl_z0 = kl_z0 * model.num_observations
    loss  = - lhood + kl_z0 + kl_gp + contr_learn_loss
    mse   = torch.mean((Xrec-data)**2)
    return loss, -lhood, kl_z0, kl_gp, Xrec, ztL, mse, contr_learn_loss


def freeze_pars(par_list):
    for par in par_list:
        try:
            par.requires_grad = False
        except:
            print('something wrong!')
            raise ValueError('This is not a parameter!?')


def train_model(args, invodevae, plotter, trainset, testset, logger, freeze_dyn=False):
    inducing_kl_meter = log_utils.CachedRunningAverageMeter(10)
    elbo_meter  = log_utils.CachedRunningAverageMeter(10)
    nll_meter   = log_utils.CachedRunningAverageMeter(10)
    kl_z0_meter = log_utils.CachedRunningAverageMeter(10)
    mse_meter   = log_utils.CachedRunningAverageMeter(10)
    contr_meter = log_utils.CachedRunningAverageMeter(10)
    time_meter  = log_utils.CachedAverageMeter()

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
            if args.task=='sin':
                [N,T] = tr_minibatch.shape[:2]
                T_  = min(T, ep//40+5)
                if T_ < T:
                    N_  = int(N*(T//T_))
                    t0s = torch.randint(0,T-T_,[N_]) 
                    tr_minibatch = tr_minibatch.repeat([N_,1,1])
                    tr_minibatch = torch.stack([tr_minibatch[n,t0:t0+T_] for n,t0 in enumerate(t0s)]) # N*ns,T//2,d
            loss, nlhood, kl_z0, kl_u, Xrec_tr, ztL_tr, tr_mse, contr_learn_cost = \
                compute_loss(invodevae, tr_minibatch, L, contr_loss=args.contr_loss)

            optimizer.zero_grad()
            loss.backward() 
            optimizer.step()

            #store values 
            elbo_meter.update(loss.item(), global_itr)
            nll_meter.update(nlhood.item(), global_itr)
            kl_z0_meter.update(kl_z0.item(), global_itr)
            mse_meter.update(tr_mse.item(), global_itr)
            contr_meter.update(contr_learn_cost.item(), global_itr)
            inducing_kl_meter.update(kl_u.item(), global_itr)
            time_meter.update(time.time() - begin, global_itr)
            global_itr +=1

        with torch.no_grad():
            torch.save(invodevae.state_dict(), os.path.join(args.save, 'invodevae.pth'))
            test_elbos,test_mses,lhoods = [],[],[]
            for itr_test,test_batch in enumerate(testset):
                test_batch = test_batch.to(invodevae.device)
                test_elbo, nlhood, kl_z0, kl_gp, Xrec_te, ztL_te, test_mse, _ = compute_loss(invodevae, test_batch, L=1, seed=test_batch.shape[1]//2)
                test_elbos.append(test_elbo.item())
                test_mses.append(test_mse.item())
                lhoods.append(nlhood.item())
            test_elbo, test_mse = np.mean(np.array(test_elbos)),np.mean(np.array(test_mses))
            logger.info('Epoch:{:4d}/{:4d} | tr_elbo:{:8.2f}({:8.2f}) | test_elbo {:5.3f} | test_mse:{:5.3f} | contr_loss:{:5.3f}'.\
                format(ep, args.Nepoch, elbo_meter.val, elbo_meter.avg, test_elbo, test_mse, contr_meter.avg))   

            if ep % args.plot_every==0:
                Xrec_tr, ztL_tr = invodevae(tr_minibatch, L=args.plotL, T_custom=3*tr_minibatch.shape[1])[:2]
                Xrec_te, ztL_te = invodevae(test_batch,   L=args.plotL, T_custom=2*test_batch.shape[1])[:2]

                plot_results(plotter, args, ztL_tr, Xrec_tr.squeeze(0), tr_minibatch, ztL_te, \
                    Xrec_te.squeeze(0), test_batch, elbo_meter, nll_meter, kl_z0_meter, inducing_kl_meter, mse_meter)


def train_mov_mnist(args, invodevae, plotter, trainset, testset, logger):
    from torchdiffeq import odeint
    invodevae.flow.odefunc.diffeq = MLP(args.ode_latent_dim//2, args.ode_latent_dim//2, 
        L=args.num_layers, H=args.num_hidden, act='softplus').to(invodevae.device).to(invodevae.dtype)
    elbo_meter  = log_utils.CachedRunningAverageMeter(10)
    nll_meter   = log_utils.CachedRunningAverageMeter(10)
    kl_z0_meter = log_utils.CachedRunningAverageMeter(10)
    mse_meter   = log_utils.CachedRunningAverageMeter(10)
    contr_meter = log_utils.CachedRunningAverageMeter(10)
    time_meter  = log_utils.CachedAverageMeter()

    logger.info('********** Started Training MOVING MNIST **********')

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

    def model_forward(invodevae, X, L, T_custom, nobj=2):
        [N,T] = X.shape[:2]
        if T_custom:
            T = T_custom

        # encode dynamics
        s0_mu, s0_logv = invodevae.vae.encoder(X) # N,q
        z0 = invodevae.vae.encoder.sample(s0_mu, s0_logv, L=L) # N,q or L,N,q
        z0 = z0.unsqueeze(0) if z0.ndim==2 else z0 # L,N,q
        q  = z0.shape[-1]
        z0 = z0.reshape(L,N,nobj,q//nobj) # L,N,nobj,q_
        
        # encode content (invariance)
        C = invodevae.inv_enc(X, L=L) # embeddings [L,N,T,q]
        q = C.shape[-1]
        c = C.mean(2) # time-invariant code [L,N,q]
        cL = torch.stack([c]*T,2) # [L,N,T,q]

        # sample trajectories
        ts  = invodevae.dt * torch.arange(T,dtype=torch.float).to(z0.device)
        ztL = []
        odef = lambda t,x: invodevae.flow.odefunc.diffeq(x)
        for z0_ in z0:
            # implement this
            zt_ = odeint(odef, z0_, ts, method='euler') # T,N,nobj,q_
            ztL.append(zt_)
        ztL = torch.stack(ztL) # L,T,N,nobj,q_
        ztL = ztL.permute(0,2,1,3,4) # L,N,T,nobj,q
        ztL = ztL.reshape(L,N,T,-1)

        # concat latent features and decode
        ztcL = torch.cat([ztL, cL], -1) # L,N,T,_
        Xrec = invodevae.vae.decoder(ztcL, [L,N,T,*X.shape[2:]]) # L,N,T,...

        return Xrec, ztL, (s0_mu, s0_logv), (None, None), C

    def compute_loss(model, X):
        T = X.shape[1]
        Xrec, ztL, (s0_mu, s0_logv), (v0_mu, v0_logv), C = model_forward(model, X, L, T_custom=T)

        # elbo
        q = invodevae.vae.encoder.q_dist(s0_mu, s0_logv, v0_mu, v0_logv)
        kl_z0 = kl(q, invodevae.vae.prior).sum(-1) #N

        # Reconstruction log-likelihood
        lhood = invodevae.vae.decoder.log_prob(X,Xrec,L) #L,N,T,d,nc,nc
        idx   = list(np.arange(X.ndim+1)) # 0,1,2,...
        lhood = lhood.sum(idx[2:]).mean(0) #N
        mse   = torch.mean((Xrec-X)**2)

        # contrastive learning
        contr_learn_loss = contrastive_loss(C)

        lhood = lhood.mean() * invodevae.num_observations
        kl_z0 = kl_z0.mean() * invodevae.num_observations
        elbo  = lhood - kl_z0
        loss  = -elbo + contr_learn_loss


        return loss, elbo, lhood, kl_z0, mse, contr_learn_loss


    for ep in range(args.Nepoch):
        L = 1 if ep<args.Nepoch//2 else 5 
        for itr,local_batch in enumerate(trainset):
            tr_minibatch = local_batch.to(invodevae.device) # N,T,...

            # model predictions
            loss, elbo, lhood, kl_z0, mse, contr_learn_loss = compute_loss(invodevae, tr_minibatch)

            optimizer.zero_grad()
            loss.backward() 
            optimizer.step()

            #store values 
            elbo_meter.update(loss.item(), global_itr)
            nll_meter.update(lhood.item(), global_itr)
            kl_z0_meter.update(kl_z0.item(), global_itr)
            mse_meter.update(mse.item(), global_itr)
            contr_meter.update(contr_learn_loss.item(), global_itr)
            time_meter.update(time.time() - begin, global_itr)
            global_itr +=1

        with torch.no_grad():
            torch.save(invodevae.state_dict(), os.path.join(args.save, 'invodevae.pth'))
            test_elbos,test_mses = [],[]
            for test_batch in testset:
                test_batch = test_batch.to(invodevae.device)
                _, test_elbo, _, kl_z0, test_mse, contr_learn_loss = compute_loss(invodevae, test_batch)
                test_elbos.append(test_elbo.item())
                test_mses.append(test_mse.item())
            test_elbo, test_mse = np.mean(np.array(test_elbos)),np.mean(np.array(test_mses))
            logger.info('Epoch:{:4d}/{:4d} | tr_elbo:{:8.2f}({:8.2f}) | test_elbo {:5.3f} | test_mse:{:5.3f} | contr_loss:{:5.3f}'.\
                format(ep, args.Nepoch, elbo_meter.val, elbo_meter.avg, test_elbo, test_mse, contr_meter.avg))   

            if ep % args.plot_every==0:
                Xrec_tr, ztL_tr = model_forward(invodevae, tr_minibatch, L=args.plotL, T_custom=2*tr_minibatch.shape[1])[:2]
                Xrec_te, ztL_te = model_forward(invodevae, test_batch,   L=args.plotL, T_custom=2*test_batch.shape[1])[:2]

                plot_results(plotter, args, ztL_tr, Xrec_tr.squeeze(0), tr_minibatch, ztL_te, \
                    Xrec_te.squeeze(0), test_batch, elbo_meter, nll_meter, kl_z0_meter, None, mse_meter)