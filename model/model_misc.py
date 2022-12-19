import os, time, numpy as np
import torch
from torch.distributions import kl_divergence as kl
import sys

from model.core.svgp import SVGP_Layer
from model.core.mlp import MLP
from model.core.deepgp import DeepGP
from model.core.flow import Flow
from model.core.vae import VAE
from model.core.invodevae import INVODEVAE
from model.core.sgp import SGP


def build_model(args, device, dtype):
    """
    Builds a model object of odevaegp.ODEVAEGP based on training sequence

    @param args: model setup arguments
    @return: an object of ODEVAEGP class
    """

    #differential function
    aug = args.task=='sin' and args.inv_latent_dim>0
    if aug:
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
        de = MLP(D_in, D_out, L=args.num_layers, H=args.num_hidden, act='softplus') #TODO add as parser args
    
    elif args.de == 'SGP': # does not work at all
        Z  = torch.randn(args.num_inducing, D_in)
        u_var = 'diag' if args.q_diag else 'chol'
        de = SGP(Z, D_out, kernel=args.kernel, whitened=True, u_var=u_var)
        de = de.to(device).to(dtype)

    else:
        print('Invalid Differential Euqation model specified')
        sys.exit()

    #marginal invariance
 
    if args.inv_latent_dim>0:
        # gp = DeepGP(args.D_in, args.D_out, args.num_inducing_inv)
        inv_gp = SVGP_Layer(D_in=args.inv_latent_dim, 
                    D_out=args.inv_latent_dim, #2q, q
                    M=args.num_inducing_inv,
                    S=args.num_features,
                    dimwise=args.dimwise,
                    q_diag=args.q_diag,
                    device=device,
                    dtype=dtype,
                    kernel = args.kernel)
        inv_gp = MLP(args.inv_latent_dim, args.inv_latent_dim, \
            L=args.num_layers, H=args.num_hidden, act='relu') #TODO add as parser args
        inv_gp = MLP(args.inv_latent_dim, args.inv_latent_dim, L=0, ) #TODO add as parser args

    else:
        inv_gp = None

    #continous latent ode 
    flow = Flow(diffeq=de, order=args.order, solver=args.solver, use_adjoint=args.use_adjoint)

    #encoder & decoder
    vae = VAE(task=args.task, v_frames=args.frames, n_filt=args.n_filt, ode_latent_dim=args.ode_latent_dim, 
            dec_act=args.dec_act, rnn_hidden=args.rnn_hidden, H=args.decoder_H, 
            inv_latent_dim=args.inv_latent_dim, order=args.order, device=device).to(dtype)

    #full model
    inodevae = INVODEVAE(flow = flow,
                        vae = vae,
                        inv_gp = inv_gp,
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

    if model.inv_gp is not None:
        kl_gp_2 = model.inv_gp.kl() + kl_gp
    else:
        kl_gp_2 = kl_gp

    return lhood.mean(), kl_z0.mean(), kl_gp_2 

def compute_loss(model, data, L, seed=None):
    """
    Compute loss for optimization
    @param model: a odegpvae object
    @param data: true observation sequence 
    @param L: number of MC samples
    @param Ndata: number of training data points 
    @return: loss, nll, regularizing_kl, inducing_kl
    """
    T = data.shape[1]
    in_data = data if seed==None else data[:,:seed]
    Xrec, ztL, (s0_mu, s0_logv), (v0_mu, v0_logv) = model(in_data, L, T_custom=T)
    lhood, kl_z0, kl_gp = elbo(model, data, Xrec, s0_mu, s0_logv, v0_mu, v0_logv,L)
    lhood = lhood * model.num_observations
    kl_z0 = kl_z0 * model.num_observations
    loss  = - lhood + kl_z0 + kl_gp
    mse   = torch.mean((Xrec-data)**2)
    return loss, -lhood, kl_z0, kl_gp, Xrec, ztL, mse

def freeze_pars(par_list):
    for par in par_list:
        try:
            par.requires_grad = False
        except:
            print('something wrong!')
            raise ValueError('This is not a parameter!?')

def train_model(args, invodevae, plotter, trainset, testset, logger, freeze_dyn=False):
    from model.misc import log_utils 
    from model.misc.plot_utils import plot_results
    inducing_kl_meter = log_utils.CachedRunningAverageMeter(10)
    elbo_meter   = log_utils.CachedRunningAverageMeter(10)
    nll_meter    = log_utils.CachedRunningAverageMeter(10)
    kl_z0_meter  = log_utils.CachedRunningAverageMeter(10)
    mse_meter    = log_utils.CachedRunningAverageMeter(10)
    time_meter   = log_utils.CachedAverageMeter()

    logger.info('********** Started Training **********')

    if freeze_dyn:
        freeze_pars(invodevae.flow.parameters())

    optimizer = torch.optim.Adam(invodevae.parameters(),lr=args.lr)
    begin = time.time()
    global_itr = 0
    for ep in range(args.Nepoch):
        L = 1 if ep<args.Nepoch//2 else 5 
        for itr,local_batch in enumerate(trainset):
            tr_minibatch = local_batch.to(invodevae.device) # N,T,...
            if args.task=='sin':
                [N,T] = tr_minibatch.shape[:2]
                T_  = min(T, ep//50+5)
                if T_ < T:
                    N_  = int(N*(T//T_))
                    t0s = torch.randint(0,T-T_,[N_]) 
                    tr_minibatch = tr_minibatch.repeat([N_,1,1])
                    tr_minibatch = torch.stack([tr_minibatch[n,t0:t0+T_] for n,t0 in enumerate(t0s)]) # N*ns,T//2,d
            loss, nlhood, kl_z0, kl_u, Xrec_tr, ztL_tr, tr_mse = compute_loss(invodevae, tr_minibatch, L)

            optimizer.zero_grad()
            loss.backward() 
            optimizer.step()

            #store values 
            elbo_meter.update(loss.item(), global_itr)
            nll_meter.update(nlhood.item(), global_itr)
            kl_z0_meter.update(kl_z0.item(), global_itr)
            mse_meter.update(tr_mse.item(), global_itr)
            inducing_kl_meter.update(kl_u.item(), global_itr)
            time_meter.update(time.time() - begin, global_itr)
            global_itr +=1

        with torch.no_grad():
            torch.save(invodevae.state_dict(), os.path.join(args.save, 'invodevae.pth'))
            mses = []
            for itr_test,test_batch in enumerate(testset):
                test_batch = test_batch.to(invodevae.device)
                test_elbo, nlhood, kl_z0, kl_gp, Xrec_te, ztL_te, test_mse = compute_loss(invodevae, test_batch, L=1, seed=test_batch.shape[1]//2)
                mses.append(test_mse.item())
            test_mse = np.mean(np.array(mses))
            logger.info('Epoch:{:4d}/{:4d} | tr_elbo:{:8.2f}({:8.2f}) | test_elbo {:5.3f} | test_mse:{:5.3f})'.\
                format(ep, args.Nepoch, elbo_meter.val, elbo_meter.avg, test_elbo.item(), test_mse))   

            if ep % args.plot_every==0:
                Xrec_tr, ztL_tr, _, _ = invodevae(tr_minibatch, L=args.plotL, T_custom=2*tr_minibatch.shape[1])
                Xrec_te, ztL_te, _, _ = invodevae(test_batch,   L=args.plotL, T_custom=2*test_batch.shape[1])

                plot_results(plotter, args, ztL_tr[0,:,:,:], Xrec_tr.squeeze(0), tr_minibatch, ztL_te[0,:,:,:], \
                    Xrec_te.squeeze(0), test_batch, elbo_meter, nll_meter, kl_z0_meter, inducing_kl_meter, mse_meter)
    