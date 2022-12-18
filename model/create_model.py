import numpy as np
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