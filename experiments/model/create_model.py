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
    if args.de == 'SVGP':
        de = SVGP_Layer(D_in=args.D_in, 
                        D_out=args.D_out, #2q, q
                        M=args.num_inducing,
                        S=args.num_features,
                        dimwise=args.dimwise,
                        q_diag=args.q_diag,
                        device=device,
                        dtype=dtype,
                        kernel = args.kernel)

        de.initialize_and_fix_kernel_parameters(lengthscale_value=args.lengthscale, variance_value=args.variance, fix=False) #1.25, 0.5, 0.65 0.25
    
    elif args.de == 'MLP':
        de = MLP(args.D_in, args.D_out, L=args.num_layers, H=args.num_hidden, act='softplus') #TODO add as parser args
    
    elif args.de == 'SGP':
        Z = torch.randn(args.num_inducing, args.D_in)
        u_var = 'diag' if args.q_diag else 'chol'
        de = SGP(Z, args.D_out, kernel=args.kernel, whitened=True, u_var=u_var)
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

    else:
        inv_gp = None

    #continous latent ode 
    flow = Flow(diffeq=de, order=args.ode, solver=args.solver, use_adjoint=args.use_adjoint)

    #encoder & decoder
    vae = VAE(frames = args.frames, n_filt=args.n_filt, ode_latent_dim=args.ode_latent_dim, 
        inv_latent_dim=args.inv_latent_dim, order= args.ode, device=device).to(dtype)

    #full model
    inodevae = INVODEVAE(flow = flow,
                        vae = vae,
                        inv_gp = inv_gp,
                        num_observations = args.Ntrain,
                        order = args.ode,
                        steps = args.frames,
                        dt = args.dt)

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
    kl_reg = kl(q, model.vae.prior).sum(-1) #N

    #Reconstruction log-likelihood
    lhood = model.vae.decoder.log_prob(X,Xrec,L) #L,N,T,d,nc,nc
    lhood = lhood.sum([2,3,4,5]).mean(0) #N

    # KL inudcing 
    if model.flow.odefunc.diffeq.type == 'SVGP':
        kl_gp = model.flow.kl()
    else:
        kl_gp = 0.0 * torch.zeros(1).to(X.device)

    if model.inv_gp is not None:
        kl_gp_2 = model.inv_gp.kl() + kl_gp
    else:
        kl_gp_2 = kl_gp

    return lhood.mean(), kl_reg.mean(), kl_gp_2 


def compute_loss(model, data, L):
    """
    Compute loss for optimization
    @param model: a odegpvae object
    @param data: true observation sequence 
    @param L: number of MC samples
    @param Ndata: number of training data points 
    @return: loss, nll, regularizing_kl, inducing_kl
    """
    Xrec, ztL, (s0_mu, s0_logv), (v0_mu, v0_logv) = model(data,L)
    lhood, kl_reg, kl_gp = elbo(model, data, Xrec, s0_mu, s0_logv, v0_mu, v0_logv,L)
    loss = - (lhood - kl_reg) * model.num_observations + kl_gp
    return loss, -lhood, kl_reg, kl_gp, Xrec, ztL  

def compute_MSE(X, Xrec):
    assert list(X.shape) == list(Xrec.shape), f'incorrect shapes X: {list(X.shape)}, X_Rec: {list(Xrec.shape)}'
    return torch.mean((Xrec-X)**2)