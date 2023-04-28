from model.core.svgp import SVGP_Layer
from model.core.mlp import MLP
from model.core.flow import Flow
from model.core.vae import VAE, SONODE_init_velocity
from model.core.inv_enc import INV_ENC
from model.core.invodevae import INVODEVAE


def build_model(args, device, dtype, params):
    """
    Builds a model object of inodevae.INVODEVAE based on training sequence

    @param args: model setup arguments
    @param device: device on which to store the model (cpu or gpu)
    @param dtype: dtype of the tensors
    @param params: dict of data properties (see config.yml)
    @return: an object of INVODEVAE class
    """

    #differential function
    aug = (args.task=='sin' or args.task=='spiral' or args.task=='lv' or args.task=='bb') and args.inv_latent_dim>0
    Nobj = 1

    if aug: # augmented dynamics
        D_in  = args.ode_latent_dim + args.inv_latent_dim
        D_out = int(args.ode_latent_dim / args.order)
    else:
        if args.task == 'mov_mnist': #multiple objects with shared dynamics
            Nobj = params[args.task]['ndigits']
            D_in = args.ode_latent_dim// Nobj
            D_out = args.ode_latent_dim// Nobj
        else:
            D_in  = args.ode_latent_dim
            D_out = int(D_in / args.order)

    # latent ode 
    if args.model == 'node':
        de = MLP(D_in, D_out, L=args.num_layers, H=args.num_hidden, act='softplus') 
    elif args.model == 'sonode':
        #de = MLP(2, 1, L=2, H=20, act='elu') #in data space 
        de = MLP(D_in, D_out, L=args.num_layers, H=args.num_hidden, act='elu') 

    flow = Flow(diffeq=de, order=args.order, solver=args.solver, use_adjoint=args.use_adjoint)

    # encoder & decoder
    if args.model == 'node':
        vae = VAE(task=args.task, v_frames=args.frames, n_filt=args.n_filt, ode_latent_dim=args.ode_latent_dim, 
            dec_act=args.dec_act, rnn_hidden=args.rnn_hidden, H=args.decoder_H, 
            inv_latent_dim=args.inv_latent_dim, T_in=args.T_in, order=args.order, device=device).to(dtype)
    elif args.model == 'sonode':
        vae = SONODE_init_velocity(dim=args.ode_latent_dim//2, nhidden=20) #match to SONODE

    # time-invariant network
    if args.inv_latent_dim>0:
        if args.task == 'bb':
            inv_enc = INV_ENC(task=args.task, inv_latent_dim=args.inv_latent_dim,
                n_filt=args.n_filt, rnn_hidden=10, T_inv=args.T_inv, vae_enc=vae.encoder, device=device).to(dtype)
        else:
            inv_enc = INV_ENC(task=args.task, inv_latent_dim=args.inv_latent_dim,
                n_filt=args.n_filt, rnn_hidden=10, T_inv=args.T_inv, vae_enc=None, device=device).to(dtype)
    else:
        inv_enc = None

    #full model
    inodevae = INVODEVAE(model = args.model,
                        flow = flow,
                        vae = vae,
                        inv_enc = inv_enc,
                        order = args.order,
                        steps = args.frames,
                        dt  = args.dt,
                        aug = aug,
                        nobj=Nobj)

    return inodevae