from model.core.svgp import SVGP_Layer
from model.core.mlp import MLP
from model.core.flow import Flow
from model.core.vae import VAE
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
    if args.de=='hb':
        if args.task==1:
            data_dim = 1
        elif args.task==1:
            data_dim = 1
        model = MODEL(data_dim, res=True, nhid=args.nhid, cont=True).to(device) #.to(0)

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

    # latent ode 
    flow = Flow(diffeq=de, order=args.order, solver=args.solver, use_adjoint=args.use_adjoint)

    # encoder & decoder
    vae = VAE(task=args.task, v_frames=args.frames, n_filt=args.n_filt, ode_latent_dim=args.ode_latent_dim, 
            dec_act=args.dec_act, rnn_hidden=args.rnn_hidden, H=args.decoder_H, 
            inv_latent_dim=args.inv_latent_dim, T_in=args.T_in, order=args.order, device=device).to(dtype)

    # time-invariant network
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
            n_filt=args.n_filt, rnn_hidden=10, T_inv=args.T_inv, vae_enc=vae.encoder, device=device).to(dtype)

    else:
        inv_enc = None

    #full model
    inodevae = INVODEVAE(flow = flow,
                        vae = vae,
                        inv_enc = inv_enc,
                        order = args.order,
                        steps = args.frames,
                        dt  = args.dt,
                        aug = aug,
                        nobj=Nobj)

    return inodevae